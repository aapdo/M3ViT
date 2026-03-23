# Sparsely-Gated Mixture-of-Experts Layers.
# See "Outrageously Large Neural Networks"
# https://arxiv.org/abs/1701.06538
#
# Author: David Rau
#
# The code is based on the TensorFlow implementation:
# https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/utils/expert_utils.py


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .parallel_experts import ParallelExperts
from .gates import BaseGate, NoisyGate_VMoE

@torch.jit.script
def compute_gating(k: int, probs: torch.Tensor, top_k_gates: torch.Tensor, top_k_indices: torch.Tensor):
    """Compute gating for expert routing.

    Args:
        k: number of experts per token
        probs: full probability matrix [B*N, num_experts]
        top_k_gates: top-k gate values [B*N, k]
        top_k_indices: top-k expert indices [B*N, k]

    Returns:
        batch_gates: gate values sorted by expert [num_nonzero]
        batch_index: original token index for each routed token [num_nonzero]
        expert_size: number of tokens routed to each expert [num_experts]
        gates: full gate matrix [B*N, num_experts]
        index_sorted_experts: sorted indices [num_nonzero]
    """
    zeros = torch.zeros_like(probs)
    gates = zeros.scatter(1, top_k_indices, top_k_gates)

    # Flatten for processing
    top_k_gates_flat = top_k_gates.flatten()  # [B*N*k]
    top_k_indices_flat = top_k_indices.flatten()  # [B*N*k]

    # Find nonzero gates (softmax output should always be > 0, but check anyway)
    nonzeros = top_k_gates_flat.nonzero().squeeze(-1)  # [num_nonzero]

    # Get corresponding expert indices
    top_k_experts_nonzero = top_k_indices_flat[nonzeros]

    # Sort by expert ID for efficient batched processing
    _, _index_sorted_experts = top_k_experts_nonzero.sort(0)

    # Count tokens per expert
    expert_size = (gates > 0).long().sum(0)

    # Get sorted indices into the flattened arrays
    index_sorted_experts = nonzeros[_index_sorted_experts]

    # Compute original token index: for flattened [B*N*k], token index is index // k
    batch_index = index_sorted_experts.div(k, rounding_mode='trunc')

    # Get gate values in sorted order
    batch_gates = top_k_gates_flat[index_sorted_experts]

    return batch_gates, batch_index, expert_size, gates, index_sorted_experts



class MoE(nn.Module):

    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self, input_size, head_size, num_experts, k,
                 bias=False, gating_activation=None,
                 activation=None, noisy_gating=True, usage_mem=10000,
                 gate=None):
        super(MoE, self).__init__()

        if gate is None:
            raise ValueError("gate parameter is required. Please provide a gate instance (e.g., NoisyGate_VMoE)")

        self.noisy_gating = noisy_gating
        self.num_experts = num_experts
        self.input_size = input_size
        self.head_size = head_size
        self.bias = bias
        self.experts = ParallelExperts(num_experts, input_size, head_size, bias)
        self.output_experts = ParallelExperts(num_experts, head_size, input_size, bias)
        self.k = min(k, self.num_experts)
        self.activation = activation

        # NOTE: Removed per-module _scatter_buf to avoid VRAM accumulation
        # Multiple MoE modules would each hold large buffers, increasing total VRAM
        # PyTorch CUDA caching allocator efficiently reuses memory for new_zeros() calls

        # Use the provided gate
        self.gate = gate

    def extra_repr(self):
        return 'k={}, noisy_gating={}'.format(self.k, self.noisy_gating)

    def forward(self, x, skip_mask=None, sample_topk=0, multiply_by_gates=True):
        """Forward pass through MoE layer.

        Args:
            x: input tensor [bsz, length, emb_size]
            skip_mask: optional mask to skip certain tokens
            sample_topk: number of experts to sample (unused, kept for backward compatibility)
            multiply_by_gates: whether to multiply expert outputs by gate values

        Returns:
            y: output tensor [bsz, length, emb_size]
            loss: gate loss (0 for backward compatibility)
        """
        bsz, length, emb_size = x.size()
        x = x.reshape(-1, emb_size)

        # Use the gate object to compute routing
        top_k_indices, top_k_gates = self.gate(x)

        # Reconstruct probs from top_k for compute_gating
        probs = torch.zeros(x.size(0), self.num_experts, device=x.device, dtype=x.dtype)
        probs.scatter_(1, top_k_indices, top_k_gates)

        batch_gates, batch_index, expert_size, gates, index_sorted_experts = \
            compute_gating(self.k, probs, top_k_gates, top_k_indices)

        # Store routing information
        self.expert_size = expert_size
        self.index_sorted_experts = index_sorted_experts
        self.batch_index = batch_index
        self.batch_gates = batch_gates

        expert_inputs = x[self.batch_index]
        h = self.experts(expert_inputs, self.expert_size)
        h = self.activation(h)
        expert_outputs = self.output_experts(h, self.expert_size)

        if multiply_by_gates:
            expert_outputs = expert_outputs * self.batch_gates[:, None]

        # Create scatter buffer via CUDA allocator (efficient reuse across calls)
        # Avoids per-module buffer accumulation: N modules * large buffer = high VRAM
        y = expert_outputs.new_zeros(bsz * length, self.input_size)
        y = y.index_add(0, self.batch_index, expert_outputs)
        y = y.view(bsz, length, self.input_size)

        return y

    def map(self, x, skip_mask=None, sample_topk=0):
        """Map inputs through experts.

        Args:
            x: tensor shape [batch_size, length, input_size]
            skip_mask: optional mask to skip certain tokens
            sample_topk: number of experts to sample (unused, kept for backward compatibility)

        Returns:
            y: tensor shape [batch_size, length, k, head_size]
            loss: gate loss (0 for backward compatibility)
        """
        bsz, length, emb_size = x.size()
        x = x.reshape(-1, emb_size)

        # Use the gate object to compute routing
        top_k_indices, top_k_gates = self.gate(x)

        # Reconstruct probs from top_k for compute_gating
        probs = torch.zeros(x.size(0), self.num_experts, device=x.device, dtype=x.dtype)
        probs.scatter_(1, top_k_indices, top_k_gates)

        batch_gates, batch_index, expert_size, gates, index_sorted_experts = \
            compute_gating(self.k, probs, top_k_gates, top_k_indices)

        # Store routing information
        self.expert_size = expert_size
        self.index_sorted_experts = index_sorted_experts
        self.batch_index = batch_index
        self.batch_gates = batch_gates

        expert_inputs = x[self.batch_index]
        expert_outputs = self.experts(expert_inputs, self.expert_size)

        zeros = torch.zeros((bsz * length * self.k, self.head_size),
            dtype=expert_outputs.dtype, device=expert_outputs.device)
        y = zeros.index_add(0, self.index_sorted_experts, expert_outputs)
        y = y.view(bsz, length, self.k, -1)

        return y, 0

    def reduce(self, x, multiply_by_gates=True):
        bsz, length, k, emb_size = x.size()
        x = x.view(-1, emb_size)

        expert_inputs = x[self.index_sorted_experts]
        expert_outputs = self.output_experts(expert_inputs, self.expert_size)

        if multiply_by_gates:
            expert_outputs = expert_outputs * self.batch_gates[:, None]

        zeros = torch.zeros((bsz * length, self.input_size), 
            dtype=expert_outputs.dtype, device=expert_outputs.device)
        y = zeros.index_add(0, self.batch_index, expert_outputs)
        y = y.view(bsz, length, self.input_size)
        return y


class TaskMoE(MoE):

    """Call a Sparsely gated mixture of experts layer with 1-layer Feed-Forward networks as experts.
    Args:
    input_size: integer - size of the input
    output_size: integer - size of the input
    num_experts: an integer - number of experts
    hidden_size: an integer - hidden size of the experts
    noisy_gating: a boolean
    k: an integer - how many experts to use for each batch element
    """

    def __init__(self, input_size, head_size, num_experts, k, task_num=9,
                 noisy_gating=True, gating_activation=None, use_aux_loss=True,
                 gates=None, **kwargs):
        if gates is None:
            raise ValueError("gates parameter is required. Please provide a list of gate instances (e.g., [NoisyGate_VMoE(...) for _ in range(task_num)])")

        if len(gates) != task_num:
            raise ValueError(f"Number of gates ({len(gates)}) must match task_num ({task_num})")

        self.task_num = task_num
        self.use_aux_loss = use_aux_loss

        # Pass the first gate to the parent MoE class
        super(TaskMoE, self).__init__(input_size, head_size, num_experts, k,
                                       noisy_gating=noisy_gating,
                                       gating_activation=gating_activation,
                                       gate=gates[0], **kwargs)

        # Override parent's single gate with task-specific gates
        self.gate = nn.ModuleList(gates)


    def forward(self, x, task_bh, skip_mask=None, sample_topk=0, multiply_by_gates=True):
        """Forward pass through TaskMoE layer with task-specific gating.

        Args:
            x: input tensor [bsz, length, emb_size]
            task_bh: task index for selecting task-specific gate
            skip_mask: optional mask to skip certain tokens
            sample_topk: number of experts to sample (unused, kept for backward compatibility)
            multiply_by_gates: whether to multiply expert outputs by gate values

        Returns:
            y: output tensor [bsz, length, emb_size]
        """

        bsz, length, emb_size = x.size()
        x = x.reshape(-1, emb_size)

        # Use task-specific gate
        top_k_indices, top_k_gates = self.gate[task_bh](x)

        # Reconstruct probs from top_k for compute_gating and aux loss
        probs = torch.zeros(x.size(0), self.num_experts, device=x.device, dtype=x.dtype)
        probs.scatter_(1, top_k_indices, top_k_gates)

        batch_gates, batch_index, expert_size, gates, index_sorted_experts = \
            compute_gating(self.k, probs, top_k_gates, top_k_indices)

        # Detach routing indices (used only for indexing, no gradients needed)
        # but KEEP batch_gates attached for gradient flow to gating network
        self.expert_size = expert_size.detach()
        self.index_sorted_experts = index_sorted_experts.detach()
        self.batch_index = batch_index.detach()
        self.batch_gates = batch_gates  # DO NOT detach! Gradient must flow to router


        expert_inputs = x[self.batch_index]
        h = self.experts(expert_inputs, self.expert_size)
        h = self.activation(h)
        expert_outputs = self.output_experts(h, self.expert_size)

        if multiply_by_gates:
            expert_outputs = expert_outputs * self.batch_gates[:, None]

        # Create scatter buffer via CUDA allocator (efficient reuse across calls)
        # Avoids per-module buffer accumulation: N modules * large buffer = high VRAM
        y = expert_outputs.new_zeros(bsz * length, self.input_size)
        y = y.index_add(0, self.batch_index, expert_outputs)
        y = y.view(bsz, length, self.input_size)

        return y

    def map(self, x, task_bh, skip_mask=None, sample_topk=0):
        """Map inputs through experts for MoEAttention.

        Args:
            x: tensor shape [batch_size, length, input_size]
            task_bh: task index for selecting task-specific gate
            skip_mask: optional mask to skip certain tokens
            sample_topk: number of experts to sample (unused, kept for backward compatibility)

        Returns:
            y: tensor shape [batch_size, length, k, head_size]
            moe_stats: tuple of (importance, load, probs, expert_size, logits) for aux loss
        """
        bsz, length, emb_size = x.size()
        x = x.reshape(-1, emb_size)

        # Use task-specific gate
        top_k_indices, top_k_gates = self.gate[task_bh](x)

        # Reconstruct probs from top_k for compute_gating and aux loss
        probs = torch.zeros(x.size(0), self.num_experts, device=x.device, dtype=x.dtype)
        probs.scatter_(1, top_k_indices, top_k_gates)

        batch_gates, batch_index, expert_size, gates, index_sorted_experts = \
            compute_gating(self.k, probs, top_k_gates, top_k_indices)

        # Detach routing indices (used only for indexing, no gradients needed)
        # but KEEP batch_gates attached for gradient flow to gating network
        self.expert_size = expert_size.detach()
        self.index_sorted_experts = index_sorted_experts.detach()
        self.batch_index = batch_index.detach()
        self.batch_gates = batch_gates  # DO NOT detach! Gradient must flow to router

        expert_inputs = x[self.batch_index]
        expert_outputs = self.experts(expert_inputs, self.expert_size)

        zeros = torch.zeros((bsz * length * self.k, self.head_size),
            dtype=expert_outputs.dtype, device=expert_outputs.device)
        y = zeros.index_add(0, self.index_sorted_experts, expert_outputs)
        y = y.view(bsz, length, self.k, -1)

        return y

    def reduce(self, x, multiply_by_gates=True):
        bsz, length, k, emb_size = x.size()
        x = x.view(-1, emb_size)

        expert_inputs = x[self.index_sorted_experts]
        expert_outputs = self.output_experts(expert_inputs, self.expert_size)

        if multiply_by_gates:
            expert_outputs = expert_outputs * self.batch_gates[:, None]

        zeros = torch.zeros((bsz * length, self.input_size), 
            dtype=expert_outputs.dtype, device=expert_outputs.device)
        y = zeros.index_add(0, self.batch_index, expert_outputs)
        y = y.view(bsz, length, self.input_size)
        return y