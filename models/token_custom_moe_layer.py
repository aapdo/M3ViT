r"""
Adaption to act as the MLP layer using an MoE MLP layer in transformer.
"""
import math
import torch
import torch.nn as nn
from fmoe.layers import FMoE, _fmoe_general_global_forward
from fmoe.linear import FMoELinear
from functools import partial
import tree
import torch
import torch.nn as nn

from fmoe.functions import prepare_forward, ensure_comm
from fmoe.functions import MOEScatter, MOEGather
from fmoe.functions import AllGather, Slice
from fmoe.gates import NaiveGate

from models.gate_funs.noisy_gate import NoisyGate
from models.gate_funs.noisy_gate_vmoe import NoisyGate_VMoE
from models.gate_funs.token_noisy_gate_vmoe import TokenNoisyGate_VMoE

from pdb import set_trace
import numpy as np

class _Expert(nn.Module):
    r"""
    An expert using 2 FMoELinear modules to speed up the computation of experts
    within one worker.
    """

    def __init__(self, num_expert, d_model, d_hidden, activation, rank=0):
        super().__init__()
        self.htoh4 = FMoELinear(num_expert, d_model, d_hidden, bias=True, rank=rank)
        self.h4toh = FMoELinear(num_expert, d_hidden, d_model, bias=True, rank=rank)
        self.activation = activation

    def forward(self, inp, fwd_expert_count):
        r"""
        First expand input to 4h (the hidden size is variable, but is called h4
        for convenience). Then perform activation. Finally shirink back to h.
        """
        x = self.htoh4(inp, fwd_expert_count)
        x = self.activation(x)
        x = self.h4toh(x, fwd_expert_count)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., norm_layer= partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        # out_features = out_features or in_features
        # hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.norm = norm_layer(out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = self.norm(x)
        return x

class RouterSelector(nn.Module):
    def __init__(self, d_model, temperature=1.0, hard=False):
        super().__init__()
        # Output dimension is 2: [task-specific, shared]
        self.w_gate = nn.Parameter(
            torch.zeros(d_model, 2), requires_grad=True
        )
        self.temperature = temperature
        self.hard = hard  # Whether to use hard selection in training

        self.reset_parameters()

    def reset_parameters(self):
        # Approach is the same as in torch.nn.Linear
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L88
        torch.nn.init.kaiming_uniform_(self.w_gate, a=math.sqrt(5))

    def forward(self, inp, task_id=None):
        """
        Returns selection probabilities/decisions for using shared_gate.

        Args:
            inp: Input tensor [B, N, D] or [B*N, D]

        Returns:
            Tensor [B, N] or [B*N]: probability or binary selection (0 or 1)
                0 = use task-specific gate
                1 = use shared_gate
        """
        shape_input = list(inp.shape)
        original_shape = shape_input[:-1]  # [B, N] or [B*N]
        channel = shape_input[-1]
        inp = inp.reshape(-1, channel)  # [B*N, D]

        # Compute logits for [task-specific, shared]
        logits = inp @ self.w_gate  # [B*N, 2]

        # Apply Gumbel-Softmax
        if self.training:
            # During training: use soft or hard based on self.hard
            probs = torch.nn.functional.gumbel_softmax(
                logits, tau=self.temperature, hard=self.hard, dim=-1
            )
        else:
            # During inference: always use hard selection
            probs = torch.nn.functional.gumbel_softmax(
                logits, tau=self.temperature, hard=True, dim=-1
            )

        # Return probability/selection for shared_gate (index 1)
        output = probs[:, 1]  # [B*N]

        # Reshape to original shape
        if len(original_shape) > 1:
            output = output.reshape(*original_shape)  # [B, N]

        return output

class TokenFMoETransformerMLP(FMoE):
    r"""
    A complete MoE MLP module in a Transformer block.
    * `activation` is the activation function to be used in MLP in each expert.
    * `d_hidden` is the dimension of the MLP layer.
    """

    def __init__(
        self,
        num_expert=32,
        d_model=1024,
        d_gate=1024,
        d_hidden=4096,
        activation=torch.nn.GELU(),
        expert_dp_comm="none",
        expert_rank=0,
        gate=NaiveGate,
        world_size=1,
        top_k=2,
        vmoe_noisy_std=1,
        gate_task_specific_dim=-1,
        multi_gate=False,
        num_experts_pertask = -1,
        num_tasks = -1,
        **kwargs
    ):
        super().__init__(num_expert=num_expert, d_model=d_model, gate=gate, world_size=world_size, top_k=top_k, **kwargs)
        self.our_d_gate = d_gate
        self.our_d_model = d_model

        self.num_expert = num_expert
        self.tot_experts = num_expert * world_size
        self.num_experts_pertask = num_experts_pertask
        self.num_tasks = num_tasks

        self.experts = _Expert(
            num_expert, d_model, d_hidden, activation, rank=expert_rank
        )
        self.gate_task_specific_dim = gate_task_specific_dim
        self.multi_gate=multi_gate
        if gate_task_specific_dim<0:
            d_gate = d_model
        else:
            d_gate = d_model+gate_task_specific_dim
        print('multi_gate',self.multi_gate)

        if gate == TokenNoisyGate_VMoE:
            if self.multi_gate:
                self.gate = nn.ModuleList([
                    gate(d_gate, num_expert, world_size, top_k,
                    noise_std=vmoe_noisy_std,num_experts_pertask=self.num_experts_pertask, num_tasks=self.num_tasks)
                    for i in range(num_tasks)])
            else:
                self.gate = gate(d_gate, num_expert, world_size, top_k,
                noise_std=vmoe_noisy_std,num_experts_pertask = self.num_experts_pertask, num_tasks = self.num_tasks)

            # Initialize router_selector and shared_gate
            self.router_selector = RouterSelector(d_model=d_model, temperature=1.0, hard=False)
            self.shared_gate = gate(d_gate, num_expert, world_size, top_k,
                noise_std=vmoe_noisy_std,num_experts_pertask = self.num_experts_pertask, num_tasks = self.num_tasks)

        else:
            raise ValueError("TokenFMoETransformerMLP only supports TokenNoisyGate_VMoE")
        self.mark_parallel_comm(expert_dp_comm)

    @property
    def get_router_selector(self):
        return self.router_selector

    def forward(self, inp: torch.Tensor, gate_inp=None, task_id = None, task_specific_feature = None, selector_output=None):
        r"""
        This module wraps up the FMoE module with reshape, residual and layer
        normalization.
        """
        if gate_inp is None:
            gate_inp = inp

        original_shape = inp.shape
        # inp -> (B*T, d_model)
        inp = inp.reshape(-1, self.d_model)

        gate_channel = gate_inp.shape[-1]
        gate_inp = gate_inp.reshape(-1, gate_channel)


        if (task_id is not None) and (task_specific_feature is not None):
            assert self.multi_gate is False
            size = gate_inp.shape[0]
            gate_inp = torch.cat((gate_inp,task_specific_feature.repeat(size,1)),dim=-1)


        output, clean_logits, noisy_logits, noise_stddev, top_logits, gates = self.forward_moe(gate_inp=gate_inp, moe_inp=inp, task_id=task_id, selector_output=selector_output)
        return output.reshape(original_shape), clean_logits, noisy_logits, noise_stddev, top_logits, gates

    def forward_moe(self, gate_inp, moe_inp, task_id=None, selector_output=None):
        r"""
        The FMoE module first computes gate output, and then conduct MoE forward
        according to the gate.  The score of the selected gate given by the
        expert is multiplied to the experts' output tensors as a weight.
        """
        moe_inp_batch_size = tree.flatten(
            tree.map_structure(lambda tensor: tensor.shape[0], moe_inp)
        )
        assert all(
            [batch_size == moe_inp_batch_size[0] for batch_size in moe_inp_batch_size]
        ), "MoE inputs must have the same batch size"

        if self.world_size > 1:
            def ensure_comm_func(tensor):
                ensure_comm(tensor, self.moe_group)

            tree.map_structure(ensure_comm_func, moe_inp)
            tree.map_structure(ensure_comm_func, gate_inp)

        # Use selector_output to determine whether to use shared_gate or task-specific gate
        if selector_output is not None and hasattr(self, 'shared_gate'):
            # selector_output: [B, N] where value > 0.5 = use shared_gate, <= 0.5 = use task-specific gate
            # Flatten selector_output to match gate_inp shape [B*N]
            selector_flat = selector_output.reshape(-1)  # [B*N]

            # Create masks for shared and task-specific tokens
            # Use threshold 0.5 for continuous values from Gumbel-Softmax
            shared_mask = (selector_flat > 0.5)  # [B*N]
            task_mask = (selector_flat <= 0.5)   # [B*N]

            # Extract tokens based on masks
            shared_tokens = gate_inp[shared_mask]  # [num_shared, d_model]
            task_tokens = gate_inp[task_mask]      # [num_task, d_model]

            # Initialize output tensors
            gate_top_k_idx = torch.zeros(gate_inp.shape[0], self.top_k, dtype=torch.long, device=gate_inp.device)
            gate_score = torch.zeros(gate_inp.shape[0], self.top_k, device=gate_inp.device)

            # Initialize tensors for gate info (same shape as gate_inp batch size)
            clean_logits = torch.zeros(gate_inp.shape[0], self.tot_experts, device=gate_inp.device)
            noisy_logits = torch.zeros(gate_inp.shape[0], self.tot_experts, device=gate_inp.device)
            noise_stddev = torch.zeros(gate_inp.shape[0], self.tot_experts, device=gate_inp.device)
            # calculate topk + 1 that will be needed for the noisy gates            
            top_logits = torch.zeros(gate_inp.shape[0], self.top_k + 1, device=gate_inp.device)
            gates = torch.zeros(gate_inp.shape[0], self.tot_experts, device=gate_inp.device)

            # Route shared tokens through shared_gate
            if shared_tokens.shape[0] > 0:
                (shared_idx, shared_score), s_clean, s_noisy, s_noise_std, s_top, s_gates = self.shared_gate(shared_tokens)
                gate_top_k_idx[shared_mask] = shared_idx
                gate_score[shared_mask] = shared_score
                clean_logits[shared_mask] = s_clean
                noisy_logits[shared_mask] = s_noisy
                noise_stddev[shared_mask] = s_noise_std
                top_logits[shared_mask] = s_top
                gates[shared_mask] = s_gates

            # Route task-specific tokens through task-specific gate
            if task_tokens.shape[0] > 0:
                if (task_id is not None) and self.multi_gate:
                    (task_idx, task_score), t_clean, t_noisy, t_noise_std, t_top, t_gates = self.gate[task_id](task_tokens)
                else:
                    (task_idx, task_score), t_clean, t_noisy, t_noise_std, t_top, t_gates = self.gate(task_tokens, task_id=task_id)
                gate_top_k_idx[task_mask] = task_idx
                gate_score[task_mask] = task_score
                clean_logits[task_mask] = t_clean
                noisy_logits[task_mask] = t_noisy
                noise_stddev[task_mask] = t_noise_std
                top_logits[task_mask] = t_top
                gates[task_mask] = t_gates
        else:
            # Original behavior: all tokens use the same gate
            if (task_id is not None) and self.multi_gate:
                (gate_top_k_idx, gate_score), clean_logits, noisy_logits, noise_stddev, top_logits, gates = self.gate[task_id](gate_inp)
            else:
                (gate_top_k_idx, gate_score), clean_logits, noisy_logits, noise_stddev, top_logits, gates = self.gate(gate_inp, task_id=task_id)


        fwd = _fmoe_general_global_forward(
            moe_inp, gate_top_k_idx, self.expert_fn, self.num_expert, self.world_size
        )


        def view_func(tensor):
            dim = tensor.shape[-1]
            tensor = tensor.view(-1, self.top_k, dim)
            return tensor

        moe_outp = tree.map_structure(view_func, fwd)

        gate_score = gate_score.view(-1, 1, self.top_k)

        def bmm_func(tensor):
            dim = tensor.shape[-1]
            tensor = torch.bmm(gate_score, tensor).reshape(-1, dim)
            return tensor

        moe_outp = tree.map_structure(bmm_func, moe_outp)

        moe_outp_batch_size = tree.flatten(
            tree.map_structure(lambda tensor: tensor.shape[0], moe_outp)
        )
        assert all(
            [batch_size == moe_outp_batch_size[0] for batch_size in moe_outp_batch_size]
        ), "MoE outputs must have the same batch size"
        return moe_outp, clean_logits, noisy_logits, noise_stddev, top_logits, gates
