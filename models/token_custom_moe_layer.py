r"""
Adaption to act as the MLP layer using an MoE MLP layer in transformer.
"""
import torch
import torch.nn as nn
from fmoe.layers import FMoE, _fmoe_general_global_forward
from fmoe.linear import FMoELinear
from functools import partial
import tree

from fmoe.functions import ensure_comm
from fmoe.gates import NaiveGate

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

class TokenFMoETransformerMLP(FMoE):
    r"""
    A simplified MoE MLP module in a Transformer block.
    Only handles expert computation - gate routing is done at Block level.

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
        world_size=1,
        top_k=2,
        **kwargs
    ):
        # Pass a dummy gate to parent FMoE (we don't use it)
        super().__init__(num_expert=num_expert, d_model=d_model, gate=NaiveGate, world_size=world_size, top_k=top_k, **kwargs)
        self.our_d_model = d_model
        self.num_expert = num_expert

        self.experts = _Expert(
            num_expert, d_model, d_hidden, activation, rank=expert_rank
        )
        self.mark_parallel_comm(expert_dp_comm)

    def forward(self, inp: torch.Tensor, gate_top_k_idx: torch.Tensor, gate_score: torch.Tensor):
        r"""
        Execute MoE forward with pre-computed gate routing.

        Args:
            inp: Input tensor [B, N, D]
            gate_top_k_idx: Expert indices from gate [B*N, top_k]
            gate_score: Expert scores from gate [B*N, top_k]

        Returns:
            output: MoE output [B, N, D]
        """
        original_shape = inp.shape

        # inp -> (B*N, d_model)
        moe_inp = inp.reshape(-1, self.d_model)

        output = self.forward_moe(moe_inp, gate_top_k_idx, gate_score)
        return output.reshape(original_shape)

    def forward_moe(self, moe_inp, gate_top_k_idx, gate_score):
        r"""
        Execute MoE computation with pre-computed gate results.

        Args:
            moe_inp: [B*N, D] MoE input
            gate_top_k_idx: [B*N, top_k] expert indices
            gate_score: [B*N, top_k] expert scores
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

        fwd = _fmoe_general_global_forward(
            moe_inp, gate_top_k_idx, self.expert_fn, self.num_expert, self.world_size
        )

        def view_func(tensor):
            dim = tensor.shape[-1]
            tensor = tensor.view(-1, self.top_k, dim)
            return tensor

        moe_outp = tree.map_structure(view_func, fwd)

        gate_score_view = gate_score.view(-1, 1, self.top_k)

        def bmm_func(tensor):
            dim = tensor.shape[-1]
            tensor = torch.bmm(gate_score_view, tensor).reshape(-1, dim)
            return tensor

        moe_outp = tree.map_structure(bmm_func, moe_outp)

        moe_outp_batch_size = tree.flatten(
            tree.map_structure(lambda tensor: tensor.shape[0], moe_outp)
        )
        assert all(
            [batch_size == moe_outp_batch_size[0] for batch_size in moe_outp_batch_size]
        ), "MoE outputs must have the same batch size"

        return moe_outp
