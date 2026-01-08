r"""
Adaption to act as the MLP layer using an MoE MLP layer in transformer.
"""
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

class FMoETransformerMLP(FMoE):
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

        if gate == NoisyGate:
            if self.multi_gate:
                self.gate = nn.ModuleList([
                    gate(d_gate, num_expert, world_size, top_k,
                    return_decoupled_activation=False, regu_experts_fromtask = False,
                    num_experts_pertask = self.num_experts_pertask,num_tasks = self.num_tasks, regu_sem=False,sem_force = False)
                    for i in range(self.our_d_gate-self.our_d_model)])
            else:
                self.gate = gate(d_gate, num_expert, world_size, top_k,
                return_decoupled_activation=False, regu_experts_fromtask = False,
                num_experts_pertask = self.num_experts_pertask,num_tasks = self.num_tasks, regu_sem=False,sem_force = False)
        elif gate == NoisyGate_VMoE:
            if self.multi_gate:
                self.gate = nn.ModuleList([
                    gate(d_gate, num_expert, world_size, top_k,
                    return_decoupled_activation=False,
                    noise_std=vmoe_noisy_std,regu_experts_fromtask = False,
                    num_experts_pertask=self.num_experts_pertask, num_tasks=self.num_tasks,regu_sem=False,sem_force = False, regu_subimage=False)
                    for i in range(self.our_d_gate-self.our_d_model)])
            else:
                self.gate = gate(d_gate, num_expert, world_size, top_k,
                return_decoupled_activation=False,
                noise_std=vmoe_noisy_std,regu_experts_fromtask = False,
                num_experts_pertask = self.num_experts_pertask, num_tasks = self.num_tasks,regu_sem=False,sem_force = False, regu_subimage=False)
        elif gate == TokenNoisyGate_VMoE:
            if self.multi_gate:
                self.gate = nn.ModuleList([
                    gate(d_gate, num_expert, world_size, top_k,
                    noise_std=vmoe_noisy_std,num_experts_pertask=self.num_experts_pertask, num_tasks=self.num_tasks)
                    for i in range(num_tasks)])
            else:
                self.gate = gate(d_gate, num_expert, world_size, top_k,
                noise_std=vmoe_noisy_std,num_experts_pertask = self.num_experts_pertask, num_tasks = self.num_tasks)
            

        else:
            raise ValueError("No such gating type")
        self.mark_parallel_comm(expert_dp_comm)

    def forward(self, inp: torch.Tensor, gate_inp=None, task_id = None, task_specific_feature = None, sem=None):
        r"""
        This module wraps up the FMoE module with reshape, residual and layer
        normalization.
        """
        if gate_inp is None:
            gate_inp = inp
        
        original_shape = inp.shape
        inp = inp.reshape(-1, self.d_model)

        gate_channel = gate_inp.shape[-1]
        gate_inp = gate_inp.reshape(-1, gate_channel)
        # print('task_id, task_specific_feature',task_id, task_specific_feature)
        if (task_id is not None) and (task_specific_feature is not None):
            assert self.multi_gate is False
            size = gate_inp.shape[0]
            gate_inp = torch.cat((gate_inp,task_specific_feature.repeat(size,1)),dim=-1)
        output = self.forward_moe(gate_inp=gate_inp, moe_inp=inp, task_id=task_id, sem=sem)
        return output.reshape(original_shape)


    def forward_moe(self, gate_inp, moe_inp, task_id=None, sem=None):
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


        if (task_id is not None) and self.multi_gate:
            # print('in custom moe_layer,task_id',task_id)
            gate_top_k_idx, gate_score = self.gate[task_id](gate_inp)
        else:
            gate_top_k_idx, gate_score = self.gate(gate_inp, task_id=task_id,sem=sem)


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
        return moe_outp
