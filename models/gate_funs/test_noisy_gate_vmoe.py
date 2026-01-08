r"""
Noisy gate for gshard and switch
for testing checkpointing
"""
from fmoe.gates.base_gate import BaseGate

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import math
import numpy as np
from collections import Counter
from pdb import set_trace

class NoisyGate_VMoE(BaseGate):
    def __init__(self, d_model, num_expert, world_size, top_k=2, noise_std=1, no_noise=False,
                 num_experts_pertask=-1,num_tasks=-1):
        super().__init__(num_expert, world_size)
        self.w_gate = nn.Parameter(
            torch.zeros(d_model, self.tot_expert), requires_grad=True
        )

        self.top_k = top_k
        self.no_noise = no_noise
        self.noise_std = noise_std

        self.softmax = nn.Softmax(1)

        self.activation = None
        self.select_idx = None
        self.num_experts_pertask = num_experts_pertask
        self.num_tasks = num_tasks
        self.patch_size = 16
        
        self.reset_parameters()

    def reset_parameters(self):
        # Approach is the same as in torch.nn.Linear
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L88

        torch.nn.init.kaiming_uniform_(self.w_gate, a=math.sqrt(5))

    def forward(self, inp, task_id=None,sem=None):
        shape_input = list(inp.shape)
        # print(shape_input)
        channel = shape_input[-1]
        other_dim = shape_input[:-1]
        inp = inp.reshape(-1, channel)

        clean_logits = inp @ self.w_gate
        raw_noise_stddev = self.noise_std / self.tot_expert
        noise_stddev = raw_noise_stddev * self.training

        if self.no_noise:
            noise_stddev *= 0
        noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)

        if self.select_idx is not None:
            assert len(self.select_idx) >= self.top_k
            noisy_logits = noisy_logits[:, self.select_idx]

        logits = noisy_logits

        if self.select_idx is not None and len(self.select_idx) == self.top_k:
            top_k_gates, top_k_indices = logits.topk(
                min(self.top_k, self.tot_expert), dim=1
            )

            return (
                top_k_indices,
                top_k_gates,
            )

        # calculate topk + 1 that will be needed for the noisy gates
        logits = self.softmax(logits)
        top_logits, top_indices = logits.topk(
            min(self.top_k + 1, self.tot_expert), dim=1
        )

        top_k_logits = top_logits[:, : self.top_k]
        top_k_indices = top_indices[:, : self.top_k]
        top_k_gates = top_k_logits

        # Todo: requires_grad=True 삭제 테스트
        # zeros = torch.zeros_like(logits)
        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_logits)

        # for prune
        # self.activation = logits.reshape(other_dim + [-1,]).contiguous()

        # print("top_k_indices are {}".format(top_k_indices))

        top_k_indices = top_k_indices.reshape(other_dim + [self.top_k]).contiguous()
        top_k_gates = top_k_gates.reshape(other_dim + [self.top_k]).contiguous()
        # print('top_k_indices',top_k_indices.shape,top_k_gates.shape)
        return (
            (top_k_indices, top_k_gates),
            clean_logits,
            noisy_logits,
            noise_stddev,
            top_logits,
            gates
        ) 

    def get_activation(self, clear=True):
        activation = self.activation
        if clear:
            self.activation = None
        return activation

    @property
    def has_activation(self):
        return self.activation is not None
