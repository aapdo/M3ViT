"""
Shareability prediction and aggregation modules (Total Method §3).

Extracted from vision_transformer_moe.py for modularity.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ShareabilityPredictor(nn.Module):
    """
    Gumbel-Softmax based shareability predictor (§3).

    Determines whether each token should use the shared (neutral) or
    task-specific (private) path. Input is token representation concatenated
    with task embedding; output is a 2-way (shared, private) distribution.
    """

    def __init__(self, d_model, d_task_emb=0, temperature=1.0, hard=False):
        super().__init__()
        self.d_model = d_model
        self.d_task_emb = d_task_emb
        d_input = d_model + d_task_emb if d_task_emb > 0 else d_model

        self.w_gate = nn.Parameter(
            torch.zeros(d_input, 2), requires_grad=True
        )
        self.temperature = temperature
        self.hard = hard

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.w_gate, a=math.sqrt(5))

    def forward(self, inp, task_emb=None):
        """
        Args:
            inp: [B, N, D] or [B*N, D]
            task_emb: [E], [B, E], or [B, N, E]

        Returns:
            [B, N] or [B*N]: shared probability (0=task-specific, 1=shared)
        """
        shape_input = list(inp.shape)
        original_shape = shape_input[:-1]
        channel = shape_input[-1]
        inp_flat = inp.reshape(-1, channel)
        BN = inp_flat.shape[0]

        assert not (self.d_task_emb > 0 and task_emb is None), (
            "ShareabilityPredictor requires task_emb when d_task_emb > 0 "
            "(check task_emb_T_E wiring)."
        )

        if task_emb is not None:
            if task_emb.dim() == 1:
                task_emb_flat = task_emb.unsqueeze(0).expand(BN, -1)
            elif task_emb.dim() == 2:
                B = shape_input[0] if len(shape_input) > 1 else 1
                N = BN // B
                task_emb_flat = task_emb.unsqueeze(1).expand(B, N, -1).reshape(BN, -1)
            else:
                task_emb_flat = task_emb.reshape(BN, -1)
            gate_inp = torch.cat([inp_flat, task_emb_flat], dim=-1)
        else:
            gate_inp = inp_flat

        logits = gate_inp @ self.w_gate

        if self.training:
            probs = F.gumbel_softmax(logits, tau=self.temperature, hard=self.hard, dim=-1)
        else:
            probs = F.gumbel_softmax(logits, tau=self.temperature, hard=True, dim=-1)

        output = probs[:, 1]

        if len(original_shape) > 1:
            output = output.reshape(*original_shape)

        return output


class Aggregation(nn.Module):
    """Aggregates results from multiple tasks at shared positions."""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, task_outputs, curr_shared_masks, aggregation_mask):
        """
        Args:
            task_outputs: dict {task_id: [B, N, D]}
            curr_shared_masks: list of [B, N] bool, len = T
            aggregation_mask: [B, N] bool

        Returns:
            aggregated: [B, N, D] (valid only where aggregation_mask == True)
        """
        if not aggregation_mask.any():
            return None

        outputs = torch.stack(
            [task_outputs[t] for t in range(len(curr_shared_masks))],
            dim=0
        )  # [T, B, N, D]

        shared_mask = torch.stack(curr_shared_masks, dim=0)  # [T, B, N]
        valid_mask = shared_mask & aggregation_mask.unsqueeze(0)  # [T, B, N]
        valid_mask_f = valid_mask.unsqueeze(-1).float()  # [T, B, N, 1]

        summed = (outputs * valid_mask_f).sum(dim=0)  # [B, N, D]
        count = valid_mask_f.sum(dim=0)  # [B, N, 1]

        aggregated = summed / (count + self.eps)
        return aggregated
