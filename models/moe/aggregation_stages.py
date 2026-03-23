"""
Aggregation stage for MoE architecture.

This module provides a modular aggregation stage that can be inserted
at different points in the MoE block pipeline.

The same AggregationStage class can be used for:
- Post-attention aggregation (uses prev_shared_masks)
- Post-MLP aggregation (uses curr_shared_masks)
"""

import torch
import torch.nn as nn


class AggregationStage(nn.Module):
    """
    Generic aggregation stage for MoE blocks.

    This stage aggregates outputs from multiple tasks at positions where
    multiple tasks share the same routing decision (shared gate usage).

    Can be used at different points in the MoE block:
    1. After attention, before norm2 (using prev block's routing state)
    2. After MLP (using current block's routing state)
    """

    def __init__(self, aggregation_module, num_tasks: int):
        """
        Args:
            aggregation_module: An Aggregation() module that implements
                               forward(task_outputs, masks, agg_needed_mask)
            num_tasks: Total number of tasks
        """
        super().__init__()
        self.agg = aggregation_module
        self.num_tasks = num_tasks

    def forward(self, outs: dict, shared_masks: list, agg_needed_mask: torch.Tensor):
        """
        Apply aggregation to task outputs based on routing state.

        Args:
            outs: Dict {task_id: tensor [B, N, C]} containing task outputs
            shared_masks: List of T boolean tensors [B, N]
                         shared_masks[t] is True where task t used shared gate
            agg_needed_mask: [B, N] bool, True where ≥2 tasks used shared gate
                            (aggregation is needed at these positions)

        Returns:
            outs: Dict {task_id: tensor [B, N, C]} with aggregated values
                  at positions where aggregation was needed

        Note:
            For post-attention aggregation: shared_masks should be prev_shared_masks
            For post-MLP aggregation: shared_masks should be curr_shared_masks
        """
        if (agg_needed_mask is None) or (not agg_needed_mask.any()):
            return outs

        # Call the aggregation module
        # aggregation_module.forward returns [B, N, C] aggregated features
        aggregated = self.agg(outs, shared_masks, agg_needed_mask)

        if aggregated is None:
            return outs

        # Update outputs: replace values at aggregation positions
        # Only update tasks that had shared_masks[t] = True
        for t in range(self.num_tasks):
            # Positions to update: where aggregation is needed AND task t was shared
            update_mask = agg_needed_mask & shared_masks[t]  # [B, N]

            if update_mask.any():
                m = update_mask.unsqueeze(-1)  # [B, N, 1] for broadcasting
                outs[t] = torch.where(m, aggregated, outs[t])

        return outs
