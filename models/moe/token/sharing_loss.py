"""
Sharing Regularization Loss (Total Method §6).

L_share = λ * max(0, S² - Σ_t S_t²)

where:
  S   = number of neutral positions (|I_ℓ^{(0)}|)
  S_t = number of neutral positions where task t participates
"""

import torch
import torch.nn as nn


class SharingRegularizationLoss(nn.Module):
    """
    Encourages concentrated sharing patterns rather than diffuse/uniform sharing.

    When Σ_t S_t² ≥ S², sharing is concentrated (same tasks repeatedly share),
    which is beneficial for attention cost. The loss penalizes the opposite case.
    """

    def __init__(self, lambda_share: float = 0.01):
        super().__init__()
        self.lambda_share = lambda_share

    def forward(self, shared_bits: torch.Tensor, num_tasks: int) -> torch.Tensor:
        """
        Args:
            shared_bits: [B, N] int64 bitmask. Bit t is set if task t
                         participates in neutral at that position.
            num_tasks: T

        Returns:
            Scalar loss tensor (0 if sharing is already concentrated or no sharing).
        """
        if self.lambda_share <= 0.0:
            return shared_bits.new_tensor(0.0)

        # S: number of neutral positions (any bit set)
        neutral_mask = (shared_bits != 0)  # [B, N]
        S = neutral_mask.float().sum()

        if S.item() == 0:
            return shared_bits.new_tensor(0.0)

        # S_t: per-task participation count
        S_t_sq_sum = shared_bits.new_tensor(0.0, dtype=torch.float32)
        for t in range(num_tasks):
            S_t = ((shared_bits >> t) & 1).float().sum()
            S_t_sq_sum = S_t_sq_sum + S_t * S_t

        # L_share = λ * max(0, S² - Σ_t S_t²)
        loss = self.lambda_share * torch.clamp(S * S - S_t_sq_sum, min=0.0)

        return loss
