"""
RouterState and bitmask utilities for MoE routing.

This module provides:
1. RouterState dataclass to store previous MoE block selector information
2. Bitmask-based utilities for efficient shared/reuse mask computation
3. Simplified aggregation and reuse checking logic
"""

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class RouterState:
    """
    Stores the router selector state from a previous MoE block.

    This state is used by the next MoE block to:
    1. Determine which tokens should be aggregated after attention
    2. (Phase 2) Determine which MoE computations can be reused

    Attributes:
        shared_bits: [B, N] int64 bitmask where each bit represents whether
                     a task uses shared gate. Bit t is 1 if task t has
                     selector_output > 0.5
        shared_selector: [T, B, N] float, optional. The raw selector outputs
                        for soft aggregation or debugging
        reuse_bits: [B, N] int64 bitmask, optional. Bitmask for next block
                    indicating which tasks can potentially reuse computation.
                    This is curr_bits from current block, passed to next block
                    where it becomes prev_bits for reuse detection.
    """
    shared_bits: torch.Tensor  # [B, N] int64 bitmask
    shared_selector: torch.Tensor | None = None  # [T, B, N] float
    reuse_bits: torch.Tensor | None = None  # [B, N] int64 - for next block


def selector_to_bits(selector_tbN: torch.Tensor, thr: float = 0.5) -> torch.Tensor:
    """
    Convert selector outputs to a bitmask representation.

    Args:
        selector_tbN: [T, B, N] float tensor where each value represents
                     the probability/selection of using shared gate for
                     each task at each position
        thr: Threshold for determining shared gate usage (default: 0.5)

    Returns:
        bits: [B, N] int64 tensor where each position's value is a bitmask
              indicating which tasks use shared gate at that position.
              Example: If tasks 0, 2, 3 use shared gate, bits = 0b00001101

    Example:
        >>> selector = torch.tensor([[[0.7, 0.3]], [[0.2, 0.8]], [[0.6, 0.1]]])  # [3,1,2]
        >>> bits = selector_to_bits(selector, thr=0.5)
        >>> # Result: bits[0,0] = 0b101 (tasks 0,2), bits[0,1] = 0b010 (task 1)

    TODO (Phase 2): Hard threshold (>0.5) has "flipping risk"
        - Selector values near 0.5 (e.g., 0.4999 ↔ 0.5001) can easily flip
          shared_bits, causing unstable aggregation decisions
        - This becomes more critical in Phase 2 when reuse/skip logic depends on bits
        - Consider improvements:
          1. Use threshold with margin (e.g., thr_in=0.6, thr_out=0.4)
          2. Implement soft aggregation instead of hard binary masks
          3. Add temporal smoothing or EMA for selector stability
    """
    assert selector_tbN.dim() == 3, f"Expected 3D tensor [T,B,N], got {selector_tbN.dim()}D"
    T, B, N = selector_tbN.shape
    device = selector_tbN.device

    # Determine which positions use shared gate
    shared_mask = (selector_tbN > thr)  # [T, B, N] bool

    # Convert to bitmask by OR-ing bit-shifted masks
    bits = torch.zeros((B, N), device=device, dtype=torch.int64)
    for t in range(T):
        bits |= (shared_mask[t].to(torch.int64) << t)

    return bits


def popcount_bits(bits_bN: torch.Tensor, num_tasks: int) -> torch.Tensor:
    """
    Count the number of set bits (popcount) in each position of the bitmask.

    Args:
        bits_bN: [B, N] int64 bitmask tensor
        num_tasks: Total number of tasks (determines how many bits to check)

    Returns:
        count: [B, N] int64 tensor with the number of set bits at each position

    Example:
        >>> bits = torch.tensor([[0b101, 0b111]])  # [1, 2]
        >>> count = popcount_bits(bits, num_tasks=3)
        >>> # Result: count = [[2, 3]] (2 bits set, 3 bits set)
    """
    device = bits_bN.device
    pos = torch.arange(num_tasks, device=device, dtype=torch.int64)  # [T]

    # Extract each bit position: [B, N, T]
    expanded = (bits_bN.unsqueeze(-1) >> pos) & 1

    # Sum across tasks dimension to get popcount
    return expanded.sum(dim=-1)  # [B, N]


def compute_masks(prev_bits: torch.Tensor, curr_bits: torch.Tensor, num_tasks: int):
    """
    Compute aggregation and reuse masks from previous and current selector bits.

    This function implements the paper's logic:
    - S_j^l: shared task set at position j in layer l (curr_bits)
    - S_j^{l-1}: shared task set at position j in layer l-1 (prev_bits)
    - U_j^l: reusable task set = S_j^{l-1} ∩ S_j^l (reuse_bits)

    Args:
        prev_bits: [B, N] int64 bitmask from previous MoE block
        curr_bits: [B, N] int64 bitmask from current MoE block
        num_tasks: Total number of tasks

    Returns:
        tuple of:
            agg_needed: [B, N] bool, True where aggregation is needed (≥2 tasks shared)
            reuse_possible: [B, N] bool, True where reuse is possible (≥2 tasks in intersection)
            reuse_bits: [B, N] int64, bitmask of reusable tasks (prev & curr)

    Usage (Phase 1):
        agg_needed is used to determine which positions need aggregation.
        reuse_possible is logged for statistics but not used for actual compute reuse.

    Usage (Phase 2):
        reuse_possible will be used to skip MoE computation and copy results.
    """
    # Compute intersection: tasks that were shared in prev AND curr
    reuse_bits = prev_bits & curr_bits

    # Count tasks at each position
    curr_shared_count = popcount_bits(curr_bits, num_tasks)  # [B, N]
    reuse_count = popcount_bits(reuse_bits, num_tasks)       # [B, N]

    # Aggregation needed: ≥2 tasks use shared gate
    agg_needed = curr_shared_count >= 2  # [B, N] bool

    # Reuse possible (Phase 2): ≥2 tasks are in the intersection
    reuse_possible = reuse_count >= 2    # [B, N] bool

    return agg_needed, reuse_possible, reuse_bits


def bits_to_task_masks(bits_bN: torch.Tensor, num_tasks: int) -> list:
    """
    Convert bitmask to a list of per-task boolean masks.

    Args:
        bits_bN: [B, N] int64 bitmask
        num_tasks: Total number of tasks

    Returns:
        masks: List of T boolean tensors, each of shape [B, N]
               masks[t] is True where task t is in the shared set

    Example:
        >>> bits = torch.tensor([[0b101]])  # [1, 1], tasks 0 and 2
        >>> masks = bits_to_task_masks(bits, num_tasks=3)
        >>> # masks[0] = [[True]], masks[1] = [[False]], masks[2] = [[True]]
    """
    masks = []
    for t in range(num_tasks):
        mask = ((bits_bN >> t) & 1).bool()  # [B, N]
        masks.append(mask)
    return masks


def bits_to_multihot(bits_bN: torch.Tensor, num_tasks: int) -> torch.Tensor:
    """
    Convert bitmask to multi-hot encoding for shared gate embedding.

    This is used to create task set embeddings for the shared gate.
    Each position (b, n) gets a multi-hot vector indicating which tasks
    share that position.

    Args:
        bits_bN: [B, N] int64 bitmask where bit t is 1 if task t shares this position
        num_tasks: Total number of tasks

    Returns:
        multihot: [B, N, T] float tensor, multi-hot encoding of shared tasks

    Example:
        >>> bits = torch.tensor([[0b101, 0b011]])  # [1, 2]
        >>> mh = bits_to_multihot(bits, num_tasks=3)
        >>> # mh[0, 0] = [1, 0, 1] (tasks 0, 2)
        >>> # mh[0, 1] = [1, 1, 0] (tasks 0, 1)
    """
    device = bits_bN.device
    pos = torch.arange(num_tasks, device=device, dtype=torch.int64)  # [T]

    # Extract each bit: [B, N, T]
    multihot = ((bits_bN.unsqueeze(-1) >> pos) & 1).float()

    return multihot
