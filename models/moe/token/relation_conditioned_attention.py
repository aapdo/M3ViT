"""
Task-Conditioned Attention with Relation-Conditioned Expert Gating (Total Method §4).

Implements:
  - BranchEmbedding: learnable embeddings for branch indices (0=neutral, 1..T=tasks)
  - RelationRouter: sparse top-k routing conditioned on (query_branch, key_branch) pair
  - ExpertProjectionPool: shared Q/K/V expert matrices mixed by router weights
  - TaskConditionedAttention: full branch-aware attention with:
      * Task branch attention (§4.2): private queries → private + neutral keys
      * Neutral branch attention (§4.3): neutral queries → neutral + participant private keys
      * Relation-conditioned expert gating (§4.4): 4 relation types per task
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BranchEmbedding(nn.Module):
    """
    Learnable branch embeddings (§4.4).

    Branch 0 = neutral, branches 1..T = task-specific.
    """

    def __init__(self, num_branches: int, embed_dim: int):
        """
        Args:
            num_branches: T+1 (neutral + T tasks)
            embed_dim: branch embedding dimension
        """
        super().__init__()
        self.embed = nn.Embedding(num_branches, embed_dim)
        nn.init.normal_(self.embed.weight, std=0.02)

    def forward(self, branch_id: torch.Tensor) -> torch.Tensor:
        """
        Args:
            branch_id: integer tensor of any shape, values in [0, num_branches)
        Returns:
            Tensor of shape [*branch_id.shape, embed_dim]
        """
        return self.embed(branch_id)


class RelationRouter(nn.Module):
    """
    Sparse top-k router conditioned on relation (a→b) (§4.4).

    Input: concatenation of query-branch and key-branch embeddings.
    Output: sparse expert weights (top-k selected, re-normalized).
    """

    def __init__(self, branch_embed_dim: int, num_experts: int, top_k: int):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.linear = nn.Linear(2 * branch_embed_dim, num_experts)
        nn.init.kaiming_uniform_(self.linear.weight, a=math.sqrt(5))
        nn.init.zeros_(self.linear.bias)

    def forward(self, f_a: torch.Tensor, f_b: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_a: query branch embedding [branch_embed_dim]
            f_b: key branch embedding [branch_embed_dim]

        Returns:
            expert_weights: [num_experts] sparse weights (top-k non-zero, sums to 1)
        """
        # Concat branch embeddings
        inp = torch.cat([f_a, f_b], dim=-1)  # [2 * branch_embed_dim]
        logits = self.linear(inp)  # [num_experts]

        # Dense softmax then top-k sparsification
        dense_weights = F.softmax(logits, dim=-1)  # [num_experts]

        if self.top_k >= self.num_experts:
            return dense_weights

        topk_vals, topk_idx = torch.topk(dense_weights, self.top_k)
        # Re-normalize selected experts
        topk_vals = topk_vals / (topk_vals.sum() + 1e-9)

        sparse_weights = torch.zeros_like(dense_weights)
        sparse_weights.scatter_(0, topk_idx, topk_vals)

        return sparse_weights


class ExpertProjectionPool(nn.Module):
    """
    Shared expert pool for Q/K/V projections per attention head (§4.4).

    Each expert e has weight matrices W_e^Q, W_e^K, W_e^V.
    The effective projection for a given relation is the weighted sum of experts.
    """

    def __init__(self, num_experts: int, dim: int, head_dim: int):
        """
        Args:
            num_experts: E experts in the pool
            dim: input dimension (model dim)
            head_dim: output dimension per head
        """
        super().__init__()
        self.num_experts = num_experts
        self.dim = dim
        self.head_dim = head_dim

        # Expert weights: [E, D, head_dim] for each of Q, K, V
        self.expert_q = nn.Parameter(torch.empty(num_experts, dim, head_dim))
        self.expert_k = nn.Parameter(torch.empty(num_experts, dim, head_dim))
        self.expert_v = nn.Parameter(torch.empty(num_experts, dim, head_dim))

        self._init_weights()

    def _init_weights(self):
        for p in [self.expert_q, self.expert_k, self.expert_v]:
            # Fan-in based init per expert
            for e in range(self.num_experts):
                nn.init.xavier_uniform_(p[e])

    def get_effective_projection(
        self, expert_weights: torch.Tensor, proj_type: str
    ) -> torch.Tensor:
        """
        Compute effective projection matrix as weighted sum of experts.

        Args:
            expert_weights: [E] sparse weights from RelationRouter
            proj_type: 'q', 'k', or 'v'

        Returns:
            effective_W: [D, head_dim]
        """
        if proj_type == 'q':
            experts = self.expert_q
        elif proj_type == 'k':
            experts = self.expert_k
        elif proj_type == 'v':
            experts = self.expert_v
        else:
            raise ValueError(f"Unknown proj_type: {proj_type}")

        # experts: [E, D, head_dim], expert_weights: [E]
        # effective = sum_e(w_e * W_e) = einsum('e, edh -> dh', weights, experts)
        return torch.einsum('e, edh -> dh', expert_weights, experts)


class TaskConditionedAttention(nn.Module):
    """
    Task-Conditioned Attention with Relation-Conditioned Expert Gating (§4.2-4.4).

    Implements:
      - Task branch attention: private queries attend to private + neutral keys
      - Neutral branch attention: neutral queries attend to neutral + participant private keys
      - Relation-conditioned Q/K/V projections via shared expert pools
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_experts_per_head: int = 4,
        expert_top_k: int = 2,
        num_tasks: int = 2,
        branch_embed_dim: int = 32,
        qkv_bias: bool = False,
        qk_scale: float | None = None,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_tasks = num_tasks
        self.scale = qk_scale or self.head_dim ** -0.5
        self.num_experts = num_experts_per_head
        self.expert_top_k = expert_top_k

        # Branch embeddings: 0=neutral, 1..T=task branches
        self.branch_embed = BranchEmbedding(num_tasks + 1, branch_embed_dim)

        # Per-head routers (Q/K/V each)
        self.routers_q = nn.ModuleList([
            RelationRouter(branch_embed_dim, num_experts_per_head, expert_top_k)
            for _ in range(num_heads)
        ])
        self.routers_k = nn.ModuleList([
            RelationRouter(branch_embed_dim, num_experts_per_head, expert_top_k)
            for _ in range(num_heads)
        ])
        self.routers_v = nn.ModuleList([
            RelationRouter(branch_embed_dim, num_experts_per_head, expert_top_k)
            for _ in range(num_heads)
        ])

        # Per-head expert projection pools
        self.expert_pools = nn.ModuleList([
            ExpertProjectionPool(num_experts_per_head, dim, self.head_dim)
            for _ in range(num_heads)
        ])

        # Optional bias for Q/K/V (added after expert projection)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.k_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.k_bias = None
            self.v_bias = None

        # Output projection
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def _compute_effective_projections(
        self, relations: list[tuple[int, int]], device: torch.device
    ) -> dict:
        """
        Pre-compute effective W_Q, W_K, W_V for each unique relation type.

        Args:
            relations: list of (branch_a, branch_b) pairs
            device: target device

        Returns:
            dict mapping (a, b) -> {
                'W_Q': [H, D, head_dim],
                'W_K': [H, D, head_dim],
                'W_V': [H, D, head_dim],
            }
        """
        cache = {}
        for (a, b) in relations:
            if (a, b) in cache:
                continue

            f_a = self.branch_embed(torch.tensor(a, device=device))
            f_b = self.branch_embed(torch.tensor(b, device=device))

            W_Q_list = []
            W_K_list = []
            W_V_list = []

            for h in range(self.num_heads):
                pool = self.expert_pools[h]
                w_q = self.routers_q[h](f_a, f_b)
                w_k = self.routers_k[h](f_a, f_b)
                w_v = self.routers_v[h](f_a, f_b)

                W_Q_list.append(pool.get_effective_projection(w_q, 'q'))
                W_K_list.append(pool.get_effective_projection(w_k, 'k'))
                W_V_list.append(pool.get_effective_projection(w_v, 'v'))

            cache[(a, b)] = {
                'W_Q': torch.stack(W_Q_list, dim=0),  # [H, D, head_dim]
                'W_K': torch.stack(W_K_list, dim=0),
                'W_V': torch.stack(W_V_list, dim=0),
            }

        return cache

    def _project_tokens(
        self, x: torch.Tensor, W: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Project tokens using pre-computed effective weight.

        Args:
            x: [*, D]
            W: [H, D, head_dim]
            bias: [D] optional

        Returns:
            [*, H, head_dim]
        """
        # x @ W per head: einsum('...d, hdk -> ...hk', x, W)
        out = torch.einsum('...d, hdk -> ...hk', x, W)
        if bias is not None:
            # Reshape bias [D] -> [H, head_dim] and add
            out = out + bias.view(self.num_heads, self.head_dim)
        return out

    def _task_branch_attention(
        self,
        x_task: torch.Tensor,
        private_mask: torch.Tensor,
        neutral_mask: torch.Tensor,
        task_id: int,
        proj_cache: dict,
    ) -> torch.Tensor:
        """
        Task branch attention (§4.2).

        Private queries attend to private (t→t) + neutral (t→0) keys.
        Q projection is relation-conditioned: W_Q^{t→t} against private keys,
        W_Q^{t→0} against neutral keys.

        Score formula from the theory:
            score[i,j] = (x_i W_Q^{t→0})(z_j W_K^{t→0})^T   when j is neutral
                       = (x_i W_Q^{t→t})(x_j W_K^{t→t})^T   when j is private

        Args:
            x_task: [B, N, D] task t's effective token sequence
            private_mask: [B, N] bool - True where position is private for task t
            neutral_mask: [B, N] bool - True where position is neutral (task t participates)
            task_id: t (0-indexed task)
            proj_cache: pre-computed effective projections

        Returns:
            attn_out: [B, N, D] attention output (only valid at private positions)
        """
        B, N, D = x_task.shape
        H = self.num_heads
        dk = self.head_dim

        branch_t = task_id + 1  # branch index for task t

        # Get effective projections for the two relation types
        W_tt = proj_cache[(branch_t, branch_t)]  # t→t
        W_t0 = proj_cache[(branch_t, 0)]          # t→0

        # --- Q projections: two variants per the theory ---
        # Q_tt: used when computing score against private keys (t→t)
        # Q_t0: used when computing score against neutral keys (t→0)
        Q_tt = self._project_tokens(x_task, W_tt['W_Q'], self.q_bias)  # [B, N, H, dk]
        Q_t0 = self._project_tokens(x_task, W_t0['W_Q'], self.q_bias)  # [B, N, H, dk]

        # --- K, V projections per relation type ---
        K_tt = self._project_tokens(x_task, W_tt['W_K'], self.k_bias)  # [B, N, H, dk]
        V_tt = self._project_tokens(x_task, W_tt['W_V'], self.v_bias)  # [B, N, H, dk]

        K_t0 = self._project_tokens(x_task, W_t0['W_K'], self.k_bias)  # [B, N, H, dk]
        V_t0 = self._project_tokens(x_task, W_t0['W_V'], self.v_bias)  # [B, N, H, dk]

        # --- Compute score matrix in two parts (relation-conditioned Q) ---
        # Permute to [B, H, N, dk]
        Q_tt_h = Q_tt.permute(0, 2, 1, 3)
        Q_t0_h = Q_t0.permute(0, 2, 1, 3)
        K_tt_h = K_tt.permute(0, 2, 1, 3)
        K_t0_h = K_t0.permute(0, 2, 1, 3)

        # score_pp[i,j] = Q_tt[i] · K_tt[j] (for private key j)
        score_pp = (Q_tt_h @ K_tt_h.transpose(-2, -1)) * self.scale  # [B, H, N, N]
        # score_pn[i,j] = Q_t0[i] · K_t0[j] (for neutral key j)
        score_pn = (Q_t0_h @ K_t0_h.transpose(-2, -1)) * self.scale  # [B, H, N, N]

        # Compose: select score based on key position's branch
        # key_is_private[j]: [B, N] -> [B, 1, 1, N]
        key_is_private = private_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N]
        attn = torch.where(key_is_private, score_pp, score_pn)  # [B, H, N, N]

        # Mask: only private(t) + neutral(t∈N) keys are valid
        valid_key_mask = (private_mask | neutral_mask)  # [B, N]
        key_mask = valid_key_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N]
        attn = attn.masked_fill(~key_mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # --- V: select based on key position type ---
        pm = private_mask.unsqueeze(-1).unsqueeze(-1)  # [B, N, 1, 1]
        V = torch.where(pm, V_tt, V_t0)  # [B, N, H, dk]
        V = V.permute(0, 2, 1, 3)  # [B, H, N, dk]

        out = (attn @ V).permute(0, 2, 1, 3).reshape(B, N, D)  # [B, N, D]

        # Zero out non-private positions (they'll be computed by neutral branch)
        out = out * private_mask.unsqueeze(-1).float()

        return out

    def _neutral_branch_attention(
        self,
        task_outs: dict,
        prev_shared_bits: torch.Tensor,
        proj_cache: dict,
    ) -> torch.Tensor | None:
        """
        Neutral branch attention (§4.3) — faithful implementation.

        Neutral queries attend to:
          - All neutral positions (0→0 relation)
          - Participant tasks' private positions (0→t relation, per-position mask)

        Q projection is relation-conditioned per the theory:
          score[i,r] = (z_i W_Q^{0→0})(z_r W_K^{0→0})^T    for neutral key r
          score[i,j] = (z_i W_Q^{0→t})(x_j W_K^{0→t})^T    for task t private key j

        Executed ONCE outside the per-task loop.

        Args:
            task_outs: {task_id: [B, N, D]} normed task outputs
            prev_shared_bits: [B, N] int64 bitmask
            proj_cache: pre-computed effective projections

        Returns:
            neutral_attn_out: [B, N, D] or None if no neutral positions
                Only valid at neutral positions.
        """
        B, N, D = task_outs[0].shape
        H = self.num_heads
        dk = self.head_dim
        T = self.num_tasks
        device = task_outs[0].device

        neutral_mask = (prev_shared_bits != 0)  # [B, N]
        if not neutral_mask.any():
            return None

        # --- Get neutral token representations ---
        # Neutral tokens are shared across participant tasks.
        # Use task 0's representation at neutral positions as the canonical neutral token.
        # (After apply_shared_broadcast, all participant tasks have the same value at neutral positions.)
        neutral_x = task_outs[0]  # [B, N, D] — neutral positions are identical across participants

        W_00 = proj_cache[(0, 0)]

        # --- Q projections: one per target relation ---
        # Q_00: used against neutral keys (0→0)
        Q_00 = self._project_tokens(neutral_x, W_00['W_Q'], self.q_bias)  # [B, N, H, dk]
        Q_00_h = Q_00.permute(0, 2, 1, 3)  # [B, H, N, dk]

        # Q_0t: used against task t's private keys (0→t), one per task
        Q_0t_list = []  # T elements, each [B, H, N, dk]
        for t in range(T):
            branch_t = t + 1
            W_0t = proj_cache[(0, branch_t)]
            Q_0t = self._project_tokens(neutral_x, W_0t['W_Q'], self.q_bias)  # [B, N, H, dk]
            Q_0t_list.append(Q_0t.permute(0, 2, 1, 3))  # [B, H, N, dk]

        # --- K, V projections per segment ---
        # 0→0 keys: neutral positions
        K_00 = self._project_tokens(neutral_x, W_00['W_K'], self.k_bias)  # [B, N, H, dk]
        V_00 = self._project_tokens(neutral_x, W_00['W_V'], self.v_bias)  # [B, N, H, dk]
        K_00_h = K_00.permute(0, 2, 1, 3)  # [B, H, N, dk]

        # 0→t keys: each task's private positions
        K_tasks_h = []
        V_tasks = []
        for t in range(T):
            branch_t = t + 1
            W_0t = proj_cache[(0, branch_t)]
            K_t = self._project_tokens(task_outs[t], W_0t['W_K'], self.k_bias)  # [B, N, H, dk]
            V_t = self._project_tokens(task_outs[t], W_0t['W_V'], self.v_bias)  # [B, N, H, dk]
            K_tasks_h.append(K_t.permute(0, 2, 1, 3))  # [B, H, N, dk]
            V_tasks.append(V_t)

        # --- Compute score matrix segment by segment (relation-conditioned Q) ---
        # Segment layout: [neutral(N) | task0_private(N) | task1_private(N) | ...]

        # Score for neutral segment: Q_00 @ K_00^T
        score_neutral_seg = (Q_00_h @ K_00_h.transpose(-2, -1)) * self.scale  # [B, H, N, N]

        # Score for each task segment: Q_0t @ K_0t^T
        score_task_segs = []
        for t in range(T):
            score_t = (Q_0t_list[t] @ K_tasks_h[t].transpose(-2, -1)) * self.scale  # [B, H, N, N]
            score_task_segs.append(score_t)

        # Concatenate scores: [B, H, N, (1+T)*N]
        attn = torch.cat([score_neutral_seg] + score_task_segs, dim=-1)

        # --- Build V sequence (unchanged — V doesn't depend on query) ---
        V_all = torch.cat([V_00] + V_tasks, dim=1)  # [B, (1+T)*N, H, dk]
        V_all = V_all.permute(0, 2, 1, 3)  # [B, H, (1+T)*N, dk]

        # --- Build attention mask ---
        # For neutral query at position i:
        #   - Can attend to neutral positions (segment 0): neutral_mask[b, j] = True
        #   - Can attend to task t's private positions (segment t+1):
        #     only if t ∈ N_{l,i} (task t participates at position i)
        #     AND j is private for task t
        #
        # Segment layout: [neutral(N) | task0_private(N) | task1_private(N) | ...]

        # Neutral segment mask: [B, N] → valid if position j is neutral
        neutral_key_valid = neutral_mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N]

        # Task segment masks: position-dependent
        task_masks = []
        for t in range(T):
            # Task t participates at query position i: bit t set in prev_shared_bits
            t_participates_at_i = ((prev_shared_bits >> t) & 1).bool()  # [B, N]
            # Key position j is private for task t: bit t NOT set
            t_private_at_j = ~((prev_shared_bits >> t) & 1).bool()  # [B, N]
            # Combined: [B, N_q, N_k] outer product
            mask_t = t_participates_at_i.unsqueeze(2) & t_private_at_j.unsqueeze(1)  # [B, N_q, N_k]
            mask_t = mask_t.unsqueeze(1)  # [B, 1, N_q, N_k] for head broadcast
            task_masks.append(mask_t)

        # Concatenate all segments: [B, 1, N, (1+T)*N]
        full_mask = torch.cat(
            [neutral_key_valid.expand(-1, -1, N, -1)] + task_masks, dim=-1
        )  # [B, 1, N, (1+T)*N]

        attn = attn.masked_fill(~full_mask, float('-inf'))

        # Also mask out non-neutral query rows
        query_valid = neutral_mask.unsqueeze(1).unsqueeze(-1)  # [B, 1, N, 1]
        attn = attn.masked_fill(~query_valid, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        # Replace NaN from all-masked rows with 0
        attn = attn.nan_to_num(0.0)
        attn = self.attn_drop(attn)

        out = (attn @ V_all).permute(0, 2, 1, 3).reshape(B, N, D)  # [B, N, D]

        # Zero out non-neutral positions
        out = out * neutral_mask.unsqueeze(-1).float()

        return out

    def forward(
        self,
        task_outs: dict,
        prev_shared_bits: torch.Tensor | None,
    ) -> dict:
        """
        Full task-conditioned attention forward.

        Args:
            task_outs: {task_id: [B, N, D]} — normed (post-LN1) task token sequences.
                At neutral positions, all participant tasks should have the same
                representation (from previous block's apply_shared_broadcast).
            prev_shared_bits: [B, N] int64 bitmask from previous block.
                None for the first block (all positions private).

        Returns:
            attn_outputs: {task_id: [B, N, D]} — attention output per task
                (before residual connection, which is handled by the caller).
        """
        T = self.num_tasks
        B, N, D = task_outs[0].shape
        device = task_outs[0].device

        # --- Handle first block: all private ---
        if prev_shared_bits is None:
            prev_shared_bits = torch.zeros(B, N, device=device, dtype=torch.int64)

        # --- Determine per-task branch masks ---
        neutral_mask_global = (prev_shared_bits != 0)  # [B, N]
        has_neutral = neutral_mask_global.any().item()

        # --- Enumerate unique relation types and pre-compute projections ---
        relations = set()
        for t in range(T):
            branch_t = t + 1
            relations.add((branch_t, branch_t))  # t→t
            if has_neutral:
                relations.add((branch_t, 0))       # t→0
                relations.add((0, 0))               # 0→0
                relations.add((0, branch_t))         # 0→t

        if not has_neutral:
            # Only t→t relations, add (0,0) as placeholder
            relations.add((0, 0))

        proj_cache = self._compute_effective_projections(list(relations), device)

        # --- Task branch attention (§4.2) ---
        task_attn_outs = {}
        for t in range(T):
            # Per task t: private vs neutral positions
            t_neutral = ((prev_shared_bits >> t) & 1).bool()  # [B, N]
            t_private = ~t_neutral  # [B, N]

            task_attn_outs[t] = self._task_branch_attention(
                x_task=task_outs[t],
                private_mask=t_private,
                neutral_mask=t_neutral,
                task_id=t,
                proj_cache=proj_cache,
            )

        # --- Neutral branch attention (§4.3) ---
        neutral_attn_out = None
        if has_neutral:
            neutral_attn_out = self._neutral_branch_attention(
                task_outs=task_outs,
                prev_shared_bits=prev_shared_bits,
                proj_cache=proj_cache,
            )

        # --- Combine: task branch (private) + neutral branch (neutral) ---
        attn_outputs = {}
        for t in range(T):
            out_t = task_attn_outs[t]  # private positions filled
            if neutral_attn_out is not None:
                # Add neutral attention output at neutral positions
                t_neutral = ((prev_shared_bits >> t) & 1).bool()  # [B, N]
                neutral_contrib = neutral_attn_out * t_neutral.unsqueeze(-1).float()
                out_t = out_t + neutral_contrib

            # Output projection
            out_t = self.proj(out_t)
            out_t = self.proj_drop(out_t)
            attn_outputs[t] = out_t

        return attn_outputs
