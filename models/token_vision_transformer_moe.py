import torch
import torch.nn as nn
from functools import partial
import math
from itertools import repeat
# from torch._six import container_abcs
import collections.abc
import warnings
from collections import OrderedDict, deque
from utils.helpers import load_pretrained,load_pretrained_pos_emb
from models.token_custom_moe_layer import TokenFMoETransformerMLP
# from .layers import DropPath, to_2tuple, trunc_normal_
from timm.layers  import lecun_normal_
# from ..builder import BACKBONES
import numpy as np
from collections import Counter
from models.gate_funs.noisy_gate import NoisyGate
from models.gate_funs.noisy_gate_vmoe import NoisyGate_VMoE
from models.gate_funs.token_noisy_gate_vmoe import TokenNoisyGate_VMoE
from torch.utils.checkpoint import checkpoint

from .gates import NoisyGate_VMoE as Custom_VMoE
from models.moe import TaskMoE
from models.router_state import RouterState, selector_to_bits, popcount_bits, bits_to_task_masks, bits_to_multihot, compute_masks
from models.aggregation_stages import AggregationStage

a=[[0],[1,17,18,19,20],[2,12,13,14,15,16],[3,9,10,11],[4,5],[6,7,8,38],[21,22,23,24,25,26,39],[27,28,29,30,31,32,33,34,35,36,37]]


class RouterSelector(nn.Module):
    def __init__(self, d_model, d_task_emb=0, temperature=1.0, hard=False):
        """
        Router selector that determines whether each token uses task-specific or shared gate.

        Args:
            d_model: Input feature dimension
            d_task_emb: Task embedding dimension. If > 0, task_emb is expected in forward
            temperature: Gumbel-Softmax temperature
            hard: Whether to use hard selection in training
        """
        super().__init__()
        self.d_model = d_model
        self.d_task_emb = d_task_emb
        d_input = d_model + d_task_emb if d_task_emb > 0 else d_model

        # Output dimension is 2: [task-specific, shared]
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
        Returns selection probabilities/decisions for using shared_gate.

        Args:
            inp: Input tensor [B, N, D] or [B*N, D]
            task_emb: Task embedding tensor [E], [B, E], or [B, N, E]

        Returns:
            Tensor [B, N] or [B*N]: probability or binary selection (0=task-specific, 1=shared)
        """
        shape_input = list(inp.shape)
        original_shape = shape_input[:-1]
        channel = shape_input[-1]
        inp_flat = inp.reshape(-1, channel)
        BN = inp_flat.shape[0]

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
            probs = torch.nn.functional.gumbel_softmax(
                logits, tau=self.temperature, hard=self.hard, dim=-1
            )
        else:
            probs = torch.nn.functional.gumbel_softmax(
                logits, tau=self.temperature, hard=True, dim=-1
            )

        output = probs[:, 1]

        if len(original_shape) > 1:
            output = output.reshape(*original_shape)

        return output


class Aggregation(nn.Module):
    """Aggregates results from multiple tasks"""
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

        # [T, B, N, D]
        outputs = torch.stack(
            [task_outputs[t] for t in range(len(curr_shared_masks))],
            dim=0
        )

        # [T, B, N]
        shared_mask = torch.stack(curr_shared_masks, dim=0)

        # [T, B, N] → only aggregate where needed
        valid_mask = shared_mask & aggregation_mask.unsqueeze(0)

        # [T, B, N, 1]
        valid_mask_f = valid_mask.unsqueeze(-1).float()

        # weighted sum
        summed = (outputs * valid_mask_f).sum(dim=0)  # [B, N, D]

        # count how many tasks contributed
        count = valid_mask_f.sum(dim=0)  # [B, N, 1]

        aggregated = summed / (count + self.eps)
        return aggregated
def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225),
        'first_conv': '', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # patch models
    'vit_tiny_patch16_224': _cfg(
        url = 'https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth',
    ),
    'vit_small_patch16_224': _cfg(
        url = 'https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth',
        # url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth',
    ),
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_base_p16_224-4e355ebd.pth',
    ),
    'vit_base_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_384-83fb41ba.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_base_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p32_384-830016f5.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch16_224': _cfg(),
    'vit_large_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_384-b3be5167.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0,
    ),
    'vit_large_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_huge_patch16_224': _cfg(),
    'vit_huge_patch32_384': _cfg(input_size=(3, 384, 384)),
    # hybrid models
    'vit_small_resnet26d_224': _cfg(),
    'vit_small_resnet50d_s3_224': _cfg(),
    'vit_base_resnet26d_224': _cfg(),
    'vit_base_resnet50d_224': _cfg(),
    'deit_base_distilled_path16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0, checkpoint=True,
    ),
}


def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return tuple(repeat(x, 2))


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # work with diff dim tensors, not just 2D ConvNets
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + \
            torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class new_Mlp(nn.Module):
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

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        if qk_scale is None:
            self.scale = head_dim ** -0.5
        else:
            self.scale = qk_scale

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # print('for attention',x.shape)
        # print(self.scale)
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                      C // self.num_heads).permute(2, 0, 3, 1, 4)
        # print(q.shape,k.shape,v.shape)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * \
            (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."

        # x = F.interpolate(x, size=2*x.shape[-1], mode='bilinear', align_corners=True)
        x = self.proj(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 moe=False, moe_mlp_ratio=1., moe_experts=8,
                 moe_top_k=4, moe_gate_dim=-1, world_size=2,
                 moe_gate_type="token_noisy_vmoe", vmoe_noisy_std=1, gate_task_specific_dim=64, multi_gate=False,
                 num_experts_pertask = -1, num_tasks = -1,
                 gate_input_ahead = False):
        super().__init__()
        self.moe = moe
        self.moe_top_k = moe_top_k
        self.tot_expert = moe_experts
        self.num_tasks = num_tasks
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.gate_input_ahead = gate_input_ahead

        # Aggregation stages (Phase 2 refactoring)
        # Each stage creates its own Aggregation module
        # - attn_post_aggr: aggregation after attention, before LN2 (uses prev_state)
        # - mlp_post_aggr: aggregation after MLP (uses curr_state, MoE only)
        if num_tasks > 0:
            self.attn_post_aggr = AggregationStage(Aggregation(), num_tasks)
            self.mlp_post_aggr = AggregationStage(Aggregation(), num_tasks) if moe else None
        else:
            self.attn_post_aggr = None
            self.mlp_post_aggr = None

        if moe:
            activation = nn.Sequential(
                act_layer(),
                nn.Dropout(drop)
            )
            if moe_gate_dim < 0:
                moe_gate_dim = dim
            self.moe_gate_dim = moe_gate_dim
            if moe_mlp_ratio < 0:
                moe_mlp_ratio = mlp_ratio
            moe_hidden_dim = int(dim * moe_mlp_ratio)

            # Store gate parameters for routing
            self.gate_task_specific_dim = gate_task_specific_dim
            self.multi_gate = multi_gate
            self.world_size = world_size
            self.num_experts_pertask = num_experts_pertask

            # Gate input dimension
            # - shared gate: always uses task embedding (multi-hot)
            # - task-specific gate: uses task embedding only if single gate (not multi_gate)
            assert gate_task_specific_dim > 0, f"gate_task_specific_dim must be positive, got {gate_task_specific_dim}"

            d_gate_with_emb = dim + gate_task_specific_dim
            d_gate_no_emb = dim

            # RouterSelector - determines shared vs task-specific
            self.router_selector = RouterSelector(d_model=dim, d_task_emb=gate_task_specific_dim, temperature=1.0, hard=False)

            # Shared gate - always receives multi-hot task embedding
            self.shared_gate = TokenNoisyGate_VMoE(
                d_gate_with_emb, moe_experts, world_size, moe_top_k,
                noise_std=vmoe_noisy_std, num_experts_pertask=num_experts_pertask, num_tasks=num_tasks
            )

            # Task-specific gate(s)
            if multi_gate:
                # Separate gate per task - no task embedding needed
                self.gate = nn.ModuleList([
                    TokenNoisyGate_VMoE(
                        d_gate_no_emb, moe_experts, world_size, moe_top_k,
                        noise_std=vmoe_noisy_std, num_experts_pertask=num_experts_pertask, num_tasks=num_tasks
                    )
                    for _ in range(num_tasks)
                ])
            else:
                # Single gate for all tasks - needs task embedding to distinguish
                self.gate = TokenNoisyGate_VMoE(
                    d_gate_with_emb, moe_experts, world_size, moe_top_k,
                    noise_std=vmoe_noisy_std, num_experts_pertask=num_experts_pertask, num_tasks=num_tasks
                )

            # MoE MLP (experts only, no gates)
            self.mlp = TokenFMoETransformerMLP(
                num_expert=moe_experts, d_model=dim, d_gate=moe_gate_dim, d_hidden=moe_hidden_dim,
                world_size=world_size, top_k=moe_top_k, activation=activation
            )
            self.mlp_drop = nn.Dropout(drop)

        else:
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


    @property
    def get_mlp(self):
        return self.mlp

    @staticmethod
    def _gates_to_load(gates):
        """Compute the true load per expert, given the gates.
        The load is the number of examples for which the corresponding gate is >0.
        Args:
            gates: a `Tensor` of shape [batch_size, n]
        Returns:
            a float32 `Tensor` of shape [n]
        """
        return (gates > 0).sum(0)

    @staticmethod
    def cv_squared(x, eps=1e-10):
        """The squared coefficient of variation of a sample.
        Useful as a loss to encourage a positive distribution to be more uniform.
        Epsilons added for numerical stability.
        Returns 0 for an empty Tensor.

        Args:
            x: a `Tensor`.
        Returns:
            a `Scalar`.
        """
        if x.numel() <= 1:
            return x.new_tensor(0.0)
        x = x.float()
        return x.var(unbiased=False) / (x.mean() ** 2 + eps)

    @staticmethod
    def _prob_in_top_k(clean_values, noisy_values, noise_stddev, noisy_top_values, top_k):
        """Helper function to NoisyTopKGating.
        Computes the probability that value is in top k, given different random noise.

        Args:
            clean_values: a `Tensor` of shape [batch, n].
            noisy_values: a `Tensor` of shape [batch, n].
            noise_stddev: a `Tensor` of shape [batch, n], or None
            noisy_top_values: a `Tensor` of shape [batch, m].
            top_k: integer, the k in top-k
        Returns:
            a `Tensor` of shape [batch, n].
        """
        from torch.distributions.normal import Normal

        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()
        threshold_positions_if_in = (
            torch.arange(batch, device=clean_values.device) * m + top_k
        )
        threshold_if_in = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_in), 1
        )
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_out), 1
        )

        normal = Normal(
            torch.tensor(0.0, device=clean_values.device),
            torch.tensor(1.0, device=clean_values.device),
        )

        prob_if_in = normal.cdf((clean_values - threshold_if_in) / (noise_stddev + 1e-9))
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / (noise_stddev + 1e-9))
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    # --------------------------
    # Phase 1 Refactoring: Split into forward_attn and forward_mlp
    # This allows aggregation to be inserted between attention and MLP
    # --------------------------

    def _attn_only(self, x):
        """Internal checkpointed function: attention only"""
        return x + self.drop_path(self.attn(self.norm1(x)))

    def _forward_attn(self, x):
        """
        Execute attention part only: LN1 -> Attn -> Residual

        Args:
            x: Input tensor [B, N, C]

        Returns:
            x: Output after attention [B, N, C]
        """
        if self.training:
            return checkpoint(self._attn_only, x, use_reentrant=False)
        else:
            return self._attn_only(x)

    def _mlp_dense_inner(self, x):
        """Internal checkpointed function: Dense MLP only"""
        normed_x = self.norm2(x)
        return x + self.drop_path(self.mlp(normed_x))

    def _forward_mlp_dense(self, x):
        """
        Execute dense MLP: LN2 -> Dense MLP -> Residual

        Args:
            x: Input tensor [B, N, C]

        Returns:
            x: Output after MLP [B, N, C]
        """
        if self.training:
            return checkpoint(self._mlp_dense_inner, x, use_reentrant=False)
        else:
            return self._mlp_dense_inner(x)

    def _mlp_moe_inner(self, x, selector_output, task_id, task_emb, gate_shared_emb, compute_mask=None):
        """Internal checkpointed function: MoE routing + MLP

        Args:
            x: Input tensor [B, N, C]
            selector_output: [B, N] router selector output
            task_id: Task ID for selecting task-specific gate (int)
            task_emb: [E] one-hot task embedding for task-specific gate
            gate_shared_emb: [B, N, E] multi-hot embedding for shared gate
            compute_mask: [B, N] bool, optional. If provided, only compute these positions.
                         Used for Phase 2 optimization to skip reusable positions.

        Returns:
            output: x + residual [B, N, C]
            gate_top_k_idx: [B*N, top_k] or [K, top_k] expert indices
            gate_score: [B*N, top_k] or [K, top_k] expert scores
            moe_output: [B, N, C] MoE output before residual
            moe_component: [B, N, C] MoE component after drop_path (for caching)
            gate_info: dict with gate statistics for cv_loss computation
        """
        B, N, C = x.shape
        device = x.device
        normed_x = self.norm2(x)

        if compute_mask is not None:
            # Phase 2: compute_mask 제공됨 - masked computation mode
            compute_flat = compute_mask.flatten()  # [B*N]

            if not compute_flat.any():
                # Case: compute_mask가 모두 False (계산할 위치 0개)
                # 이 경우 MoE 계산을 완전히 skip하고 zero output 반환
                # wrapper에서 cache만 사용하거나 output = x가 되어야 함
                moe_output = torch.zeros(B, N, C, device=device, dtype=x.dtype)
                moe_component = torch.zeros(B, N, C, device=device, dtype=x.dtype)
                output = x + moe_component  # output = x

                # Empty gate_info (gate routing 없음)
                gate_info = self._create_empty_gate_info(device)

                # Dummy gate outputs (wrapper에서 사용하지 않음)
                gate_top_k_idx = torch.zeros(0, self.moe_top_k, dtype=torch.long, device=device)
                gate_score = torch.zeros(0, self.moe_top_k, device=device)

                return output, gate_top_k_idx, gate_score, moe_output, moe_component, gate_info

            else:
                # Case: compute_mask에 True가 하나 이상 존재 (scatter/gather)
                num_compute = compute_flat.sum().item()

                # Gather
                normed_x_flat = normed_x.reshape(B * N, C)  # [B*N, C]
                normed_x_compute = normed_x_flat[compute_flat]  # [K, C]

                selector_flat = selector_output.flatten()  # [B*N]
                selector_compute = selector_flat[compute_flat]  # [K]

                if gate_shared_emb is not None:
                    gate_shared_emb_flat = gate_shared_emb.reshape(B * N, -1)  # [B*N, E]
                    gate_shared_emb_compute = gate_shared_emb_flat[compute_flat]  # [K, E]
                else:
                    gate_shared_emb_compute = None

                # Route only computed tokens
                gate_top_k_idx, gate_score, gate_info = self._route_tokens(
                    normed_x_compute,
                    selector_output=selector_compute,
                    task_id=task_id,
                    task_emb=task_emb,
                    gate_shared_emb=gate_shared_emb_compute
                )

                # MoE forward on computed tokens
                moe_output_compute = self.mlp(normed_x_compute, gate_top_k_idx, gate_score)  # [K, C]

                # Scatter back to full shape
                moe_output = torch.zeros(B, N, C, device=device, dtype=x.dtype)
                moe_output.reshape(B * N, C)[compute_flat] = moe_output_compute

                moe_component = self.drop_path(self.mlp_drop(moe_output))
                output = x + moe_component

        else:
            # Phase 1 or no masking: Compute all positions
            gate_top_k_idx, gate_score, gate_info = self._route_tokens(
                normed_x,
                selector_output=selector_output,
                task_id=task_id,
                task_emb=task_emb,
                gate_shared_emb=gate_shared_emb
            )

            moe_output = self.mlp(normed_x, gate_top_k_idx, gate_score)
            moe_component = self.drop_path(self.mlp_drop(moe_output))
            output = x + moe_component

        return output, gate_top_k_idx, gate_score, moe_output, moe_component, gate_info

    def _forward_mlp_moe(self, x, selector_output, task_id, task_emb, gate_shared_emb):
        """
        Execute MoE MLP: LN2 -> Gate routing -> MoE -> Residual

        Args:
            x: Input tensor [B, N, C]
            selector_output: [B, N] router selector output
            task_id: Task ID for selecting task-specific gate (int)
            task_emb: [E] one-hot task embedding for task-specific gate
            gate_shared_emb: [B, N, E] multi-hot embedding for shared gate

        Returns:
            x: Output after MoE [B, N, C]
            gate_info: dict for cv_loss computation
        """
        if self.training:
            output, gate_idx, gate_score, moe_out, moe_component, gate_info = checkpoint(
                self._mlp_moe_inner,
                x, selector_output, task_id, task_emb, gate_shared_emb,
                use_reentrant=False
            )
        else:
            output, gate_idx, gate_score, moe_out, moe_component, gate_info = self._mlp_moe_inner(
                x, selector_output, task_id, task_emb, gate_shared_emb
            )

        # Keep external API unchanged (only return output and gate_info)
        return output, gate_info

    def _forward_mlp_moe_with_cache(
        self, x, selector_output, task_id, task_emb, gate_shared_emb,
        cached_moe_component, cached_valid_mask,
        can_reuse_mask, cache_fill_mask
    ):
        """
        Cache-aware MoE forward with computation reuse.

        Strategy (revised to separate compute_mask from cache_fill_mask):
        - can_reuse_mask: positions where this task can copy from cache
        - cache_fill_mask: positions where this task computes and populates cache (reuse intersection only)
        - compute_mask: ALL positions that need MoE computation (built internally)
          = task_specific | (task_shared & ~can_reuse_mask)
          This ensures shared positions outside reuse intersection are also computed

        Case 1 (No caching):
            - can_reuse_mask=None: 재사용 시나리오가 아님
            - 정상적인 MoE forward 수행

        Case 2 (Full reuse):
            - can_reuse_mask.all() == True (모든 위치가 cache에서 복사됨)
            - cache_fill_mask is empty (채울 게 없음)
            - task-specific 위치가 없음
            - gate routing skip, expert computation skip
            - empty gate_info 반환 (cv_loss 중복 방지)

        Case 3 (Partial reuse):
            - can_reuse_mask와 cache_fill_mask 모두 존재
            - compute_mask = task_specific | (task_shared & ~can_reuse_mask)
              (재사용 교집합 밖 shared 위치도 포함됨)
            - can_reuse_mask 위치: cache에서 복사
            - cache_fill_mask 위치: 계산 후 cache에 저장 (재사용 교집합에만 제한)

        Checkpoint Safety:
            - Cache populate는 checkpoint 외부에서 수행
            - Backward 시 재실행되어도 side-effect 없음
            - No .detach() - gradient flow preserved

        Args:
            x: [B, N, C] input tensor
            selector_output: [B, N] router selector output
            task_id: int, task ID
            task_emb: [E] task embedding
            gate_shared_emb: [B, N, E] shared gate embedding
            cached_moe_component: [B, N, C] cached MoE component after drop_path (functional update)
            cached_valid_mask: [B, N] bool, cache validity (mutable, in-place update)
            can_reuse_mask: [B, N] bool, where to reuse cache
            cache_fill_mask: [B, N] bool, where to compute and populate cache (reuse intersection only)

        Returns:
            output: [B, N, C] MoE output
            gate_info: dict with gate statistics
            compute_positions: [B, N] bool, positions where actual computation happened
            cached_moe_component: [B, N, C] updated cache (functional update, preserves gradient)
        """
        B, N, C = x.shape
        device = x.device

        # ===== Case 1: No caching - standard forward =====
        if can_reuse_mask is None:
            output, gate_info = self._forward_mlp_moe(x, selector_output, task_id, task_emb, gate_shared_emb)
            # 재사용 없음: 모든 위치에서 계산 수행
            compute_positions = torch.ones(B, N, dtype=torch.bool, device=device)
            # No cache update needed
            return output, gate_info, compute_positions, cached_moe_component

        # ===== Case 2: Full reuse (all shared positions cached, no task-specific) =====
        # Full reuse 조건:
        # 1. can_reuse_mask가 모든 위치를 커버함 (전체 위치에서 cache 사용)
        # 2. cache_fill_mask가 empty (채울 게 없음)
        # 3. task-specific 위치가 없음 (selector_output <= 0.5인 위치는 task-specific)
        # 4. CRITICAL: can_reuse_mask.all()로 전체 커버 여부 확인 (안전성 보장)
        task_specific_mask = (selector_output <= 0.5)  # [B, N]

        # Full reuse 안전성 체크:
        # - cache_fill_mask가 empty이고 (채울 게 없음)
        # - task_specific 위치가 없고 (모든 위치가 shared)
        # - can_reuse_mask가 전체를 커버함 (모든 위치에서 cache 사용)
        # 이 조건들이 모두 만족되어야 cached_moe_component를 전체에 적용 가능
        if ((cache_fill_mask is None or not cache_fill_mask.any()) and
            not task_specific_mask.any() and
            can_reuse_mask is not None and can_reuse_mask.all()):
            # 이 task는 모든 위치에서 shared를 사용하고, 전부 이미 캐시되어 있음
            # Gate routing 및 expert computation 완전히 skip

            # Safety: Ensure cache is actually populated
            assert cached_moe_component is not None, (
                "Full reuse condition met but cached_moe_component is None. "
                "This indicates a logic error in cache initialization."
            )

            # Cache에서 MoE component 가져와서 residual 구성
            output = x + cached_moe_component

            # Empty gate_info 반환 (이 task는 gate decision을 하지 않음)
            # → cv_loss 계산 시 이 task의 기여도가 0이 됨 (중복 카운팅 방지)
            gate_info = self._create_empty_gate_info(device)

            # Full reuse: 계산한 위치 없음
            compute_positions = torch.zeros(B, N, dtype=torch.bool, device=device)
            # No cache update needed (all from cache)
            return output, gate_info, compute_positions, cached_moe_component

        # ===== Case 3: Partial reuse - compute some, reuse some =====
        # cache_fill_mask 위치: 이 task가 대표로 계산 수행 후 cache에 저장 (재사용 교집합에만 제한)
        # can_reuse_mask 위치: 이전 task의 결과 재사용
        # task_specific 위치: 정상 계산
        # shared-non-reuse 위치: shared이지만 재사용 교집합 밖 (정상 계산, cache에는 저장 안 함)

        # Step 1: Compute mask = ALL positions that need actual MoE computation
        # ✅ CRITICAL FIX: 재사용 교집합 밖 shared 위치도 포함되어야 함
        # compute_mask = task_specific | (task_shared & ~can_reuse_mask)
        task_specific = (selector_output <= 0.5)  # [B, N]
        task_shared = (selector_output > 0.5)  # [B, N]

        # Positions that don't use cache (either task-specific or shared-but-not-reusing)
        compute_positions = task_specific.clone()  # Start with task-specific
        if can_reuse_mask is not None:
            # Add shared positions that are NOT being reused
            compute_positions = compute_positions | (task_shared & ~can_reuse_mask)
        else:
            # No reuse at all: all shared positions need computation
            compute_positions = compute_positions | task_shared

        # Step 2: MoE computation (on compute_positions)
        # checkpoint 내부: 순수 계산만 (side-effect 없음)
        if self.training:
            output, gate_idx, gate_score, moe_out, moe_component, gate_info = checkpoint(
                self._mlp_moe_inner,
                x, selector_output, task_id, task_emb, gate_shared_emb, compute_positions,
                use_reentrant=False
            )
        else:
            output, gate_idx, gate_score, moe_out, moe_component, gate_info = self._mlp_moe_inner(
                x, selector_output, task_id, task_emb, gate_shared_emb, compute_positions
            )

        # Step 3: Cache populate (gradient flow 유지)
        # cache_fill_mask 위치만 cache에 저장 (재사용 교집합에만 제한)
        # detach하지 않음 → gradient가 cache를 통해 MoE 파라미터로 전달됨
        if cache_fill_mask is not None and cache_fill_mask.any():
            # In-place update는 bool mask에만 허용 (gradient 무관)
            cached_valid_mask[cache_fill_mask] = True

            # moe_component는 in-place 대신 functional update
            # 이렇게 하면 cached_moe_component가 계산 그래프에 포함되어
            # 재사용 task의 gradient가 대표 task의 MoE 파라미터로 전달됨
            #
            # Memory Trade-off:
            # torch.where 체인이 task마다 쌓여서 num_tasks 큰 경우 그래프 길어질 수 있음
            # 이는 gradient flow를 위한 의도된 설계이지만,
            # 메모리 예산이 빡빡하면 accumulate buffer + single write 방식 고려 필요
            cached_moe_component = torch.where(
                cache_fill_mask.unsqueeze(-1),  # [B, N, 1]
                moe_component,  # 새로 계산된 값 (gradient 있음)
                cached_moe_component  # 기존 값 유지
            )

        # Step 4: Reuse 위치에서 cached component 사용 (functional merge)
        # In-place update 대신 torch.where 사용 → autograd 안정성 보장
        if can_reuse_mask is not None and can_reuse_mask.any():
            output = torch.where(
                can_reuse_mask.unsqueeze(-1),  # [B, N, 1]
                x + cached_moe_component,  # Cached component 사용 (gradient 흐름)
                output  # 정상 계산 결과 유지
            )

        # gate_info는 compute_positions에 대한 정보만 담고 있음
        # Block.forward()에서 그대로 사용 (필터링 불필요)
        # compute_positions = task_specific | (task_shared & ~can_reuse_mask)
        # cached_moe_component 반환 (functional update된 버전, gradient 유지)
        return output, gate_info, compute_positions, cached_moe_component

    def _route_tokens(self, x, selector_output, task_id, task_emb, gate_shared_emb):
        """
        Route tokens through appropriate gates based on selector output.

        Args:
            x: Input tensor [B, N, D] (dense) or [K, D] (masked)
            selector_output: [B, N] (dense) or [K] (masked) router selector output
            task_id: Task ID for selecting task-specific gate (int)
            task_emb: [E] one-hot task embedding for task-specific gate
            gate_shared_emb: [B, N, E] (dense) or [K, E] (masked) multi-hot embedding for shared gate

        Returns:
            gate_top_k_idx: [B*N, top_k] or [K, top_k] expert indices
            gate_score: [B*N, top_k] or [K, top_k] expert scores
            gate_info: dict with clean_logits, noisy_logits, noise_stddev, top_logits, gates
        """
        device = x.device

        # Handle both dense [B, N, D] and masked [K, D] inputs
        if x.dim() == 3:
            # Dense mode: [B, N, D]
            B, N, D = x.shape
            B_N = B * N
            x_flat = x.reshape(B_N, D)
            selector_flat = selector_output.reshape(-1)
            if gate_shared_emb is not None:
                shared_emb_flat = gate_shared_emb.reshape(B_N, -1)
            else:
                shared_emb_flat = None
        elif x.dim() == 2:
            # Masked mode: [K, D]
            K, D = x.shape
            B_N = K
            x_flat = x  # Already flat
            selector_flat = selector_output  # Already flat [K]
            shared_emb_flat = gate_shared_emb  # Already flat [K, E] or None
        else:
            raise ValueError(f"Expected x to be 2D or 3D, got {x.dim()}D")

        # Flatten x to [B*N, D] or use [K, D] directly
        # (x_flat already set above)

        # Create masks
        shared_mask = (selector_flat > 0.5)
        task_mask = (selector_flat <= 0.5)

        # Prepare task embedding flat: [E] -> [B*N, E] or [K, E]
        task_emb_flat = None
        if task_emb is not None:
            task_emb_flat = task_emb.unsqueeze(0).expand(B_N, -1)

        # Initialize output tensors
        gate_top_k_idx = torch.zeros(B_N, self.moe_top_k, dtype=torch.long, device=device)
        gate_score = torch.zeros(B_N, self.moe_top_k, device=device)

        # Initialize tensors for cv_loss computation
        clean_logits = torch.zeros(B_N, self.tot_expert, device=device)
        noisy_logits = torch.zeros(B_N, self.tot_expert, device=device)
        noise_stddev = torch.zeros(B_N, self.tot_expert, device=device)
        top_logits = torch.zeros(B_N, self.moe_top_k + 1, device=device)
        gates = torch.zeros(B_N, self.tot_expert, device=device)

        # Route shared tokens through shared_gate
        if shared_mask.any():
            shared_inp = x_flat[shared_mask]
            # CRITICAL: shared_emb_flat must exist when routing shared tokens
            # self.shared_gate expects input dim = d_gate_with_emb = dim + gate_task_specific_dim
            # Without embedding concat, input would be dim (shape mismatch)
            assert shared_emb_flat is not None, (
                "shared_emb_flat is None but shared tokens exist. "
                "This indicates gate_shared_emb was not generated in Block.forward(). "
                "Ensure gate_task_represent is provided when gate_task_specific_dim > 0."
            )
            shared_inp = torch.cat([shared_inp, shared_emb_flat[shared_mask]], dim=-1)
            (s_idx, s_score), s_clean, s_noisy, s_noise_std, s_top, s_gates = self.shared_gate(shared_inp)
            gate_top_k_idx[shared_mask] = s_idx
            gate_score[shared_mask] = s_score
            clean_logits[shared_mask] = s_clean
            noisy_logits[shared_mask] = s_noisy
            noise_stddev[shared_mask] = s_noise_std
            top_logits[shared_mask] = s_top
            gates[shared_mask] = s_gates

        # Route task-specific tokens through task-specific gate
        if task_mask.any():
            task_inp = x_flat[task_mask]
            if self.multi_gate:
                # multi_gate: separate gate per task, no need for task embedding
                (t_idx, t_score), t_clean, t_noisy, t_noise_std, t_top, t_gates = self.gate[task_id](task_inp)
            else:
                # single gate: concat task embedding to distinguish tasks
                assert task_emb_flat is not None, "task_emb required for single gate mode"
                task_inp = torch.cat([task_inp, task_emb_flat[task_mask]], dim=-1)
                (t_idx, t_score), t_clean, t_noisy, t_noise_std, t_top, t_gates = self.gate(task_inp, task_id=task_id)
            gate_top_k_idx[task_mask] = t_idx
            gate_score[task_mask] = t_score
            clean_logits[task_mask] = t_clean
            noisy_logits[task_mask] = t_noisy
            noise_stddev[task_mask] = t_noise_std
            top_logits[task_mask] = t_top
            gates[task_mask] = t_gates

        gate_info = {
            'clean_logits': clean_logits,
            'noisy_logits': noisy_logits,
            'noise_stddev': noise_stddev,
            'top_logits': top_logits,
            'gates': gates,
        }

        return gate_top_k_idx, gate_score, gate_info

    def _compute_cv_loss(self, gate_info):
        """
        Compute cv_loss from gate information.

        Args:
            gate_info: dict with clean_logits, noisy_logits, noise_stddev, top_logits, gates

        Returns:
            cv_loss: scalar tensor
        """
        gates = gate_info['gates']
        noise_stddev = gate_info['noise_stddev']
        clean_logits = gate_info['clean_logits']
        noisy_logits = gate_info['noisy_logits']
        top_logits = gate_info['top_logits']

        importance = gates.sum(0)  # [E]

        noise_ok = (noise_stddev is not None) and (noise_stddev.mean().item() > 1e-6)
        if (self.moe_top_k < self.tot_expert) and noise_ok:
            load = self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits, self.moe_top_k).sum(0)
        else:
            load = self._gates_to_load(gates)

        return self.cv_squared(importance) + self.cv_squared(load)

    def _create_empty_gate_info(self, device):
        """
        Create empty gate_info dict for when no computation happened.

        Used in full reuse case where all positions are cached.

        Args:
            device: torch device

        Returns:
            gate_info: dict with empty tensors
        """
        return {
            'clean_logits': torch.empty(0, self.tot_expert, device=device),
            'noisy_logits': torch.empty(0, self.tot_expert, device=device),
            'noise_stddev': torch.empty(0, self.tot_expert, device=device),
            'top_logits': torch.empty(0, self.moe_top_k + 1, device=device),
            'gates': torch.empty(0, self.tot_expert, device=device),
        }

    def _filter_gate_info_by_mask(self, gate_info, mask_2d):
        """
        Filter gate_info to only include positions where mask is True.

        Used to exclude reusable positions from cv_loss computation for
        subsequent tasks (avoid double-counting).

        Args:
            gate_info: dict with [B*N, ...] tensors
            mask_2d: [B, N] bool, True for positions to keep

        Returns:
            filtered_gate_info: dict with [K, ...] tensors where K = mask.sum()
        """
        mask_flat = mask_2d.flatten()

        if not mask_flat.any():
            return self._create_empty_gate_info(mask_2d.device)

        filtered = {}
        for key, tensor in gate_info.items():
            filtered[key] = tensor[mask_flat]

        return filtered

    def _build_router_state(self, selector_tbN: torch.Tensor) -> RouterState:
        """
        Build RouterState from selector outputs.

        Args:
            selector_tbN: [T, B, N] selector outputs

        Returns:
            RouterState with shared_bits and shared_selector.
            The shared_bits will be used by the next block as prev_state.shared_bits
            for post-attention aggregation and reuse detection.
        """
        bits = selector_to_bits(selector_tbN, thr=0.5)  # [B, N] int64
        return RouterState(
            shared_bits=bits,
            shared_selector=selector_tbN.detach()
        )

    def forward(self, outs: dict, prev_state: RouterState | None,
                gate_inp=None, task_emb_T_E=None, gate_task_represent=None):
        """
        Multi-task forward with RouterState management.

        Pipeline:
        1. Attention for all tasks (LN1 -> Attn -> Residual)
        2. Attn-post aggregation (using prev_state, if available)
        3. MLP for all tasks (LN2 -> MLP/MoE -> Residual)
        4. For MoE: compute selector, build curr_state, mlp-post aggregation

        Args:
            outs: Dict {task_id: tensor [B, N, C]} containing task outputs
            prev_state: RouterState from previous MoE block (None for first MoE)
            gate_inp: Gate input for MoE routing
            task_emb_T_E: [T, E] cached task embeddings (one-hot based) for selector
            gate_task_represent: MLP module to convert multi-hot to embedding for shared gate

        Returns:
            outs: Updated dict with processed outputs
            curr_state: RouterState from this block (None for non-MoE)
            aux: Dict with cv_loss, stats, etc.
        """
        aux = {
            'cv_loss': outs[0].new_tensor(0.0),
            'stats': {
                'total_positions': 0,
                'shared_positions_prev': 0,
                'aggregated_positions_attn': 0,
                'shared_positions_curr': 0,
                'aggregated_positions_mlp': 0,
            }
        }

        # (1) Attention for all tasks (checkpoint inside forward_attn)
        for task in range(self.num_tasks):
            outs[task] = self._forward_attn(outs[task])

        B, N, _ = outs[0].shape
        aux['stats']['total_positions'] = B * N

        # (2) Attn-post aggregation (always, using prev_state if available)
        #     This is OUTSIDE checkpoint for safety (path depends on state)
        if prev_state is not None and self.attn_post_aggr is not None:
            prev_bits = prev_state.shared_bits  # [B, N]
            prev_count = popcount_bits(prev_bits, self.num_tasks)  # [B, N]
            attn_agg_needed = (prev_count >= 2)  # [B, N]

            prev_task_masks = bits_to_task_masks(prev_bits, self.num_tasks)
            outs = self.attn_post_aggr(outs, prev_task_masks, attn_agg_needed)

            # Stats
            aux['stats']['shared_positions_prev'] = (prev_count > 0).sum().item()
            aux['stats']['aggregated_positions_attn'] = attn_agg_needed.sum().item()

        # (3) MLP execution for non-MoE block
        if not self.moe:
            for task in range(self.num_tasks):
                outs[task] = self._forward_mlp_dense(outs[task])
            # Pass through prev_state unchanged (preserves previous MoE block's state)
            return outs, prev_state, aux

        # ---- MoE block specific logic ----

        # (4) Compute selector AFTER aggregation (outside checkpoint)
        #     Selector is computed on post-aggregation representation
        #     Pass task_emb_T_E[task] (one-hot based embedding) to selector
        is_first_moe = (prev_state is None)

        if is_first_moe:
            # (4a) First MoE block: force selector to 0 (task-specific only, no shared gate)
            curr_selector_tbN = torch.zeros(self.num_tasks, B, N, device=outs[0].device)
            curr_selector = [curr_selector_tbN[t] for t in range(self.num_tasks)]
        else:
            # (4b) Subsequent MoE blocks: compute selector normally
            curr_selector = []
            for task in range(self.num_tasks):
                task_emb = task_emb_T_E[task] if task_emb_T_E is not None else None
                curr_selector.append(self.router_selector(outs[task], task_emb=task_emb))  # [B, N]
            curr_selector_tbN = torch.stack(curr_selector, dim=0)  # [T, B, N]

        # (5) Build current state from current selector (need curr_bits for shared gate embedding)
        # Note: reuse_bits will be added later after detection
        curr_bits = selector_to_bits(curr_selector_tbN, thr=0.5)
        curr_count = popcount_bits(curr_bits, self.num_tasks)

        # ===== Reuse detection and cache initialization =====
        # 목적: Post-attention aggregation 이후 동일해진 토큰 위치에서 MoE 중복 연산 제거
        #
        # 핵심 아이디어:
        # 1. 이전 블록에서 shared였던 task들이 aggregation으로 동일한 representation을 갖게 됨
        # 2. 현재 블록에서도 shared를 선택한 task들은 동일한 MoE 입력을 받음
        # 3. 따라서 이 교집합에 속한 task들은 동일한 gate routing과 expert output을 갖게 됨
        # 4. 첫 번째 task가 계산하고 cache에 저장, 나머지 task들은 복사만 수행
        #
        # 재사용 가능 조건:
        # - prev_bits: 이전 블록에서 shared gate를 선택한 task 집합 (prev_state.shared_bits)
        # - curr_bits: 현재 블록에서 shared gate를 선택한 task 집합
        # - reuse_bits = prev_bits & curr_bits (교집합)
        # - reuse_possible_mask = popcount(reuse_bits) >= 2 (2개 이상 task가 공유)

        reuse_bits = None                # [B, N] int64 bitmask - 재사용 가능한 task 집합
        reuse_possible_mask = None       # [B, N] bool - 재사용 가능한 위치

        # Tensor caches (forward pass 내에서만 유지되는 로컬 캐시)
        # 목적: dict 대신 텐서를 사용하여 Python overhead 제거
        cached_moe_component = None      # [B, N, C] - 캐시된 MoE component (drop_path 적용 후)
        cached_valid_mask = None         # [B, N] bool - 해당 위치의 cache가 채워졌는지 여부

        if not is_first_moe:
            # 두 번째 MoE 블록부터는 reuse detection 수행
            # prev_state.shared_bits: 이전 블록에서 aggregation이 발생한 task 집합
            prev_bits = prev_state.shared_bits

            # compute_masks: prev_bits와 curr_bits의 교집합(reuse_bits)과
            # 재사용 가능한 위치(reuse_possible_mask)를 계산
            agg_needed, reuse_possible_mask, reuse_bits = compute_masks(
                prev_bits, curr_bits, self.num_tasks
            )

            # 재사용 가능한 위치가 있다면 tensor cache 초기화
            if reuse_possible_mask.any():
                B, N = reuse_possible_mask.shape
                device = outs[0].device
                C = outs[0].shape[-1]

                # 모든 위치에 대해 캐시 공간 할당
                # (실제로는 reuse_possible_mask가 True인 위치만 사용됨)
                # dtype 명시: fp16/bf16 학습 시 혼합 dtype 문제 방지
                cached_moe_component = torch.zeros(B, N, C, dtype=outs[0].dtype, device=device)
                cached_valid_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        # ===== END Reuse detection =====

        # (6) Create shared gate embedding from multi-hot
        #     CRITICAL: gate_shared_emb는 항상 생성되어야 함
        #     이유: gate_task_specific_dim > 0로 고정했고, d_gate_with_emb = dim + gate_task_specific_dim이므로
        #     self.shared_gate는 항상 dim+64 차원 입력을 기대함
        #     gate_task_represent가 None이면 런타임 shape mismatch 발생
        assert gate_task_represent is not None, (
            "gate_task_represent must not be None when gate_task_specific_dim > 0. "
            "Shared gate expects input dimension dim+gate_task_specific_dim but would receive dim."
        )
        mh = bits_to_multihot(curr_bits, self.num_tasks)  # [B, N, T]
        gate_shared_emb = gate_task_represent(mh)  # [B, N, E]

        # Ensure gate_shared_emb has correct dimension
        assert gate_shared_emb.shape[-1] == self.gate_task_specific_dim, (
            f"gate_shared_emb dimension mismatch: expected {self.gate_task_specific_dim}, "
            f"got {gate_shared_emb.shape[-1]}. Check gate_task_represent output dimension."
        )

        # (7) MoE MLP for all tasks WITH CACHE REUSE
        #     동적 대표 선택(dynamic representative selection) 전략:
        #     - 재사용 가능한 위치마다 "가장 먼저 도달한 shared task"가 대표가 됨
        #     - 대표 task: gate routing + expert computation 수행하고 cache에 저장
        #     - 이후 shared task들: cache에서 결과를 복사만 수행 (gate routing skip)
        #
        #     3가지 경우 처리:
        #     1. 재사용 없음 (can_reuse_mask=None): 정상 MoE 수행
        #     2. 전체 재사용 (need_compute_mask가 empty): cache에서 전부 복사
        #     3. 부분 재사용: 일부는 계산하고 cache 채우기, 일부는 복사
        # In single-gate mode (multi_gate=False), task_emb is required for routing
        # RouterSelector was initialized with d_task_emb=64, so task_emb must have dim 64
        if not self.multi_gate:
            assert task_emb_T_E is not None, (
                "task_emb_T_E must not be None in single-gate mode (multi_gate=False). "
                "_route_tokens expects task_emb for routing."
            )
            assert task_emb_T_E.shape[-1] == self.gate_task_specific_dim, (
                f"task_emb_T_E dimension mismatch: expected {self.gate_task_specific_dim}, "
                f"got {task_emb_T_E.shape[-1]}. Check task embedding dimension."
            )

        for task in range(self.num_tasks):
            task_emb = task_emb_T_E[task] if task_emb_T_E is not None else None

            # ===== Step 1: 이 task가 재사용 가능한 위치 확인 =====
            can_reuse_mask = None  # [B, N] bool - 이 task가 cache를 복사할 수 있는 위치
            cache_fill_mask = None  # [B, N] bool - 이 task가 계산 후 cache에 저장할 위치

            # 재사용 경로 진입 조건:
            # 1. reuse_bits가 존재하고
            # 2. cache 텐서들이 실제로 초기화되었어야 함 (cached_valid_mask is not None)
            # 이 가드가 없으면 reuse_possible_mask.any()=False인데 task_in_reuse.any()=True일 때 크래시
            if reuse_bits is not None and cached_valid_mask is not None:
                # reuse_bits에서 현재 task의 bit 추출
                # (이 task가 재사용 가능한 task 집합에 포함되는 위치)
                task_in_reuse = ((reuse_bits >> task) & 1).bool()  # [B, N]

                # 이 task가 재사용 가능한 위치가 실제로 하나라도 존재할 때만
                # cache-aware forward로 진입
                if task_in_reuse.any():
                    # Cache가 이미 채워진 위치에서만 재사용 가능
                    # (이전 task가 이미 계산을 완료한 위치)
                    can_reuse_mask = task_in_reuse & cached_valid_mask

                    # ===== Step 2: Cache 채우기 위치 결정 (cache_fill_mask) =====
                    # Cache에 저장할 위치: 재사용 교집합 내에서 shared이고 아직 cache 없는 위치만
                    # 1) reuse 대상 위치여야 하고 (task_in_reuse)
                    # 2) shared gate를 선택했고 (task_shared)
                    # 3) 아직 cache가 없는 위치 (이 task가 대표가 됨)
                    #
                    # task_in_reuse로 제한하지 않으면, reuse 교집합이 아닌 shared 위치까지
                    # 불필요하게 cache에 채워져서 메모리/성능 낭비 발생
                    task_shared = (curr_selector[task] > 0.5)  # [B, N]
                    cache_fill_mask = task_in_reuse & task_shared & ~cached_valid_mask

            # ===== Step 3: Cache-aware MoE 수행 =====
            # _forward_mlp_moe_with_cache가 다음을 처리:
            # - can_reuse_mask 위치: cache에서 복사 (gate routing skip)
            # - cache_fill_mask 위치: 계산 후 cache에 저장 (대표 task 역할)
            # - 나머지 위치: 정상적인 task-specific MoE 수행
            #
            # cached_moe_component를 받아서 다시 업데이트:
            # functional update로 gradient flow 유지하면서 task 간 공유
            outs[task], gate_info, compute_positions, cached_moe_component = self._forward_mlp_moe_with_cache(
                outs[task],
                selector_output=curr_selector[task],
                task_id=task,
                task_emb=task_emb,
                gate_shared_emb=gate_shared_emb,
                # Cache references
                # cached_valid_mask은 mutable (bool, gradient 무관)
                # cached_moe_component는 functional update 후 반환받아서 재전달
                cached_moe_component=cached_moe_component,
                cached_valid_mask=cached_valid_mask,
                # Masks
                can_reuse_mask=can_reuse_mask,
                cache_fill_mask=cache_fill_mask
            )

            # ===== Step 4: cv_loss 계산 (중복 방지) =====
            # gate_info는 _forward_mlp_moe_with_cache가 이미 올바르게 준비함:
            # - No caching: 모든 위치 포함 (dense)
            # - Full reuse: empty (cv_loss 기여 0)
            # - Partial reuse: 실제 계산한 위치만 포함 (sparse, compute_positions에 해당)
            #
            # 따라서 gate_info를 필터링하지 않고 그대로 사용하면 됨
            # (필터링하면 sparse gate_info에서 shape mismatch 발생)
            aux['cv_loss'] = aux['cv_loss'] + self._compute_cv_loss(gate_info)

        # (8) MLP-post aggregation (using curr_bits)
        if self.mlp_post_aggr is not None:
            mlp_agg_needed = (curr_count >= 2)  # [B, N]
            curr_task_masks = bits_to_task_masks(curr_bits, self.num_tasks)
            outs = self.mlp_post_aggr(outs, curr_task_masks, mlp_agg_needed)

            # Stats
            aux['stats']['shared_positions_curr'] = (curr_count > 0).sum().item()
            aux['stats']['aggregated_positions_mlp'] = mlp_agg_needed.sum().item()

        # (9) Build curr_state with curr_bits for next block
        # curr_bits를 shared_bits로 저장: 다음 블록에서 prev_state.shared_bits로 읽혀서
        # post-attn aggregation과 reuse detection에 사용됨
        curr_state = self._build_router_state(curr_selector_tbN)

        return outs, curr_state, aux


class TokenVisionTransformerMoE(nn.Module):
    def __init__(self, model_name='vit_large_patch16_384', img_size=384, patch_size=16, in_chans=3, embed_dim=1024, depth=24,
                    num_heads=16, num_classes=19, mlp_ratio=4., qkv_bias=True, qk_scale=None,  representation_size=None, distilled=False, 
                    drop_rate=0.1, attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None, norm_layer=partial(nn.LayerNorm, eps=1e-6), norm_cfg=None,
                    pos_embed_interp=False, random_init=False, align_corners=False,
                    act_layer=None, weight_init='', moe_mlp_ratio=-1, moe_experts=8, moe_top_k=4, world_size=2, gate_dim=-1,
                    moe_gate_type="token_noisy_vmoe", vmoe_noisy_std=1, gate_task_specific_dim=64,multi_gate=False,
                    num_experts_pertask = -1, num_tasks = -1, gate_input_ahead=False, 
                    **kwargs):
        super(TokenVisionTransformerMoE, self).__init__(**kwargs)
        # print(hybrid_backbone is None)
        self.model_name = model_name
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_features = self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.hybrid_backbone = hybrid_backbone
        
        self.norm_cfg = norm_cfg
        self.pos_embed_interp = pos_embed_interp
        self.random_init = random_init
        self.align_corners = align_corners
        self.h = int(self.img_size[0]/self.patch_size)
        self.w = int(self.img_size[1]/self.patch_size)

        self.num_stages = self.depth
        # Only keep the last stage output to save memory (decoder only uses the last one)
        self.out_indices = (self.num_stages - 1,)

        self.num_token = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.norm_layer = norm_layer
        self.moe_experts = moe_experts
        self.moe_top_k = moe_top_k
        self.multi_gate = multi_gate

        self.patch_embed = PatchEmbed(
            img_size=self.img_size, patch_size=self.patch_size, in_chans=self.in_chans, embed_dim=self.embed_dim)
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(
            1, self.num_patches + 1, self.embed_dim))
        self.pos_drop = nn.Dropout(p=self.drop_rate)

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate,
                                                self.depth)]  # stochastic depth decay rule
        blocks = []
        # Todo: param에 num_task 추가.
        self.num_tasks = num_tasks
        self.gate_task_specific_dim = gate_task_specific_dim
        self.gate_input_ahead = gate_input_ahead
        # if self.gate_task_specific_dim<0 or self.multi_gate:
            # self.gate_task_represent = None
        # else:

        self.gate_task_represent = new_Mlp(in_features=self.num_tasks, hidden_features=int(self.gate_task_specific_dim), out_features=self.gate_task_specific_dim,)

            # self.gamma = nn.Parameter(torch.Tensor([1]), requires_grad=True)
        for i in range(self.depth):
            if i % 2 == 0:
                # Non-MoE block: also needs num_tasks for attn_post_aggr
                blocks.append(Block(dim=self.embed_dim, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias, qk_scale=self.qk_scale,
                drop=self.drop_rate, attn_drop=self.attn_drop_rate, drop_path=dpr[i], norm_layer=self.norm_layer,
                num_tasks=self.num_tasks))
            else:
                # MoE block
                blocks.append(Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                              moe=True, moe_mlp_ratio=moe_mlp_ratio, moe_experts=moe_experts, moe_top_k=moe_top_k, moe_gate_dim=gate_dim, world_size=world_size,
                              moe_gate_type=moe_gate_type, vmoe_noisy_std=vmoe_noisy_std,
                              gate_task_specific_dim=self.gate_task_specific_dim,multi_gate=self.multi_gate,
                              num_experts_pertask = num_experts_pertask, num_tasks = self.num_tasks,
                              gate_input_ahead = self.gate_input_ahead))
        self.blocks = nn.Sequential(*blocks)

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        # self.repr = nn.Linear(embed_dim, representation_size)
        # self.repr_act = nn.Tanh()
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # # Classifier head(s)
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        # self.head_dist = None
        # if distilled:
        #     self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()
        
        # Weight init
        assert weight_init in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in weight_init else 0.
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if weight_init.startswith('jax'):
            # leave cls token as zeros to match jax impl
            for n, m in self.named_modules():
                _init_vit_weights(m, n, head_bias=head_bias, jax_impl=True)

        self.init_weights()
        self.idx = 0

        # Get wandb logger instance (will be None if not initialized)
        try:
            from utils.wandb_logger import get_wandb_logger
            self.wandb_logger = get_wandb_logger
        except:
            self.wandb_logger = None

    def init_weights(self, pretrained=None):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        self.default_cfg = default_cfgs[self.model_name]
        load_pretrained_pos_emb(self, num_classes=self.num_classes, in_chans=self.in_chans, pos_embed_interp=self.pos_embed_interp,
        num_patches=self.patch_embed.num_patches, align_corners=self.align_corners, img_h=self.h, img_w=self.w)
        if not self.random_init:
            self.default_cfg = default_cfgs[self.model_name]
            if self.model_name in ['vit_small_patch16_224', 'vit_base_patch16_224']:
                load_pretrained(self, num_classes=self.num_classes, in_chans=self.in_chans, pos_embed_interp=self.pos_embed_interp,
                                num_patches=self.patch_embed.num_patches, align_corners=self.align_corners, filter_fn=self._conv_filter, img_h=self.h, img_w=self.w)
            else:
                load_pretrained(self, num_classes=self.num_classes, in_chans=self.in_chans, pos_embed_interp=self.pos_embed_interp,
                                num_patches=self.patch_embed.num_patches, align_corners=self.align_corners, img_h=self.h, img_w=self.w)
        else:
            print('Initialize weight randomly')


    @property
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x, gate_inp=None, sem=None):
        """
        Simplified Forward:
        - Delegates block-level logic to Block.forward
        - Only handles: input prep, block iteration, state passing, aux aggregation
        """
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Create task embedding cache: one-hot -> embedding for each task
        # task_emb_T_E: [T, E] - cached embeddings for all tasks
        # This is computed once and reused across all blocks
        eye = torch.eye(self.num_tasks, device=x.device, dtype=x.dtype)  # [T, T]
        task_emb_T_E = self.gate_task_represent(eye)  # [T, E]

        # Initialize dict to store outputs for each task
        outs = {task: x for task in range(self.num_tasks)}

        # RouterState: stores previous MoE block selector (None for first MoE block)
        prev_state = None

        # Initialize total cv_loss
        total_cv_loss = x.new_tensor(0.0)

        # Initialize statistics counters
        stats = {
            'moe_blocks': 0,
            'total_positions': 0,
            'shared_positions_prev': 0,
            'aggregated_positions_attn': 0,
            'shared_positions_curr': 0,
            'aggregated_positions_mlp': 0,
        }

        # Block iteration with state management
        for blk in self.blocks:
            outs, curr_state, aux = blk(
                outs, prev_state,
                gate_inp=gate_inp,
                task_emb_T_E=task_emb_T_E,
                gate_task_represent=self.gate_task_represent
            )

            # Accumulate cv_loss
            total_cv_loss = total_cv_loss + aux['cv_loss']

            # Accumulate stats (only MoE blocks contribute meaningful stats)
            if blk.moe:
                stats['moe_blocks'] += 1
                for key in ['total_positions', 'shared_positions_prev', 'aggregated_positions_attn',
                           'shared_positions_curr', 'aggregated_positions_mlp']:
                    stats[key] += aux['stats'].get(key, 0)

            # Update state: current state becomes prev for next block
            # Only MoE blocks produce curr_state; NonMoE blocks return None
            if curr_state is not None:
                prev_state = curr_state

        # Calculate statistics ratios
        if stats['total_positions'] > 0:
            stats['attn_aggregation_ratio'] = stats['aggregated_positions_attn'] / stats['total_positions']
            stats['mlp_aggregation_ratio'] = stats['aggregated_positions_mlp'] / stats['total_positions']
            stats['shared_prev_ratio'] = stats['shared_positions_prev'] / stats['total_positions']
            stats['shared_curr_ratio'] = stats['shared_positions_curr'] / stats['total_positions']
        else:
            stats['attn_aggregation_ratio'] = 0.0
            stats['mlp_aggregation_ratio'] = 0.0
            stats['shared_prev_ratio'] = 0.0
            stats['shared_curr_ratio'] = 0.0

        # Log MoE stats to wandb if available (only during training)
        if self.training and stats['total_positions'] > 0 and self.wandb_logger is not None:
            logger = self.wandb_logger()
            if logger is not None:
                logger.log_moe_stats(stats)

        return outs, total_cv_loss

    # def to_2D(self, x):
    #     n, hw, c = x.shape
    #     h = w = int(math.sqrt(hw))
    #     x = x.transpose(1, 2).reshape(n, c, h, w)
    #     return x

    # def to_1D(self, x):
    #     n, c, h, w = x.shape
    #     x = x.reshape(n, c, -1).transpose(1, 2)
    #     return x

    def _conv_filter(self, state_dict, patch_size=16):
        """ convert patch embedding weight from manual patchify + linear proj to conv"""
        out_dict = {}
        for k, v in state_dict.items():
            if 'patch_embed.proj.weight' in k:
                v = v.reshape((v.shape[0], 3, patch_size, patch_size))
            out_dict[k] = v
        return out_dict


def _init_vit_weights(m, n: str = '', head_bias: float = 0., jax_impl: bool = False):#
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(m, nn.Linear):
        if n.startswith('head'):
            nn.init.zeros_(m.weight)
            nn.init.constant_(m.bias, head_bias)
        elif n.startswith('pre_logits'):
            lecun_normal_(m.weight)
            nn.init.zeros_(m.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    if 'mlp' in n:
                        nn.init.normal_(m.bias, std=1e-6)
                    else:
                        nn.init.zeros_(m.bias)
            else:
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    elif jax_impl and isinstance(m, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)