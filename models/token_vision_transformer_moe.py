import torch
import torch.nn as nn
import torch.nn.functional as F
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
from models.gate_funs.ckpt_noisy_gate_vmoe import NoisyGate_VMoE
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

class TokenBlock(nn.Module):

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
        self.tot_expert = moe_experts * world_size
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
                # single gate: concat task embedding to distinguish
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

    # ==========================================================================
    # Aux
    # ==========================================================================
    def _init_aux(self, outs: dict):
        """
        Initialize auxiliary outputs (loss/stat tracking) on the same device/dtype as outs tensors.
        outs: {task_id: [B, N, C]}
        """
        # Pick a reference tensor safely
        if 0 in outs:
            ref = outs[0]
        else:
            # fall back to first key
            first_k = next(iter(outs.keys()))
            ref = outs[first_k]

        aux = {
            "cv_loss": ref.new_tensor(0.0),
            "stats": {
                "total_positions": 0,
                "shared_positions": 0,
                "shared_tasktoken_count": 0,
                "aggregated_positions": 0,
            },
        }

        # Fill total_positions if shape is available
        if ref is not None and ref.dim() >= 2:
            B, N = ref.shape[0], ref.shape[1]
            aux["stats"]["total_positions"] = int(B * N)

        return aux

    def _merge_aux(self, aux: dict, moe_aux: dict | None) -> dict:
        """
        Merge per-stage aux dicts.

        Expected format (both aux and moe_aux):
        {
            "cv_loss": Tensor scalar,
            "stats": { ... numeric counters ... }   # optional / nested ok
        }

        Rules:
        - cv_loss: summed
        - stats: recursively add numeric values if same key exists
        - missing keys are copied over
        """
        if moe_aux is None:
            return aux

        # --- cv_loss ---
        if "cv_loss" in moe_aux and moe_aux["cv_loss"] is not None:
            if "cv_loss" not in aux or aux["cv_loss"] is None:
                aux["cv_loss"] = moe_aux["cv_loss"]
            else:
                aux["cv_loss"] = aux["cv_loss"] + moe_aux["cv_loss"]

        # --- stats ---
        def _merge_stats(dst: dict, src: dict):
            for k, v in src.items():
                if k not in dst:
                    dst[k] = v
                    continue

                dv = dst[k]

                # nested dict: recurse
                if isinstance(dv, dict) and isinstance(v, dict):
                    _merge_stats(dv, v)
                    continue

                # numeric add (int/float/0-dim tensor)
                if isinstance(dv, (int, float)) and isinstance(v, (int, float)):
                    dst[k] = dv + v
                    continue

                if torch.is_tensor(dv) and torch.is_tensor(v):
                    # if they're scalar tensors, sum them; otherwise overwrite (safer)
                    if dv.ndim == 0 and v.ndim == 0:
                        dst[k] = dv + v
                    else:
                        dst[k] = v
                    continue

                # fallback: overwrite with src
                dst[k] = v

        if "stats" in moe_aux and moe_aux["stats"] is not None:
            if "stats" not in aux or aux["stats"] is None:
                aux["stats"] = {}
            _merge_stats(aux["stats"], moe_aux["stats"])

        # copy any other top-level keys (optional)
        for k, v in moe_aux.items():
            if k in ("cv_loss", "stats"):
                continue
            if k not in aux:
                aux[k] = v
            else:
                # if both are dicts, deep-merge; else overwrite
                if isinstance(aux[k], dict) and isinstance(v, dict):
                    _merge_stats(aux[k], v)
                else:
                    aux[k] = v

        return aux
    # ==========================================================================
    # Aggregation
    # ==========================================================================
    def prev_aggregate_stage(self, outs: dict, prev_state, aux: dict):
        """
        Post-attention aggregation stage (uses prev_state.shared_bits).
        Aggregates only where prev_count >= 2.

        Returns:
            outs: updated outs
            aux: updated aux (stats)
        """
        # No previous state or no aggregation module => no-op
        if prev_state is None or self.attn_post_aggr is None:
            return outs, aux

        prev_bits = prev_state.shared_bits  # [B, N] int64 bitmask
        prev_count = popcount_bits(prev_bits, self.num_tasks)  # [B, N] int counts
        attn_agg_needed = (prev_count >= 2)  # [B, N] bool

        # Build per-task masks from prev_bits
        prev_task_masks = bits_to_task_masks(prev_bits, self.num_tasks)  # list/tuple of [B, N] bool

        # Perform aggregation (AggregationStage should handle dict outs)
        outs = self.attn_post_aggr(outs, prev_task_masks, attn_agg_needed)

        return outs, aux

    def after_moe_aggregate_stage(self, outs: dict, curr_bits: torch.Tensor, aux: dict):
        """
        MoE 이후 aggregation 단계.
        Args:
            outs: {task_id: [B, N, C]}
            curr_bits: [B, N] int64 bitmask (현재 블록 selector 기반)
            aux: dict
        Returns:
            outs, aux
        """
        if self.mlp_post_aggr is None:
            return outs, aux

        curr_count = popcount_bits(curr_bits, self.num_tasks)  # [B, N]
        mlp_agg_needed = (curr_count >= 2)  # [B, N]

        curr_task_masks = bits_to_task_masks(curr_bits, self.num_tasks)
        outs = self.mlp_post_aggr(outs, curr_task_masks, mlp_agg_needed)

        # stats 업데이트
        stats = aux.setdefault("stats", {})
        stats["shared_positions"] = stats.get("shared_positions", 0) + int((curr_count > 0).sum().item())
        stats["shared_tasktoken_count"] = stats.get("shared_tasktoken_count", 0) + int(curr_count.sum().item())
        stats["aggregated_positions"] = stats.get("aggregated_positions", 0) + int(mlp_agg_needed.sum().item())

        return outs, aux
    
    # ==========================================================================
    # LN1 + Attn + Residual
    # ==========================================================================
    def attn_stage(self, outs: dict):
        """
        Attention stage orchestration:
        for each task: LN1 -> Attn -> Residual  (checkpointing should live inside self._forward_attn)
        outs: {task_id: [B, N, C]}
        returns: outs (updated)
        """

        for task in range(self.num_tasks):
            outs[task] = self._forward_attn(outs[task])
        return outs
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
    def _attn_only(self, x):
        """Internal checkpointed function: attention only"""
        return x + self.drop_path(self.attn(self.norm1(x)))
    # ==========================================================================
    # LN2 + MLP + Residual
    # ==========================================================================
    def dense_mlp_stage(self, outs: dict):
        """
        Dense MLP stage orchestration (non-MoE):
        for each task: LN2 -> MLP -> Residual  (checkpointing should live inside self._forward_mlp_dense)

        Args:
            outs: {task_id: Tensor[B, N, C]}

        Returns:
            outs: updated dict
        """


        for task in range(self.num_tasks):
            outs[task] = self._forward_mlp_dense(outs[task])
        return outs
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
    def _mlp_dense_inner(self, x):
        """Internal checkpointed function: Dense MLP only"""
        normed_x = self.norm2(x)
        return x + self.drop_path(self.mlp(normed_x))
    # ==========================================================================
    # LN2 + MoE + Residual
    # ==========================================================================
    def router_stage(self, outs: dict, prev_state: "RouterState|None",
                    task_emb_T_E=None, gate_task_represent=None):
        """
        Router stage: selector -> curr_bits -> reuse detection -> shared gate embedding

        Returns:
            router_out: dict with at least
                - is_first_moe: bool
                - selector: list[T] of [B, N] (per-task)
                - selector_tbN: [T, B, N]
                - curr_bits: [B, N] int64
                - curr_count: [B, N] int64/long (popcount)
                - reuse_bits: [B, N] int64 or None
                - reuse_possible_mask: [B, N] bool or None
                - cached_moe_component: [B, N, C] or None
                - cached_valid_mask: [B, N] bool or None
                - gate_shared_emb: [B, N, E] or None
        """
        assert self.moe, "router_stage is only used when moe=True"

        # -------- basic shapes --------
        B, N, C = outs[0].shape
        device = outs[0].device
        dtype = outs[0].dtype
        T = self.num_tasks

        is_first_moe = (prev_state is None)

        # -------- (1) selector 계산/강제 --------
        if is_first_moe:
            # 첫 MoE 블록은 task-specific만 (shared=0)
            selector_tbN = torch.zeros(T, B, N, device=device, dtype=outs[0].dtype)
            selector_list = [selector_tbN[t] for t in range(T)]
        else:
            selector_list = []
            for t in range(T):
                # RouterSelector가 task_emb를 요구하는 구조면 여기서 넣어줌
                task_emb = task_emb_T_E[t] if task_emb_T_E is not None else None
                selector_list.append(self.router_selector(outs[t], task_emb=task_emb))  # [B, N]
            selector_tbN = torch.stack(selector_list, dim=0)  # [T, B, N]

        # -------- (2) curr_bits / count --------
        curr_bits = selector_to_bits(selector_tbN, thr=0.5)          # [B, N] int64
        curr_count = popcount_bits(curr_bits, T)                     # [B, N]

        # -------- (3) reuse detection + cache init --------
        reuse_bits = None
        reuse_possible_mask = None
        cached_moe_component = None
        cached_valid_mask = None

        if not is_first_moe:
            prev_bits = prev_state.shared_bits  # [B, N] int64
            # compute_masks는 네가 이미 쓰던 그대로 호출
            # (agg_needed는 여기선 router_stage에서 사용 안 하면 버려도 됨)
            _, reuse_possible_mask, reuse_bits = compute_masks(prev_bits, curr_bits, T)

            if reuse_possible_mask is not None and reuse_possible_mask.any():
                cached_moe_component = torch.zeros(B, N, C, device=device, dtype=dtype)
                cached_valid_mask = torch.zeros(B, N, device=device, dtype=torch.bool)

        # -------- (4) shared gate embedding (gate_shared_emb) --------
        gate_shared_emb = None

        # 주의:
        # - shared_gate 입력이 dim+gate_task_specific_dim으로 고정이라면,
        #   "shared token이 존재하는 블록"에서는 gate_shared_emb가 반드시 필요.
        # - 첫 MoE 블록은 selector=0 강제라 shared token 자체가 없으므로 생략 가능.
        if not is_first_moe:
            if self.gate_task_specific_dim > 0:
                assert gate_task_represent is not None, (
                    "gate_task_represent must be provided when shared routing can happen "
                    "(gate_task_specific_dim > 0 and not first MoE block)."
                )
                mh = bits_to_multihot(curr_bits, T)                  # [B, N, T]
                gate_shared_emb = gate_task_represent(mh)            # [B, N, E]
                assert gate_shared_emb.shape[-1] == self.gate_task_specific_dim, (
                    f"gate_shared_emb dim mismatch: expected {self.gate_task_specific_dim}, "
                    f"got {gate_shared_emb.shape[-1]}"
                )

        # -------- single-gate safety (optional but recommended) --------
        # multi_gate=False인 경우, task-specific gate 입력에 task_emb concat이 필요할 수 있음
        # (너의 _route_tokens 구현 기준)
        if not self.multi_gate:
            assert task_emb_T_E is not None, (
                "task_emb_T_E must not be None when multi_gate=False (single gate needs task embedding)."
            )
            assert task_emb_T_E.shape[-1] == self.gate_task_specific_dim, (
                f"task_emb_T_E dim mismatch: expected {self.gate_task_specific_dim}, "
                f"got {task_emb_T_E.shape[-1]}"
            )

        router_out = {
            "is_first_moe": is_first_moe,
            "selector": selector_list,                 # list[T] of [B, N]
            "selector_tbN": selector_tbN,              # [T, B, N]
            "curr_bits": curr_bits,                    # [B, N] int64
            "curr_count": curr_count,                  # [B, N]
            "reuse_bits": reuse_bits,                  # [B, N] int64 or None
            "reuse_possible_mask": reuse_possible_mask,# [B, N] bool or None
            "cached_moe_component": cached_moe_component,  # [B, N, C] or None
            "cached_valid_mask": cached_valid_mask,        # [B, N] or None
            "gate_shared_emb": gate_shared_emb,        # [B, N, E] or None
            "task_emb_T_E": task_emb_T_E,              # (forward에서 따로 넣던거 여기서 포함)
        }
        return router_out
    def moe_stage(self, outs: dict, router_out: dict):
        """
        routing + experts + (shared) drop_path + cache reuse

        router_out expected keys:
        - selector: List[Tensor[B,N]] length T     # <- (changed from curr_selector)
        - gate_shared_emb: Tensor[B,N,E] or None   # <- can be None (e.g., first MoE)
        - reuse_bits: Tensor[B,N] int64 or None
        - cached_moe_component: Tensor[B,N,C] or None   (cache for mlp_drop(expert_out), NOT drop_path)
        - cached_valid_mask: Tensor[B,N] bool or None
        - task_emb_T_E: Tensor[T,E] or None   (IMPORTANT for single-gate)
        """
        T = self.num_tasks
        device = outs[0].device
        B, N, C = outs[0].shape

        selector_list = router_out["selector"]  # List[Tensor[B,N]]
        gate_shared_emb = router_out.get("gate_shared_emb", None)
        reuse_bits = router_out.get("reuse_bits", None)

        cached = router_out.get("cached_moe_component", None)
        cached_valid = router_out.get("cached_valid_mask", None)

        task_emb_T_E = router_out.get("task_emb_T_E", None)
        if not self.multi_gate:
            assert task_emb_T_E is not None, "single-gate(multi_gate=False) requires task_emb_T_E in router_out"

        # ---- shared drop-path mask sampled ONCE for this block ----
        shared_dp_mask = None
        dummy = outs[0].new_zeros((B, N, C))
        _, shared_dp_mask = self._shared_drop_path(dummy)

        moe_aux = {
            "cv_loss": outs[0].new_tensor(0.0),
            "stats": {
                "computed_tokens": 0,
                "reused_tokens": 0,
            }
        }

        for task in range(T):
            x = outs[task]                 # [B,N,C]
            selector = selector_list[task] # [B,N]
            normed = self.norm2(x)         # [B,N,C]

            task_emb = task_emb_T_E[task] if task_emb_T_E is not None else None

            # ---- compute reuse masks for this task ----
            can_reuse_mask = None
            cache_fill_mask = None

            if (reuse_bits is not None) and (cached_valid is not None):
                task_in_reuse = ((reuse_bits >> task) & 1).bool()  # [B,N]
                if task_in_reuse.any():
                    can_reuse_mask = task_in_reuse & cached_valid

                    task_shared = (selector > 0.5)
                    cache_fill_mask = task_in_reuse & task_shared & (~cached_valid)

            # ---- compute_mask: tokens we must actually run gate+experts on ----
            if can_reuse_mask is None:
                compute_mask = torch.ones((B, N), dtype=torch.bool, device=device)
            else:
                task_specific = (selector <= 0.5)
                task_shared = (selector > 0.5)
                compute_mask = task_specific | (task_shared & (~can_reuse_mask))

            # ---- full reuse (nothing to compute) ----
            if not compute_mask.any():
                assert cached is not None, "compute_mask empty but cache is None"
                expert_drop_out = cached
                moe_aux["stats"]["reused_tokens"] += int((can_reuse_mask is not None) and can_reuse_mask.sum().item())

                if shared_dp_mask is not None:
                    keep_prob = 1.0 - float(self.drop_path.drop_prob)
                    moe_component = expert_drop_out / keep_prob * shared_dp_mask
                else:
                    moe_component = expert_drop_out

                outs[task] = x + moe_component
                continue

            # ---- Gather compute tokens ----
            compute_flat = compute_mask.flatten()             # [B*N]
            K = int(compute_flat.sum().item())
            moe_aux["stats"]["computed_tokens"] += K

            normed_flat = normed.reshape(B * N, C)
            normed_compute = normed_flat[compute_flat]        # [K,C]

            selector_flat = selector.flatten()
            selector_compute = selector_flat[compute_flat]    # [K]

            if gate_shared_emb is None:
                shared_emb_compute = None
            else:
                shared_emb_flat = gate_shared_emb.reshape(B * N, -1)
                shared_emb_compute = shared_emb_flat[compute_flat]  # [K,E]

            # ---- gate routing only for computed tokens ----
            gate_idx, gate_score, gate_info = self._route_tokens(
                normed_compute,
                selector_output=selector_compute,
                task_id=task,
                task_emb=task_emb,
                gate_shared_emb=shared_emb_compute
            )

            # ---- experts forward for computed tokens ----
            expert_out_compute = self.mlp(normed_compute, gate_idx, gate_score)  # [K,C]

            # ---- Scatter back to dense ----
            expert_out = torch.zeros((B * N, C), device=device, dtype=x.dtype)
            expert_out[compute_flat] = expert_out_compute
            expert_out = expert_out.view(B, N, C)  # [B,N,C]

            # ---- mlp_drop (ONLY computed path) ----
            expert_drop_out = self.mlp_drop(expert_out)  # [B,N,C]

            # ---- cache populate ----
            if (cache_fill_mask is not None) and cache_fill_mask.any():
                assert cached is not None, "cache_fill_mask exists but cached tensor is None"
                cached_valid[cache_fill_mask] = True

                cached = torch.where(
                    cache_fill_mask.unsqueeze(-1),
                    expert_drop_out,
                    cached
                )

            # ---- merge reuse positions from cache ----
            if (can_reuse_mask is not None) and can_reuse_mask.any():
                assert cached is not None, "can_reuse_mask exists but cached tensor is None"
                moe_aux["stats"]["reused_tokens"] += int(can_reuse_mask.sum().item())
                expert_drop_out = torch.where(
                    can_reuse_mask.unsqueeze(-1),
                    cached,
                    expert_drop_out
                )

            # ---- apply SHARED drop_path mask ----
            if shared_dp_mask is not None:
                keep_prob = 1.0 - float(self.drop_path.drop_prob)
                moe_component = expert_drop_out / keep_prob * shared_dp_mask
            else:
                moe_component = expert_drop_out

            outs[task] = x + moe_component

            # cv loss from ONLY computed tokens gate_info (already K-sized)
            moe_aux["cv_loss"] = moe_aux["cv_loss"] + self._compute_cv_loss(gate_info)

        # write back cache (functional update)
        router_out["cached_moe_component"] = cached
        return outs, moe_aux
    def _shared_drop_path(self, x: torch.Tensor):
        """
        Shared stochastic depth mask (same for all tasks in this block).
        - Typical DropPath is per-sample (B,1,1) broadcast.
        Returns:
        x_dropped, mask  (mask is None if disabled)
        """
        # Identity or eval mode: no drop
        if (not self.training) or (not hasattr(self.drop_path, "drop_prob")):
            return x, None

        drop_prob = float(getattr(self.drop_path, "drop_prob", 0.0))
        if drop_prob <= 0.0:
            return x, None

        keep_prob = 1.0 - drop_prob
        B = x.shape[0]
        # per-sample mask, broadcast over tokens & channels
        mask = x.new_empty((B, 1, 1)).bernoulli_(keep_prob)
        x = x / keep_prob * mask
        return x, mask
    # ==========================================================================
    # forward
    # ==========================================================================
    def _state_for_next_block(self, router_out: dict) -> RouterState:
        # 다음 블록에서 필요한 최소: shared_bits, shared_selector(선택)
        curr_bits = router_out["curr_bits"]                  # [B, N] int64
        selector_tbN = router_out.get("selector_tbN", None)  # [T, B, N] or None

        return RouterState(
            shared_bits=curr_bits,
            shared_selector=None if selector_tbN is None else selector_tbN.detach()
        )

    def forward(self, outs: dict, prev_state: RouterState | None,
                task_emb_T_E=None, gate_task_represent=None):
        aux = self._init_aux(outs)

        outs = self.attn_stage(outs)                           # LN1+Attn+res
        outs, aux = self.prev_aggregate_stage(outs, prev_state, aux)

        if not self.moe:
            outs = self.dense_mlp_stage(outs)                  # LN2+MLP+res
            return outs, prev_state, aux

        # router_out이 task_emb_T_E까지 포함하도록 router_stage에서 처리
        router_out = self.router_stage(
            outs, prev_state,
            task_emb_T_E=task_emb_T_E,
            gate_task_represent=gate_task_represent
        )

        outs, moe_aux = self.moe_stage(outs, router_out)       # routing+experts(+shared drop_path cache)
        aux = self._merge_aux(aux, moe_aux)

        outs, aux = self.after_moe_aggregate_stage(outs, router_out["curr_bits"], aux)

        curr_state = self._state_for_next_block(router_out)
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

        if self.gate_task_specific_dim <= 0:
            raise ValueError("gate_task_specific_dim must be > 0 for TokenVisionTransformerMoE MoE routing.")
        if self.num_tasks <= 0:
            raise ValueError("num_tasks must be > 0 to build task embeddings for routing.")

        self.gate_task_represent = new_Mlp(
            in_features=self.num_tasks,
            hidden_features=int(self.gate_task_specific_dim),
            out_features=self.gate_task_specific_dim,
        )

            # self.gamma = nn.Parameter(torch.Tensor([1]), requires_grad=True)
        for i in range(self.depth):
            if i % 2 == 0:
                # Non-MoE block: also needs num_tasks for attn_post_aggr
                blocks.append(TokenBlock(dim=self.embed_dim, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias, qk_scale=self.qk_scale,
                drop=self.drop_rate, attn_drop=self.attn_drop_rate, drop_path=dpr[i], norm_layer=self.norm_layer,
                num_tasks=self.num_tasks))
            else:
                # MoE block
                blocks.append(TokenBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
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

    def forward(self, x, sem=None):
        """
        Simplified Forward (refactored for new TokenBlock API):
        - Prepares tokens
        - Builds per-task outputs dict (safe: clone per task)
        - Builds task_emb_T_E once per forward (stable: fp32 one-hot -> cast back)
        - Iterates blocks with RouterState passing
        - Aggregates aux (cv_loss + stats)
        """
        B = x.shape[0]

        # ---- patch + cls + pos ----
        x = self.patch_embed(x)                    # [B, C, H', W'] or [B, embed, h, w] depending on PatchEmbed
        x = x.flatten(2).transpose(1, 2)           # [B, N, C]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)      # [B, 1+N, C]
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # ---- task embedding cache: one-hot (fp32) -> gate_task_represent -> cast to x.dtype ----
        # task_emb_T_E: [T, E]
        if self.gate_task_represent is None:
            raise RuntimeError("gate_task_represent is not initialized; check gate_task_specific_dim/num_tasks.")
        eye = torch.eye(self.num_tasks, device=x.device, dtype=torch.float32)
        task_emb_T_E = self.gate_task_represent(eye).to(dtype=x.dtype)

        # ---- per-task outputs (clone to avoid accidental in-place cross-talk) ----
        outs = {task: x.clone() for task in range(self.num_tasks)}

        # ---- RouterState from previous MoE block ----
        prev_state = None

        # ---- totals ----
        total_cv_loss = x.new_tensor(0.0)
        T = self.num_tasks
        stats = {
            "moe_blocks": 0,
            "total_positions": 0,           # B*N per MoE block, summed
            "shared_positions": 0,          # #positions where curr_count > 0
            "shared_tasktoken_count": 0,    # sum(curr_count) across positions
            "aggregated_positions": 0,      # #positions where curr_count >= 2
            "computed_tokens": 0,           # task-token granularity
            "reused_tokens": 0,             # task-token granularity
        }

        # ---- block iteration ----
        for blk in self.blocks:
            outs, curr_state, aux = blk(
                outs,
                prev_state,
                task_emb_T_E=task_emb_T_E,
                gate_task_represent=self.gate_task_represent,
            )

            # cv loss
            total_cv_loss = total_cv_loss + aux.get("cv_loss", x.new_tensor(0.0))

            # stats (count only MoE blocks)
            if getattr(blk, "moe", False):
                stats["moe_blocks"] += 1
                for k in [
                    "total_positions",
                    "shared_positions",
                    "shared_tasktoken_count",
                    "aggregated_positions",
                    "computed_tokens",
                    "reused_tokens",
                ]:
                    stats[k] += aux.get("stats", {}).get(k, 0)

            # state for next block
            prev_state = curr_state

        # ---- ratios ----
        # A) position-scale ratios (denom = B*N*moe_blocks)
        total_pos = float(stats["total_positions"])
        if total_pos > 0:
            stats["shared_position_ratio"] = stats["shared_positions"] / total_pos
            stats["aggregation_ratio"] = stats["aggregated_positions"] / total_pos
        else:
            stats["shared_position_ratio"] = 0.0
            stats["aggregation_ratio"] = 0.0

        # B) task-token-scale ratio (denom = B*N*T*moe_blocks)
        total_tasktoken = float(stats["total_positions"]) * T
        if total_tasktoken > 0:
            stats["shared_tasktoken_ratio"] = stats["shared_tasktoken_count"] / total_tasktoken
        else:
            stats["shared_tasktoken_ratio"] = 0.0

        # C) compute/reuse ratios (denom = computed + reused)
        total_dispatched = stats["computed_tokens"] + stats["reused_tokens"]
        if total_dispatched > 0:
            stats["reuse_ratio"] = stats["reused_tokens"] / total_dispatched
            stats["compute_ratio"] = stats["computed_tokens"] / total_dispatched
        else:
            stats["reuse_ratio"] = 0.0
            stats["compute_ratio"] = 1.0

        # ---- wandb logging (train only) ----
        if self.training and stats["total_positions"] > 0 and self.wandb_logger is not None:
            logger = self.wandb_logger()
            if logger is not None:
                logger.log_moe_stats(stats)

        return outs, total_cv_loss

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