import torch
import torch.nn as nn
from functools import partial
import math
from itertools import repeat
# from torch._six import container_abcs
import collections.abc
import warnings
from collections import OrderedDict
from utils.helpers import load_pretrained,load_pretrained_pos_emb
from models.ckpt_custom_moe_layer import FMoETransformerMLP
# from .layers import DropPath, to_2tuple, trunc_normal_
from timm.layers  import lecun_normal_
# from ..builder import BACKBONES
import numpy as np
from collections import Counter
from models.gate_funs.noisy_gate import NoisyGate
from models.gate_funs.ckpt_noisy_gate_vmoe import NoisyGate_VMoE
from torch.utils.checkpoint import checkpoint

a=[[0],[1,17,18,19,20],[2,12,13,14,15,16],[3,9,10,11],[4,5],[6,7,8,38],[21,22,23,24,25,26,39],[27,28,29,30,31,32,33,34,35,36,37]]

def _gates_to_load(gates):
    """Compute the true load per expert, given the gates.
    The load is the number of examples for which the corresponding gate is >0.
    Args:
        gates: a `Tensor` of shape [batch_size, n]
    Returns:
        a float32 `Tensor` of shape [n]
    """
    return (gates > 0).sum(0)

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
        torch.tensor([0.0], device=clean_values.device),
        torch.tensor([1.0], device=clean_values.device),
    )

    prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
    prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
    prob = torch.where(is_in, prob_if_in, prob_if_out)
    return prob

def cv_squared(x):
    """The squared coefficient of variation of a sample.
    Useful as a loss to encourage a positive distribution to be more uniform.
    Epsilons added for numerical stability.
    Returns 0 for an empty Tensor.

    Args:
        x: a `Tensor`.
    Returns:
        a `Scalar`.
    """
    eps = 1e-10
    if x.shape[0] == 1:
        return torch.Tensor([0])
    return x.float().var() / (x.float().mean() ** 2 + eps)

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

class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(
                    1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 moe=False, moe_mlp_ratio=-1, moe_experts=64,
                 moe_top_k=2, moe_gate_dim=-1, world_size=1, gate_return_decoupled_activation=False,
                 moe_gate_type="noisy", vmoe_noisy_std=1, gate_task_specific_dim=-1, multi_gate=False, 
                 regu_experts_fromtask = False, num_experts_pertask = -1, num_tasks = -1,
                 gate_input_ahead = False,regu_sem=False,sem_force=False,regu_subimage=False,expert_prune=False):
        super().__init__()
        self.moe = moe
        self.last_moe_analysis = None
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if 
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.gate_input_ahead = gate_input_ahead
        self.expert_prune = expert_prune
        self.expert_hidden_dim = None
        self.dense_hidden_dim = int(dim * mlp_ratio)
        self.active_vs_dense_flops_ratio = None
        if moe:
            self.tot_expert = moe_experts * world_size
            self.moe_top_k = moe_top_k
            activation = nn.Sequential(
                act_layer(),
                nn.Dropout(drop)
            )
            if moe_gate_dim < 0:
                moe_gate_dim = dim
            if moe_mlp_ratio < 0:
                moe_mlp_ratio = mlp_ratio
            moe_hidden_dim = int(dim * moe_mlp_ratio)
            self.expert_hidden_dim = int(moe_hidden_dim)
            self.active_vs_dense_flops_ratio = float(self.moe_top_k * self.expert_hidden_dim) / float(max(self.dense_hidden_dim, 1))

            if moe_gate_type == "noisy":
                moe_gate_fun = NoisyGate
            elif moe_gate_type == "noisy_vmoe":
                moe_gate_fun = NoisyGate_VMoE
            else:
                raise ValueError("unknow gate type of {}".format(moe_gate_type))

            self.mlp = FMoETransformerMLP(num_expert=moe_experts, d_model=dim, d_gate=moe_gate_dim, d_hidden=moe_hidden_dim,
                                          world_size=world_size, top_k=moe_top_k, activation=activation, gate=moe_gate_fun,
                                          gate_return_decoupled_activation=gate_return_decoupled_activation, vmoe_noisy_std=vmoe_noisy_std, 
                                          gate_task_specific_dim=gate_task_specific_dim,multi_gate=multi_gate,
                                          regu_experts_fromtask = regu_experts_fromtask, num_experts_pertask = num_experts_pertask, num_tasks = num_tasks,
                                          regu_sem=regu_sem,sem_force=sem_force,regu_subimage=regu_subimage,expert_prune=self.expert_prune)
            self.mlp_drop = nn.Dropout(drop)
        else:
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
    
    def _ckpt_main_moe(self, x, gate_inp, task_specific_feature, sem, task_id_tensor):
        """Checkpointed MoE forward: attn + norm2 + mlp, returns summaries for cv_loss"""
        # attn + norm2
        x = x + self.drop_path(self.attn(self.norm1(x)))
        normed_x = self.norm2(x)

        task_id = int(task_id_tensor.item())

        moe_output, clean_logits, noisy_logits, noise_stddev, top_logits, gates = \
            self.mlp(normed_x, gate_inp, task_id, task_specific_feature, sem)

        x = x + self.drop_path(self.mlp_drop(moe_output))

        # Compute summaries for cv_loss (gates tensor stays inside checkpoint)
        importance = gates.sum(0)  # [E]

        # Compute load vector
        if self.moe_top_k < self.tot_expert and abs(noise_stddev) > 1e-6:
            load = _prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits, self.moe_top_k).sum(0)
        else:
            load = _gates_to_load(gates)

        probs = gates.float()
        token_entropy = -(probs.clamp_min(1e-12).log() * probs).sum(dim=-1)
        gate_entropy_sum = token_entropy.sum()
        top1_prob_sum = probs.max(dim=-1).values.sum()
        gate_token_count = probs.new_tensor(float(probs.shape[0]))
        expert_load_hist = (probs > 0).sum(dim=0).to(probs.dtype)
        moe_out_norm_ratio = moe_output.float().norm(p=2) / (normed_x.float().norm(p=2) + 1e-12)
        clean_logit_std = clean_logits.float().std(dim=-1, unbiased=False).mean()

        return (
            x,
            importance,
            load,
            gate_entropy_sum,
            top1_prob_sum,
            gate_token_count,
            expert_load_hist,
            moe_out_norm_ratio,
            clean_logit_std,
        )

    def _ckpt_non_moe(self, x):
        """Checkpointed non-MoE forward: attn + norm2 + mlp"""
        x = x + self.drop_path(self.attn(self.norm1(x)))
        normed_x = self.norm2(x)
        x = x + self.drop_path(self.mlp(normed_x))
        return x

    def forward(self, x, gate_inp=None, task_id=None, task_specific_feature=None, sem=None):
        if self.gate_input_ahead:  # False
            gate_inp = x

        if not self.moe:
            # non-moe path: fully checkpointed
            if self.training:
                x = checkpoint(self._ckpt_non_moe, x, use_reentrant=False)
            else:
                x = self._ckpt_non_moe(x)
            self.last_moe_analysis = None
            return x, None

        # MoE path
        task_id_tensor = torch.tensor(task_id, device=x.device) if task_id is not None else torch.tensor(0, device=x.device)

        if self.training:
            (
                x,
                importance,
                load,
                gate_entropy_sum,
                top1_prob_sum,
                gate_token_count,
                expert_load_hist,
                moe_out_norm_ratio,
                clean_logit_std,
            ) = checkpoint(
                self._ckpt_main_moe,
                x, gate_inp, task_specific_feature, sem, task_id_tensor,
                use_reentrant=False
            )
        else:
            (
                x,
                importance,
                load,
                gate_entropy_sum,
                top1_prob_sum,
                gate_token_count,
                expert_load_hist,
                moe_out_norm_ratio,
                clean_logit_std,
            ) = self._ckpt_main_moe(x, gate_inp, task_specific_feature, sem, task_id_tensor)

        # CV loss calculation outside checkpoint
        if self.training:
            cv_loss = cv_squared(importance) + cv_squared(load)
        else:
            cv_loss = 0

        load_f = load.float()
        if load_f.numel() <= 1:
            expert_load_cv = 0.0
        else:
            expert_load_cv = float((load_f.var(unbiased=False) / (load_f.mean().pow(2) + 1e-10)).item())

        self.last_moe_analysis = {
            "gate_entropy_sum": float(gate_entropy_sum.item()),
            "top1_prob_sum": float(top1_prob_sum.item()),
            "gate_token_count": int(gate_token_count.item()),
            "expert_load_hist": [int(v) for v in expert_load_hist.tolist()],
            "expert_load_cv": expert_load_cv,
            "clean_logit_std": float(clean_logit_std.item()),
            "moe_out_norm_ratio": float(moe_out_norm_ratio.item()),
            "expert_hidden_dim": int(self.expert_hidden_dim) if self.expert_hidden_dim is not None else 0,
            "active_vs_dense_flops_ratio": float(self.active_vs_dense_flops_ratio) if self.active_vs_dense_flops_ratio is not None else 0.0,
        }

        return x, cv_loss

class VisionTransformerMoE(nn.Module):
    def __init__(self, model_name='vit_large_patch16_384', img_size=384, patch_size=16, in_chans=3, embed_dim=1024, depth=24,
                    num_heads=16, num_classes=19, mlp_ratio=4., qkv_bias=True, qk_scale=None,  representation_size=None, distilled=False, 
                    drop_rate=0.1, attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None, norm_layer=partial(nn.LayerNorm, eps=1e-6), norm_cfg=None,
                    pos_embed_interp=False, random_init=False, align_corners=False,
                    act_layer=None, weight_init='', moe_mlp_ratio=-1, moe_experts=64, moe_top_k=2, world_size=1, gate_dim=-1,
                    gate_return_decoupled_activation=False, moe_gate_type="noisy", vmoe_noisy_std=1, gate_task_specific_dim=-1,multi_gate=False,
                    regu_experts_fromtask = False, num_experts_pertask = -1, num_tasks = -1, gate_input_ahead=False, regu_sem=False, sem_force=False, regu_subimage=False, 
                    expert_prune=False, **kwargs):
        super(VisionTransformerMoE, self).__init__(**kwargs)
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
        self.moe_mlp_ratio = moe_mlp_ratio
        self.moe_top_k = moe_top_k
        self.gate_return_decoupled_activation = gate_return_decoupled_activation
        self.multi_gate = multi_gate
        self.regu_sem = regu_sem
        self.sem_force = sem_force
        # print(self.hybrid_backbone is None)
        self.expert_prune = expert_prune
        print('set expert prune as ',self.expert_prune)
        if self.hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                self.hybrid_backbone, img_size=self.img_size, in_chans=self.in_chans, embed_dim=self.embed_dim)
        else:
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
        self.num_tasks = gate_dim-embed_dim
        self.gate_task_specific_dim = gate_task_specific_dim
        self.gate_input_ahead = gate_input_ahead
        if self.gate_task_specific_dim<0 or self.multi_gate:
            self.gate_task_represent = None
        else:
            self.gate_task_represent = new_Mlp(in_features=self.num_tasks, hidden_features=int(self.gate_task_specific_dim), out_features=self.gate_task_specific_dim,)
            # self.gamma = nn.Parameter(torch.Tensor([1]), requires_grad=True)
        for i in range(self.depth):
            if i % 2 == 0:
                blocks.append(Block(dim=self.embed_dim, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio, qkv_bias=self.qkv_bias, qk_scale=self.qk_scale,
                drop=self.drop_rate, attn_drop=self.attn_drop_rate, drop_path=dpr[i], norm_layer=self.norm_layer))
            else:
                blocks.append(Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                              moe=True, moe_mlp_ratio=moe_mlp_ratio, moe_experts=moe_experts, moe_top_k=moe_top_k, moe_gate_dim=gate_dim, world_size=world_size,
                              gate_return_decoupled_activation=self.gate_return_decoupled_activation,
                              moe_gate_type=moe_gate_type, vmoe_noisy_std=vmoe_noisy_std, 
                              gate_task_specific_dim=self.gate_task_specific_dim,multi_gate=self.multi_gate,
                              regu_experts_fromtask = regu_experts_fromtask, num_experts_pertask = num_experts_pertask, num_tasks = num_tasks,
                              gate_input_ahead = self.gate_input_ahead,regu_sem=regu_sem,sem_force=sem_force,regu_subimage=regu_subimage,expert_prune=self.expert_prune))
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
        self.latest_moe_stats = None
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

    def _conv_filter(self, state_dict, patch_size=16):
        """ convert patch embedding weight from manual patchify + linear proj to conv"""
        out_dict = {}
        for k, v in state_dict.items():
            if 'patch_embed.proj.weight' in k:
                v = v.reshape((v.shape[0], 3, patch_size, patch_size))
            out_dict[k] = v
        return out_dict

    def to_2D(self, x):
        n, hw, c = x.shape
        h = w = int(math.sqrt(hw))
        x = x.transpose(1, 2).reshape(n, c, h, w)
        return x

    def to_1D(self, x):
        n, c, h, w = x.shape
        x = x.reshape(n, c, -1).transpose(1, 2)
        return x

    def get_groundtruth_sem(self, sem):
        batch = sem.shape[0]
        hint = np.ones((batch,1,int(sem.shape[2]/self.patch_size),int(sem.shape[3]/self.patch_size)))*255
        idx = 0
        for k in range(batch):
            for i in range(int(sem.shape[2]/self.patch_size)):
                for j in range(int(sem.shape[3]/self.patch_size)):
                    patch = sem[k][:,self.patch_size*i:self.patch_size*(i+1),self.patch_size*j:self.patch_size*(j+1)].cpu().numpy().flatten()
                    index , num=Counter(patch).most_common(1)[0]
                    if num>0.4*(self.patch_size*self.patch_size):
                        hint[k,:,i,j]=index
                        if index != 255:
                            idx = idx+1
        filename = 'gt_patch_{}.npy'.format(self.idx)
        self.idx=self.idx+1
        # np.save(filename, hint)
        return torch.tensor(hint, device=sem.device) 

    def forward_features(self, x, gate_inp, task_id,sem):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        task_specific_feature = None
        if (task_id is not None) and (self.gate_task_represent is not None):
            task_specific = torch.zeros(self.num_tasks,device=x.device)
            task_specific[task_id]=1.0
            task_specific_feature = self.gate_task_represent(task_specific)
        out = None
        total_cv_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype, requires_grad=True)
        stats = {"moe_blocks": 0, "total_positions": 0}
        analysis = {
            "gate_entropy_sum": 0.0,
            "top1_prob_sum": 0.0,
            "gate_token_count": 0,
            "expert_load_cv_sum": 0.0,
            "clean_logit_std_sum": 0.0,
            "moe_out_norm_ratio_sum": 0.0,
            "expert_hidden_dim_sum": 0.0,
            "active_vs_dense_flops_ratio_sum": 0.0,
            "analysis_block_count": 0,
            "expert_load_hist": None,
        }

        for i, blk in enumerate(self.blocks):
            if blk.moe:
                x, cv_loss = blk(x, gate_inp, task_id, task_specific_feature, sem=sem)
                if cv_loss is not None:
                    total_cv_loss = total_cv_loss + cv_loss
                stats["moe_blocks"] += 1
                stats["total_positions"] += int(B * max(x.shape[1] - 1, 0))

                block_analysis = getattr(blk, "last_moe_analysis", None)
                if isinstance(block_analysis, dict):
                    analysis["gate_entropy_sum"] += float(block_analysis.get("gate_entropy_sum", 0.0))
                    analysis["top1_prob_sum"] += float(block_analysis.get("top1_prob_sum", 0.0))
                    analysis["gate_token_count"] += int(block_analysis.get("gate_token_count", 0))
                    analysis["expert_load_cv_sum"] += float(block_analysis.get("expert_load_cv", 0.0))
                    analysis["clean_logit_std_sum"] += float(block_analysis.get("clean_logit_std", 0.0))
                    analysis["moe_out_norm_ratio_sum"] += float(block_analysis.get("moe_out_norm_ratio", 0.0))
                    analysis["expert_hidden_dim_sum"] += float(block_analysis.get("expert_hidden_dim", 0.0))
                    analysis["active_vs_dense_flops_ratio_sum"] += float(block_analysis.get("active_vs_dense_flops_ratio", 0.0))
                    analysis["analysis_block_count"] += 1

                    block_hist = block_analysis.get("expert_load_hist", None)
                    if block_hist is not None:
                        if analysis["expert_load_hist"] is None:
                            analysis["expert_load_hist"] = [0] * len(block_hist)
                        if len(analysis["expert_load_hist"]) == len(block_hist):
                            for j in range(len(block_hist)):
                                analysis["expert_load_hist"][j] += int(block_hist[j])
            else:
                x, _ = blk(x)

            if i in self.out_indices:
                out = x

        gate_token_count = analysis["gate_token_count"]
        if gate_token_count > 0:
            gate_entropy = analysis["gate_entropy_sum"] / float(gate_token_count)
            top1_prob_mean = analysis["top1_prob_sum"] / float(gate_token_count)
        else:
            gate_entropy = 0.0
            top1_prob_mean = 0.0

        block_count = max(analysis["analysis_block_count"], 1)
        expert_load_hist = analysis["expert_load_hist"] if analysis["expert_load_hist"] is not None else []
        if len(expert_load_hist) > 0:
            dead_expert_ratio = float(sum(1 for v in expert_load_hist if v == 0)) / float(len(expert_load_hist))
        else:
            dead_expert_ratio = 0.0

        stats["analysis"] = {
            "gate_entropy": gate_entropy,
            "top1_prob_mean": top1_prob_mean,
            "expert_load_hist": expert_load_hist,
            "dead_expert_ratio": dead_expert_ratio,
            "expert_load_cv": analysis["expert_load_cv_sum"] / float(block_count),
            "clean_logit_std": analysis["clean_logit_std_sum"] / float(block_count),
            "moe_out_norm_ratio": analysis["moe_out_norm_ratio_sum"] / float(block_count),
            "expert_hidden_dim": analysis["expert_hidden_dim_sum"] / float(block_count),
            "active_vs_dense_flops_ratio": analysis["active_vs_dense_flops_ratio_sum"] / float(block_count),
        }

        self.latest_moe_stats = stats
        if self.training and stats["moe_blocks"] > 0 and self.wandb_logger is not None:
            logger = self.wandb_logger()
            if logger is not None:
                logger.log_moe_stats(stats)

        return out, total_cv_loss

    def forward(self, x, gate_inp=None, task_id=None,sem=None):
        if sem is not None and (self.regu_sem or self.sem_force):
            sem = self.get_groundtruth_sem(sem)
        out, cv_losses = self.forward_features(x, gate_inp, task_id=task_id, sem=sem)
        return out, cv_losses


def _init_vit_weights(m, n: str = '', head_bias: float = 0., jax_impl: bool = False):
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
