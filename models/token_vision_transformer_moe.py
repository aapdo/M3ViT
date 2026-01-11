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

a=[[0],[1,17,18,19,20],[2,12,13,14,15,16],[3,9,10,11],[4,5],[6,7,8,38],[21,22,23,24,25,26,39],[27,28,29,30,31,32,33,34,35,36,37]]


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
                 moe_gate_type="token_noisy_vmoe", vmoe_noisy_std=1, gate_task_specific_dim=-1, multi_gate=False,
                 num_experts_pertask = -1, num_tasks = -1,
                 gate_input_ahead = False):
        super().__init__()
        self.moe = moe
        self.moe_top_k = moe_top_k
        self.tot_expert = moe_experts
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.gate_input_ahead = gate_input_ahead

        if moe:
            activation = nn.Sequential(
                act_layer(),
                nn.Dropout(drop)
            )
            if moe_gate_dim < 0:
                moe_gate_dim = dim
            if moe_mlp_ratio < 0:
                moe_mlp_ratio = mlp_ratio
            moe_hidden_dim = int(dim * moe_mlp_ratio)

            if moe_gate_type == "noisy":
                moe_gate_fun = NoisyGate
            elif moe_gate_type == "noisy_vmoe":
                moe_gate_fun = NoisyGate_VMoE
            elif moe_gate_type == "token_noisy_vmoe":
                moe_gate_fun = TokenNoisyGate_VMoE
            else:
                raise ValueError("unknow gate type of {}".format(moe_gate_type))

            self.mlp = TokenFMoETransformerMLP(num_expert=moe_experts, d_model=dim, d_gate=moe_gate_dim, d_hidden=moe_hidden_dim,
                                          world_size=world_size, top_k=moe_top_k, activation=activation, gate=moe_gate_fun,
                                          vmoe_noisy_std=vmoe_noisy_std, gate_task_specific_dim=gate_task_specific_dim,multi_gate=multi_gate,
                                          num_experts_pertask = num_experts_pertask, num_tasks = num_tasks
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

    def _ckpt_moe(self, x, gate_inp, task_specific_feature, selector_output, task_id_tensor):
        """Checkpointed MoE forward: attn + norm2 + mlp, returns (x, importance, load)"""
        # attn + norm2
        x = x + self.drop_path(self.attn(self.norm1(x)))
        normed_x = self.norm2(x)

        task_id = int(task_id_tensor.item())

        moe_output, clean_logits, noisy_logits, noise_stddev, top_logits, gates = \
            self.mlp(normed_x, gate_inp, task_id, task_specific_feature, selector_output)

        x = x + self.drop_path(self.mlp_drop(moe_output))

        # Compute summaries (importance, load) - no cv_loss here
        importance = gates.sum(0)  # [E]

        # Compute load - check noise_stddev properly
        noise_ok = (noise_stddev is not None) and (noise_stddev.mean().item() > 1e-6)

        if (self.moe_top_k < self.tot_expert) and noise_ok:
            load = self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits, self.moe_top_k).sum(0)
        else:
            load = self._gates_to_load(gates)

        return x, importance, load

    def _ckpt_non_moe(self, x):
        """Checkpointed non-MoE forward: attn + norm2 + mlp"""
        x = x + self.drop_path(self.attn(self.norm1(x)))
        normed_x = self.norm2(x)
        x = x + self.drop_path(self.mlp(normed_x))
        return x

    def forward(self, x, gate_inp=None, task_id=None, task_specific_feature=None, selector_output=None):
        if self.gate_input_ahead:
            gate_inp = x

        if not self.moe:
            # non-moe path: fully checkpointed
            if self.training:
                x = checkpoint(self._ckpt_non_moe, x, use_reentrant=False)
            else:
                x = self._ckpt_non_moe(x)
            return x, None, None

        # MoE path
        task_id_tensor = torch.tensor(
            0 if task_id is None else int(task_id),
            device=x.device,
            dtype=torch.int64
        )

        if self.training:
            x, importance, load = checkpoint(
                self._ckpt_moe,
                x, gate_inp, task_specific_feature, selector_output, task_id_tensor,
                use_reentrant=False
            )
        else:
            x, importance, load = self._ckpt_moe(
                x, gate_inp, task_specific_feature, selector_output, task_id_tensor
            )

        return x, importance, load

class TokenVisionTransformerMoE(nn.Module):
    def __init__(self, model_name='vit_large_patch16_384', img_size=384, patch_size=16, in_chans=3, embed_dim=1024, depth=24,
                    num_heads=16, num_classes=19, mlp_ratio=4., qkv_bias=True, qk_scale=None,  representation_size=None, distilled=False, 
                    drop_rate=0.1, attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None, norm_layer=partial(nn.LayerNorm, eps=1e-6), norm_cfg=None,
                    pos_embed_interp=False, random_init=False, align_corners=False,
                    act_layer=None, weight_init='', moe_mlp_ratio=-1, moe_experts=8, moe_top_k=4, world_size=2, gate_dim=-1,
                    moe_gate_type="token_noisy_vmoe", vmoe_noisy_std=1, gate_task_specific_dim=-1,multi_gate=False,
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
                              moe_gate_type=moe_gate_type, vmoe_noisy_std=vmoe_noisy_std, 
                              gate_task_specific_dim=self.gate_task_specific_dim,multi_gate=self.multi_gate,
                              num_experts_pertask = num_experts_pertask, num_tasks = num_tasks,
                              gate_input_ahead = self.gate_input_ahead))
        self.blocks = nn.Sequential(*blocks)

        # Initialize aggregation module
        self.aggregation = Aggregation()

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
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # x.shape => [Batch, Number of Token, Dimension]
        x = x + self.pos_embed
        x = self.pos_drop(x)

        task_specific_feature = None

        # TODO: support task conditioned router embedding
        # if (task_id is not None) and (self.gate_task_represent is not None):
        #     task_specific = torch.zeros(self.num_tasks,device=x.device)
        #     task_specific[task_id]=1.0
        #     task_specific_feature = self.gate_task_represent(task_specific)

        # Initialize dict to store outputs for each task
        outs = {task: x for task in range(self.num_tasks)}
        prev_router_outputs = []
        is_first_moe = True

        # Initialize total cv_loss
        total_cv_loss = x.new_tensor(0.0)

        # Initialize statistics counters
        stats = {
            'total_tokens': 0,
            'reusable_tokens': 0,
            'aggregated_tokens': 0,
            'shared_gate_tokens': 0,
            'moe_blocks': 0
        }

        for i, blk in enumerate(self.blocks):
            if blk.moe:
                stats['moe_blocks'] += 1

                # Compute current router outputs for all tasks
                current_router_outputs = []
                for task in range(self.num_tasks):
                    current_router_outputs.append(blk.mlp.router_selector(outs[task]))

                if is_first_moe:
                    # First MoE block: force all tokens to use task-specific gate (no shared_gate)
                    # Set selector_output to 0 to force task-specific gate usage
                    for task in range(self.num_tasks):
                        forced_selector = torch.zeros_like(current_router_outputs[task])
                        outs[task], imp, load = blk(
                            outs[task],
                            gate_inp=gate_inp,
                            task_id=task,
                            task_specific_feature=task_specific_feature,
                            selector_output=forced_selector
                        )
                        # Accumulate cv_loss
                        if (imp is not None) and (load is not None):
                            total_cv_loss = total_cv_loss + blk.cv_squared(imp) + blk.cv_squared(load)
                    is_first_moe = False
                else:
                    # Not first MoE block: analyze aggregation opportunities
                    # For each token position [B, N], classify into three categories:
                    # 1. prev & curr both True (>=2 tasks): reusable - compute once and copy
                    # 2. any task has curr False: not reusable - each task computes independently
                    # 3. prev False, curr True (>=2 tasks): compute independently, then aggregate results

                    curr_shared_masks = []  # [T, B, N]: curr > 0.5
                    prev_shared_masks = []  # [T, B, N]: prev > 0.5
                    reusable_masks = []     # [T, B, N]: prev & curr both True

                    for task in range(self.num_tasks):
                        curr_mask = (current_router_outputs[task] > 0.5)  # [B, N]
                        prev_mask = (prev_router_outputs[task] > 0.5)     # [B, N]
                        reusable_mask = curr_mask & prev_mask              # [B, N]

                        curr_shared_masks.append(curr_mask)
                        prev_shared_masks.append(prev_mask)
                        reusable_masks.append(reusable_mask)

                    # Stack masks: [num_tasks, B, N]
                    curr_shared_stacked = torch.stack(curr_shared_masks, dim=0)      # [T, B, N]
                    reusable_stacked = torch.stack(reusable_masks, dim=0)            # [T, B, N]

                    # Count how many tasks have curr_shared for each position
                    curr_shared_count = curr_shared_stacked.sum(dim=0)  # [B, N]

                    # Count how many tasks are reusable (prev & curr both True) for each position
                    reusable_count = reusable_stacked.sum(dim=0)  # [B, N]

                    # Collect statistics
                    total_positions = B * (x.shape[1])  # B * N (number of tokens)
                    stats['total_tokens'] += total_positions

                    # Shared gate usage: positions where at least one task uses shared_gate
                    shared_positions = (curr_shared_count > 0).sum().item()
                    stats['shared_gate_tokens'] += shared_positions

                    # Reusable positions: where at least 2 tasks have prev & curr both True
                    reusable_positions = (reusable_count >= 2).sum().item()
                    stats['reusable_tokens'] += reusable_positions

                    # MoE block forward for each task
                    task_outputs = {}
                    for task in range(self.num_tasks):
                        outs[task], imp, load = blk(
                            outs[task],
                            gate_inp=gate_inp,
                            task_id=task,
                            task_specific_feature=task_specific_feature,
                            selector_output=current_router_outputs[task]
                        )
                        task_outputs[task] = outs[task]

                        # Accumulate cv_loss
                        if (imp is not None) and (load is not None):
                            total_cv_loss = total_cv_loss + blk.cv_squared(imp) + blk.cv_squared(load)

                    # Reuse results for skipped positions (prev & curr both True)
                    # Note: Reuse logic is removed to avoid checkpoint issues

                    # Aggregate results for positions where prev: False, curr: True (>=2 tasks)
                    # These positions need aggregation_function to combine results from multiple tasks
                    aggregation_needed_mask = (reusable_count < curr_shared_count) & (curr_shared_count >= 2)

                    # Collect aggregation statistics
                    aggregated_positions = aggregation_needed_mask.sum().item()
                    stats['aggregated_tokens'] += aggregated_positions

                    if aggregation_needed_mask.any():
                        # Apply aggregation for positions where multiple tasks use shared_gate in curr
                        # but not all used it in prev
                        aggregated = self.aggregation(
                            task_outputs,
                            curr_shared_masks,
                            aggregation_needed_mask
                        )

                        if aggregated is not None:
                            # Update all tasks' outputs at aggregation positions using torch.where
                            for task in range(self.num_tasks):
                                update_mask = aggregation_needed_mask & curr_shared_masks[task]
                                if update_mask.any():
                                    m = update_mask.unsqueeze(-1)
                                    outs[task] = torch.where(m, aggregated, outs[task])

                # Update prev_router_outputs for next iteration
                prev_router_outputs = [curr.clone() for curr in current_router_outputs]

            else:
                # Non-MoE block: each task processes independently
                for task in range(self.num_tasks):
                    outs[task], _, _ = blk(outs[task])

        # Calculate statistics ratios
        if stats['total_tokens'] > 0:
            stats['reuse_ratio'] = stats['reusable_tokens'] / stats['total_tokens']
            stats['aggregation_ratio'] = stats['aggregated_tokens'] / stats['total_tokens']
            stats['shared_gate_ratio'] = stats['shared_gate_tokens'] / stats['total_tokens']
        else:
            stats['reuse_ratio'] = 0.0
            stats['aggregation_ratio'] = 0.0
            stats['shared_gate_ratio'] = 0.0

        # Log MoE stats to wandb if available (only during training)
        if self.training and stats['total_tokens'] > 0 and self.wandb_logger is not None:
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