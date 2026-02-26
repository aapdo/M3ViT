# This code is referenced from 
# https://github.com/facebookresearch/astmt/
# 
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# License: Attribution-NonCommercial 4.0 International
from collections import OrderedDict
from xml.parsers.expat import model
import torch
import cv2
import numpy as np


import torch.nn.functional as F
import math
import logging
import warnings
import errno
import os
import sys
import re
import zipfile
from urllib.parse import urlparse  # noqa: F401
import torch.distributed as dist
def tens2image(tens):
    """Converts tensor with 2 or 3 dimensions to numpy array"""
    im = tens.numpy()

    if im.shape[0] == 1:
        im = np.squeeze(im, axis=0)

    if im.ndim == 3:
        im = im.transpose((1, 2, 0))

    return im


def pascal_color_map(N=256, normalized=False):
    """
    Python implementation of the color map function for the PASCAL VOC data set.
    Official Matlab version can be found in the PASCAL VOC devkit
    http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
    """

    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap / 255 if normalized else cmap
    return cmap


def fixed_resize(sample, resolution, flagval=None):
    """
    Fixed resize to
    resolution (tuple): resize image to size specified by tuple eg. (512, 512).
    resolution (int): bring smaller side to resolution eg. image of shape 321 x 481 -> 512 x 767
    """
    if flagval is None:
        if ((sample == 0) | (sample == 1)).all():
            flagval = cv2.INTER_NEAREST
        else:
            flagval = cv2.INTER_CUBIC

    if isinstance(resolution, int):
        tmp = [resolution, resolution]
        tmp[int(np.argmax(sample.shape[:2]))] = int(
            round(float(resolution) / np.min(sample.shape[:2]) * np.max(sample.shape[:2])))
        resolution = tuple(tmp)

    if sample.ndim == 2 or (sample.ndim == 3 and sample.shape[2] == 3):
        sample = cv2.resize(sample, resolution[::-1], interpolation=flagval)
    else:
        tmp = sample
        sample = np.zeros(np.append(resolution, tmp.shape[2]), dtype=np.float32)
        for ii in range(sample.shape[2]):
            sample[:, :, ii] = cv2.resize(tmp[:, :, ii], resolution[::-1], interpolation=flagval)
    return sample


def im_normalize(im, max_value=1):
    """
    Normalize image to range 0 - max_value
    """
    imn = max_value * (im - im.min()) / max((im.max() - im.min()), 1e-8)
    return imn


def generate_param_report(logfile, param):
    log_file = open(logfile, 'w')
    for key, val in param.items():
        log_file.write(key + ':' + str(val) + '\n')
    log_file.close()


def ind2sub(array_shape, inds):
    rows, cols = [], []
    for k in range(len(inds)):
        if inds[k] == 0:
            continue
        cols.append((inds[k].astype('int') // array_shape[1]))
        rows.append((inds[k].astype('int') % array_shape[1]))
    return rows, cols


HASH_REGEX = re.compile(r'-([a-f0-9]*)\.')
_logger = logging.getLogger(__name__)
_UPCYCLING_RUNTIME_OPTIONS = {}


def set_upcycling_runtime_options(options=None):
    """Inject runtime options (from train args) for upcycling in load_pretrained."""
    global _UPCYCLING_RUNTIME_OPTIONS
    if options is None:
        _UPCYCLING_RUNTIME_OPTIONS = {}
    else:
        _UPCYCLING_RUNTIME_OPTIONS = dict(options)


def _as_cfg_dict(cfg):
    if cfg is None:
        return None
    if isinstance(cfg, dict):
        return dict(cfg)
    try:
        return dict(cfg)
    except Exception:
        out = {}
        for k in dir(cfg):
            if k.startswith('_'):
                continue
            v = getattr(cfg, k)
            if not callable(v):
                out[k] = v
        return out


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _merge_cfg_with_runtime_options(cfg):
    merged = _as_cfg_dict(cfg)
    if merged is None:
        return None
    if not _UPCYCLING_RUNTIME_OPTIONS:
        return merged
    for k, v in _UPCYCLING_RUNTIME_OPTIONS.items():
        if v is not None:
            merged[k] = v
    return merged


def _resolve_deit_init_mode_from_cfg(cfg):
    mode = str(_cfg_get(cfg, "deit_init_mode", "deit_upcycling") or "deit_upcycling")
    mode = mode.strip().lower()
    valid_modes = {"scratch", "deit_warm_start", "deit_upcycling"}
    if mode not in valid_modes:
        raise ValueError(
            f"Unsupported deit_init_mode '{mode}'. Expected one of {sorted(valid_modes)}"
        )
    return mode


def load_state_dict_from_url(url, model_dir=None, file_name=None, check_hash=False, progress=True, map_location=None):
    # Issue warning to move data if old env is set
    if os.getenv('TORCH_MODEL_ZOO'):
        warnings.warn(
            'TORCH_MODEL_ZOO is deprecated, please use env TORCH_HOME instead')

    if model_dir is None:
        hub_dir = torch.hub.get_dir()
        model_dir = os.path.join(hub_dir, 'checkpoints')
        try:
            os.makedirs(model_dir)
        except OSError as e:
            if e.errno == errno.EEXIST:
                # Directory already exists, ignore.
                pass
            else:
                # Unexpected OSError, re-raise.
                raise
    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        hash_prefix = HASH_REGEX.search(
            filename).group(1) if check_hash else None
        torch.hub.download_url_to_file(
            url, cached_file, hash_prefix, progress=progress)
    if zipfile.is_zipfile(cached_file):
        state_dict = torch.load(
            cached_file, map_location=map_location)['model']
    else:
        state_dict = torch.load(cached_file, map_location=map_location)
    return state_dict


def _infer_num_prefix_tokens(pos_embed):
    total_tokens = int(pos_embed.shape[1])
    for num_prefix_tokens in (2, 1, 0):
        num_patch_tokens = total_tokens - num_prefix_tokens
        if num_patch_tokens <= 0:
            continue
        side = int(math.sqrt(num_patch_tokens))
        if side * side == num_patch_tokens:
            return num_prefix_tokens
    return 1


def _target_num_prefix_tokens(model, num_patches):
    pos_embed = getattr(model, "pos_embed", None)
    if isinstance(pos_embed, torch.Tensor):
        target = int(pos_embed.shape[1]) - int(num_patches)
        if target >= 0:
            return target
    num_token = getattr(model, "num_token", None)
    if num_token is not None:
        return int(num_token)
    return 1


def _adapt_prefix_tokens(prefix_tokens, target_prefix_tokens):
    _, source_prefix_tokens, channels = prefix_tokens.shape
    if source_prefix_tokens == target_prefix_tokens:
        return prefix_tokens
    if source_prefix_tokens > target_prefix_tokens:
        return prefix_tokens[:, :target_prefix_tokens]

    if source_prefix_tokens > 0:
        cls_token = prefix_tokens[:, :1]
    else:
        cls_token = prefix_tokens.new_zeros((prefix_tokens.shape[0], 1, channels))
    extra = cls_token.expand(-1, target_prefix_tokens - source_prefix_tokens, -1)
    return torch.cat((prefix_tokens, extra), dim=1)


def _align_pos_embed_prefix_tokens(pos_embed, target_prefix_tokens):
    source_prefix_tokens = _infer_num_prefix_tokens(pos_embed)
    prefix_tokens = pos_embed[:, :source_prefix_tokens]
    patch_tokens = pos_embed[:, source_prefix_tokens:]
    prefix_tokens = _adapt_prefix_tokens(prefix_tokens, target_prefix_tokens)
    return torch.cat((prefix_tokens, patch_tokens), dim=1)

def load_pretrained_pos_emb(model, cfg=None, num_classes=1000, in_chans=3, filter_fn=None, strict=True, pos_embed_interp=False, num_patches=576, align_corners=False, img_h=None, img_w=None):
    if cfg is None:
        cfg = getattr(model, 'default_cfg')
    cfg = _merge_cfg_with_runtime_options(cfg)
    deit_init_mode = _resolve_deit_init_mode_from_cfg(cfg)
    if deit_init_mode == "scratch":
        print("[INIT] skip DeiT pos-embed load (deit_init_mode=scratch)")
        return
    if cfg is None or 'url' not in cfg or not cfg['url']:
        _logger.warning(
            "Pretrained model URL is invalid, using random initialization.")
        return

    if 'pretrained_finetune' in cfg and cfg['pretrained_finetune']:
        state_dict = torch.load(cfg['pretrained_finetune'])
        print('load pre-trained weight from ' + cfg['pretrained_finetune'])
    else:
        state_dict = load_state_dict_from_url(
            cfg['url'], progress=False, map_location='cpu')
        # print('load pre-trained weight from imagenet21k')

    # if filter_fn is not None:
    #     state_dict = filter_fn(state_dict)
    pos_emb_state_dict = OrderedDict()
    for key, item in state_dict.items():
        if "pos_embed" in key:
            pos_emb_state_dict[key] = state_dict[key]
    if "pos_embed" not in pos_emb_state_dict:
        return

    target_prefix_tokens = _target_num_prefix_tokens(model, num_patches)
    
    if pos_embed_interp:
        # print('loaded pos_embed shape',pos_emb_state_dict['pos_embed'].shape)
        source_pos_embed = pos_emb_state_dict['pos_embed']
        source_prefix_tokens = _infer_num_prefix_tokens(source_pos_embed)
        n, c, hw = source_pos_embed.transpose(1, 2).shape
        h = w = int(math.sqrt(hw))
        pos_embed_weight = source_pos_embed[:, (-h * w):]
        pos_embed_weight = pos_embed_weight.transpose(1, 2)
        n, c, hw = pos_embed_weight.shape
        h = w = int(math.sqrt(hw))
        pos_embed_weight = pos_embed_weight.view(n, c, h, w)
        # print(pos_embed_weight.shape)
        if img_h is None:
            pos_embed_weight = F.interpolate(pos_embed_weight, size=int(
                math.sqrt(num_patches)), mode='bilinear', align_corners=align_corners)
        else:
            pos_embed_weight = F.interpolate(pos_embed_weight, size=(img_h, img_w), mode='bilinear', align_corners=align_corners)
        # print('after interpolation', pos_embed_weight.shape)
        pos_embed_weight = pos_embed_weight.view(n, c, -1).transpose(1, 2)

        prefix_tokens = source_pos_embed[:, :source_prefix_tokens]
        prefix_tokens = _adapt_prefix_tokens(prefix_tokens, target_prefix_tokens)
        pos_emb_state_dict['pos_embed'] = torch.cat(
            (prefix_tokens, pos_embed_weight), dim=1)
    else:
        pos_emb_state_dict["pos_embed"] = _align_pos_embed_prefix_tokens(
            pos_emb_state_dict["pos_embed"], target_prefix_tokens
        )
    strict = False
    msg = model.load_state_dict(pos_emb_state_dict, strict=strict)
    print('=========pos emb is loaded from ================',cfg['url'])

def load_pretrained(model, cfg=None, num_classes=1000, in_chans=3, filter_fn=None, strict=True, pos_embed_interp=False, num_patches=576, align_corners=False, img_h=None, img_w=None):
    if cfg is None:
        cfg = getattr(model, 'default_cfg')
    cfg = _merge_cfg_with_runtime_options(cfg)
    deit_init_mode = _resolve_deit_init_mode_from_cfg(cfg)
    if deit_init_mode == "scratch":
        print("[INIT] skip DeiT checkpoint load (deit_init_mode=scratch)")
        return
    if cfg is None or 'url' not in cfg or not cfg['url']:
        _logger.warning(
            "Pretrained model URL is invalid, using random initialization.")
        return

    if 'pretrained_finetune' in cfg and cfg['pretrained_finetune']:
        state_dict = torch.load(cfg['pretrained_finetune'])
        print('load pre-trained weight from ' + cfg['pretrained_finetune'])
    else:
        state_dict = load_state_dict_from_url(
            cfg['url'], progress=False, map_location='cpu')
        print('load pre-trained weight from imagenet21k')

    if filter_fn is not None:
        state_dict = filter_fn(state_dict)

    if in_chans == 1:
        conv1_name = cfg['first_conv']
        _logger.info(
            'Converting first conv (%s) pretrained weights from 3 to 1 channel' % conv1_name)
        conv1_weight = state_dict[conv1_name + '.weight']
        # Some weights are in torch.half, ensure it's float for sum on CPU
        conv1_type = conv1_weight.dtype
        conv1_weight = conv1_weight.float()
        O, I, J, K = conv1_weight.shape
        if I > 3:
            assert conv1_weight.shape[1] % 3 == 0
            # For models with space2depth stems
            conv1_weight = conv1_weight.reshape(O, I // 3, 3, J, K)
            conv1_weight = conv1_weight.sum(dim=2, keepdim=False)
        else:
            conv1_weight = conv1_weight.sum(dim=1, keepdim=True)
        conv1_weight = conv1_weight.to(conv1_type)
        state_dict[conv1_name + '.weight'] = conv1_weight
    elif in_chans != 3:
        conv1_name = cfg['first_conv']
        conv1_weight = state_dict[conv1_name + '.weight']
        conv1_type = conv1_weight.dtype
        conv1_weight = conv1_weight.float()
        O, I, J, K = conv1_weight.shape
        if I == 3:
            _logger.warning(
                'Deleting first conv (%s) from pretrained weights.' % conv1_name)
            del state_dict[conv1_name + '.weight']
            strict = False
        else:
            # NOTE this strategy should be better than random init, but there could be other combinations of
            # the original RGB input layer weights that'd work better for specific cases.
            _logger.info(
                'Repeating first conv (%s) weights in channel dim.' % conv1_name)
            repeat = int(math.ceil(in_chans / 3))
            conv1_weight = conv1_weight.repeat(1, repeat, 1, 1)[
                :, :in_chans, :, :]
            conv1_weight *= (3 / float(in_chans))
            conv1_weight = conv1_weight.to(conv1_type)
            state_dict[conv1_name + '.weight'] = conv1_weight

    classifier_name = cfg['classifier']
    if num_classes == 1000 and cfg['num_classes'] == 1001:
        # special case for imagenet trained models with extra background class in pretrained weights
        classifier_weight = state_dict[classifier_name + '.weight']
        state_dict[classifier_name + '.weight'] = classifier_weight[1:]
        classifier_bias = state_dict[classifier_name + '.bias']
        state_dict[classifier_name + '.bias'] = classifier_bias[1:]
    elif num_classes != cfg['num_classes']:
        # completely discard fully connected for all other differences between pretrained and created model
        del state_dict[classifier_name + '.weight']
        del state_dict[classifier_name + '.bias']
        strict = False

    target_prefix_tokens = _target_num_prefix_tokens(model, num_patches)
    if pos_embed_interp:
        print('loaded pos_embed shape',state_dict['pos_embed'].shape)
        source_pos_embed = state_dict["pos_embed"]
        source_prefix_tokens = _infer_num_prefix_tokens(source_pos_embed)
        # print(f"[Pretrained] before interp vit_state_dict['pos_embed'] (flattened): {state_dict['pos_embed'].flatten().tolist()}")
        n, c, hw = source_pos_embed.transpose(1, 2).shape
        h = w = int(math.sqrt(hw))
        pos_embed_weight = source_pos_embed[:, (-h * w):]
        pos_embed_weight = pos_embed_weight.transpose(1, 2)
        n, c, hw = pos_embed_weight.shape
        h = w = int(math.sqrt(hw))
        pos_embed_weight = pos_embed_weight.view(n, c, h, w)
        print(pos_embed_weight.shape)
        if img_h is None:
            pos_embed_weight = F.interpolate(pos_embed_weight, size=int(
                math.sqrt(num_patches)), mode='bilinear', align_corners=align_corners)
        else:
            pos_embed_weight = F.interpolate(pos_embed_weight, size=(img_h, img_w), mode='bilinear', align_corners=align_corners)
        print('after interpolation', pos_embed_weight.shape)
        pos_embed_weight = pos_embed_weight.view(n, c, -1).transpose(1, 2)

        prefix_tokens = source_pos_embed[:, :source_prefix_tokens]
        prefix_tokens = _adapt_prefix_tokens(prefix_tokens, target_prefix_tokens)
        print('prefix_tokens', prefix_tokens.shape)
        state_dict['pos_embed'] = torch.cat(
            (prefix_tokens, pos_embed_weight), dim=1)
        # print(f"[Pretrained] after interp vit_state_dict['pos_embed'] (flattened): {state_dict['pos_embed'].flatten().tolist()}")
    elif "pos_embed" in state_dict:
        state_dict["pos_embed"] = _align_pos_embed_prefix_tokens(
            state_dict["pos_embed"], target_prefix_tokens
        )

    if deit_init_mode == "deit_upcycling":
        state_dict = _inject_moe_expert_from_deit_mlp(state_dict, model, cfg)

        if cfg.get('use_virtual_group_initialization', False):
            state_dict = _inject_virtual_group_init_for_gates(
                state_dict, model,
                cfg=cfg,
                init="normal", std=0.02
            )
    else:
        print(f"[INJECT] skip MoE upcycling (deit_init_mode={deit_init_mode})")

    msg = model.load_state_dict(state_dict, strict=strict)
    print('============load model weights from============',cfg['url'], msg)

def get_dist_info():
    if torch.__version__ < "1.0":
        initialized = dist._initialized
    else:
        initialized = dist.is_initialized()
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size

import torch

def _inject_moe_expert_from_deit_mlp(
    state_dict,
    model,
    cfg,
    *,
    verbose=True,
    verify_shape=True,
    sample_blocks=(1,),
):
    """
    DeiT dense MLP(fc1/fc2) -> MoE experts(FMoELinear) 초기화 (EP 환경: local experts 기준)

    - moe_mlp_ratio=4 (기본): fc1/fc2를 local_experts 개수만큼 그대로 복사.
    - moe_mlp_ratio=1:
        * DeiT MLP를 granularity(G)로 분할해 '그룹 템플릿' 생성
        * local_experts가 G의 배수라면 그룹 반복으로 채움
        * local_experts가 G보다 작으면 앞에서부터 잘라서 사용
      (GELU + softmax-then-topk 전제하에 공식 scaling 적용)
    """
    model_sd = model.state_dict()
    moe_mlp_ratio = float(getattr(model, "moe_mlp_ratio", 4.0))
    # moe_mlp_ratio=-1 means "use mlp_ratio" (same as dense MLP)
    if moe_mlp_ratio < 0:
        moe_mlp_ratio = float(getattr(model, "mlp_ratio", 4.0))

    default_topk = int(_cfg_get(cfg, "moe_top_k", getattr(model, "moe_top_k", 4)))
    use_weight_scaling = bool(_cfg_get(cfg, "use_weight_scaling", False))
    moe_router_pre_softmax = True  # 너는 softmax-then-topk 사용

    if verbose:
        print(f"[INJECT] === _inject_moe_expert_from_deit_mlp START ===")
        print(
            f"[INJECT] moe_mlp_ratio={moe_mlp_ratio}, default_topk={default_topk}, "
            f"use_weight_scaling={use_weight_scaling}, moe_router_pre_softmax={moe_router_pre_softmax}"
        )

    for i, blk in enumerate(model.blocks):
        if not getattr(blk, "moe", False):
            if verbose:
                print(f"[INJECT][SKIP] block {i}: not a MoE block")
            continue

        # DeiT MLP key
        k_fc1_w = f"blocks.{i}.mlp.fc1.weight"
        k_fc1_b = f"blocks.{i}.mlp.fc1.bias"
        k_fc2_w = f"blocks.{i}.mlp.fc2.weight"
        k_fc2_b = f"blocks.{i}.mlp.fc2.bias"

        if k_fc1_w not in state_dict or k_fc2_w not in state_dict:
            if verbose:
                missing = [k for k in [k_fc1_w, k_fc1_b, k_fc2_w, k_fc2_b] if k not in state_dict]
                print(f"[INJECT][SKIP] block {i}: DeiT MLP keys missing: {missing}")
            continue

        if verbose:
            print(f"[INJECT] block {i}: MoE block found, E_local will be determined next")

        fc1_w = state_dict[k_fc1_w]  # [hidden, embed]
        fc1_b = state_dict[k_fc1_b]  # [hidden]
        fc2_w = state_dict[k_fc2_w]  # [embed, hidden]
        fc2_b = state_dict[k_fc2_b]  # [embed]

        # local expert count (EP 환경에서 이 값이 8로 나오는 게 정상)
        E_local = int(getattr(blk.mlp, "num_expert", getattr(model, "moe_experts", 1)))
        block_topk = int(getattr(blk, "moe_top_k", default_topk))
        block_world_size = int(
            getattr(
                blk,
                "world_size",
                getattr(blk.mlp, "world_size", _cfg_get(cfg, "moe_world_size", 1)),
            )
        )
        if block_world_size < 1:
            block_world_size = 1
        total_experts = int(
            getattr(
                blk,
                "tot_expert",
                _cfg_get(
                    cfg,
                    "tot_experts",
                    _cfg_get(cfg, "moe_experts", E_local * block_world_size),
                ),
            )
        )
        if total_experts <= 0:
            total_experts = E_local * block_world_size
        if verbose:
            print(
                f"[INJECT] block {i}: E_local={E_local}, total_experts={total_experts}, "
                f"world_size={block_world_size}, topk={block_topk}, "
                f"fc1_w={tuple(fc1_w.shape)}, fc2_w={tuple(fc2_w.shape)}"
            )

        # MoE expert keys (FMoELinear 기반)
        k_e1_w = f"blocks.{i}.mlp.experts.htoh4.weight"
        k_e1_b = f"blocks.{i}.mlp.experts.htoh4.bias"
        k_e2_w = f"blocks.{i}.mlp.experts.h4toh.weight"
        k_e2_b = f"blocks.{i}.mlp.experts.h4toh.bias"

        if moe_mlp_ratio == 1.0 or moe_mlp_ratio == 1:
            # ---------------------------
            # moe_mlp_ratio=1 : 분할 upcycling (+ optional GELU scaling)
            # ---------------------------
            if verbose:
                print(f"[INJECT] block {i}: moe_mlp_ratio=1 path (split upcycling)")

            hidden_dim = fc1_w.shape[0]
            expected_e1_w = model_sd.get(k_e1_w, None)
            if expected_e1_w is not None and expected_e1_w.ndim == 3:
                expert_hidden_dim = int(expected_e1_w.shape[1])
                granularity = hidden_dim // expert_hidden_dim
            else:
                granularity = 4
            assert granularity > 0 and hidden_dim % granularity == 0, (
                f"block {i}: invalid granularity={granularity} for hidden_dim={hidden_dim}"
            )
            assert total_experts % granularity == 0, (
                f"block {i}: total_experts={total_experts} must be divisible by granularity={granularity}"
            )
            expansion_rate = total_experts // granularity  # E in sqrt(E*G^2/T)
            if verbose:
                print(
                    f"[INJECT] block {i}: hidden_dim={hidden_dim}, granularity={granularity}, "
                    f"total_experts={total_experts}, expansion_rate={expansion_rate}"
                )

            # GELU 기준 scaling: sqrt(E*G^2/T)
            # NOTE: 2025 Eq.2는 squared-ReLU 기반이라 GELU에 그대로 적용하지 않음.
            if use_weight_scaling and moe_router_pre_softmax:
                moe_activation_scale = (expansion_rate * granularity * granularity) / float(max(block_topk, 1))
                weight_scale = moe_activation_scale ** 0.5
            else:
                moe_activation_scale = 1.0
                weight_scale = 1.0
            if verbose:
                print(
                    f"[INJECT] block {i}: moe_activation_scale={moe_activation_scale:.4f}, "
                    f"weight_scale={weight_scale:.4f}"
                )

            # 1) scale dense weights
            fc1_w_scaled = fc1_w * weight_scale
            fc2_w_scaled = fc2_w * weight_scale
            fc1_b_scaled = fc1_b * weight_scale

            # 2) split into granularity chunks
            fc1_w_chunks = fc1_w_scaled.chunk(granularity, dim=0)  # 4 chunks: [hidden/4, embed]
            fc2_w_chunks = fc2_w_scaled.chunk(granularity, dim=1)  # 4 chunks: [embed, hidden/4]
            fc1_b_chunks = fc1_b_scaled.chunk(granularity, dim=0)  # 4 chunks: [hidden/4]

            # ---- build a template group of granularity experts ----
            template_e1_w = torch.stack(fc1_w_chunks, dim=0)  # [G, hidden/G, embed]
            template_e1_b = torch.stack(fc1_b_chunks, dim=0)  # [G, hidden/G]
            template_e2_w = torch.stack(fc2_w_chunks, dim=0)  # [G, embed, hidden/G]

            # fc2 bias: official-style => repeat per expert
            # local shape should be [E_local, embed]
            template_e2_b = fc2_b.unsqueeze(0).repeat(granularity, 1)  # [G, embed]

            # ---- fill local experts ----
            # local_experts가 4의 배수면 repeat로 채우고, 아니면 앞에서부터 잘라서 채움
            if E_local % granularity == 0:
                reps = E_local // granularity
                if verbose:
                    print(f"[INJECT] block {i}: E_local={E_local} divisible by granularity={granularity}, reps={reps} (repeat mode)")
                e1_w = template_e1_w.repeat(reps, 1, 1).contiguous()  # [E_local, hidden/4, embed]
                e1_b = template_e1_b.repeat(reps, 1).contiguous()     # [E_local, hidden/4]
                e2_w = template_e2_w.repeat(reps, 1, 1).contiguous()  # [E_local, embed, hidden/4]
                e2_b = template_e2_b.repeat(reps, 1).contiguous()     # [E_local, embed]
            else:
                # 예: E_local=6 같은 이상 케이스 방어
                if verbose:
                    print(f"[INJECT] block {i}: E_local={E_local} NOT divisible by granularity={granularity}, using truncate mode")
                e1_w = template_e1_w[:E_local].contiguous()
                e1_b = template_e1_b[:E_local].contiguous()
                e2_w = template_e2_w[:E_local].contiguous()
                e2_b = fc2_b.unsqueeze(0).repeat(E_local, 1).contiguous()

        else:
            # ---------------------------
            # moe_mlp_ratio=4 : local experts만큼 단순 복사
            # ---------------------------
            if verbose:
                print(f"[INJECT] block {i}: moe_mlp_ratio={moe_mlp_ratio} path (simple copy, E_local={E_local})")
            e1_w = fc1_w.unsqueeze(0).repeat(E_local, 1, 1).contiguous()
            e1_b = fc1_b.unsqueeze(0).repeat(E_local, 1).contiguous()
            e2_w = fc2_w.unsqueeze(0).repeat(E_local, 1, 1).contiguous()
            e2_b = fc2_b.unsqueeze(0).repeat(E_local, 1).contiguous()

        # 항상 state_dict에 기록
        state_dict[k_e1_w] = e1_w
        state_dict[k_e1_b] = e1_b
        state_dict[k_e2_w] = e2_w
        state_dict[k_e2_b] = e2_b

        # ---- logging / shape verify ----
        if verbose and (i in sample_blocks):
            print(f"[INJECT] --- block {i} summary (sample_blocks) ---")
            print(f"[INJECT] block {i}: E_local={E_local}, moe_mlp_ratio={moe_mlp_ratio}, topk={block_topk}")
            print(f"  DeiT fc1_w {tuple(fc1_w.shape)}  -> experts.htoh4.w {tuple(e1_w.shape)}")
            print(f"  DeiT fc1_b {tuple(fc1_b.shape)}  -> experts.htoh4.b {tuple(e1_b.shape)}")
            print(f"  DeiT fc2_w {tuple(fc2_w.shape)}  -> experts.h4toh.w {tuple(e2_w.shape)}")
            print(f"  DeiT fc2_b {tuple(fc2_b.shape)}  -> experts.h4toh.b {tuple(e2_b.shape)}")

            if verify_shape:
                for kk, vv in [(k_e1_w, e1_w), (k_e1_b, e1_b), (k_e2_w, e2_w), (k_e2_b, e2_b)]:
                    if kk not in model_sd:
                        print(f"  [WARN] model has no key: {kk}")
                        continue
                    exp_shape = tuple(model_sd[kk].shape)
                    got_shape = tuple(vv.shape)
                    if exp_shape != got_shape:
                        print(f"  [SHAPE MISMATCH] {kk}: model expects {exp_shape}, injected {got_shape}")
                    else:
                        print(f"  [OK] {kk}: shape {got_shape} matches model")

    if verbose:
        print(f"[INJECT] === _inject_moe_expert_from_deit_mlp DONE ===")

    return state_dict

def _auto_virtual_group_size(
    tot_experts,
    *,
    local_experts=None,
    world_size=None,
    dense_hidden=None,
    expert_hidden=None,
):
    tot_experts = int(tot_experts)
    if tot_experts <= 0:
        return 1

    primary = None
    if (
        dense_hidden is not None
        and expert_hidden is not None
        and expert_hidden > 0
        and dense_hidden % expert_hidden == 0
    ):
        primary = int(dense_hidden // expert_hidden)

    if primary is None or primary <= 0:
        if local_experts is not None and int(local_experts) > 0:
            primary = int(local_experts)
        elif world_size is not None and int(world_size) > 0 and tot_experts % int(world_size) == 0:
            primary = int(tot_experts // int(world_size))
        else:
            primary = 1

    g = int(primary)
    if local_experts is not None and int(local_experts) > 0:
        g = math.gcd(g, int(local_experts))
    g = math.gcd(g, tot_experts)

    if g <= 0:
        g = 1
    if tot_experts % g != 0:
        g = 1
    return g


def _inject_virtual_group_init_for_gates(
    state_dict, model,
    *,
    cfg=None,
    init="normal",
    std=0.02,
    verbose=True,
    sample_blocks=(1,)
):
    model_sd = model.state_dict()
    gate_key_pattern = re.compile(
        r"^blocks\.\d+\.(?:mlp\.)?(?:gate(?:\.\d+)?|shared_gate)\.w_gate$"
    )
    gate_keys = [k for k in model_sd.keys() if gate_key_pattern.search(k)]

    if not gate_keys:
        raise KeyError(
            "No gate w_gate keys matched expected patterns: "
            "blocks.{i}.gate.w_gate | blocks.{i}.gate.{j}.w_gate | "
            "blocks.{i}.shared_gate.w_gate | blocks.{i}.mlp.gate.w_gate | "
            "blocks.{i}.mlp.gate.{j}.w_gate"
        )

    if verbose:
        print(f"[VGI] === _inject_virtual_group_init_for_gates START ===")
        print(f"[VGI] found {len(gate_keys)} gate keys to initialize, init={init}, std={std}")

    def build_grouped_w_gate(ref: torch.Tensor, group_size: int) -> torch.Tensor:
        assert ref.ndim == 2, ref.shape
        d_model, tot_experts = ref.shape
        assert group_size >= 1
        assert tot_experts % group_size == 0, (
            f"group_size={group_size} must divide tot_experts={tot_experts}"
        )

        w = torch.empty((d_model, tot_experts), device=ref.device, dtype=ref.dtype)
        if init == "normal":
            torch.nn.init.normal_(w, mean=0.0, std=std)
        else:
            raise ValueError(f"Unsupported init={init}. Use init='normal'.")

        if group_size == 1:
            return w

        num_groups = tot_experts // group_size
        chunks = torch.tensor_split(w, num_groups, dim=1)
        proto = chunks[0]
        return torch.cat([proto for _ in range(num_groups)], dim=1).contiguous()

    for k in gate_keys:
        ref = state_dict.get(k, model_sd[k])
        ref_src = "state_dict" if k in state_dict else "model_sd (fresh)"
        tot_experts = int(ref.shape[1])

        m = re.search(r"^blocks\.(\d+)\.", k)
        blk_id = int(m.group(1)) if m else None
        blk = None
        blk_mlp = None
        if blk_id is not None and hasattr(model, "blocks") and blk_id < len(model.blocks):
            blk = model.blocks[blk_id]
            blk_mlp = getattr(blk, "mlp", None)

        local_experts = getattr(blk_mlp, "num_expert", _cfg_get(cfg, "moe_experts_local", None))
        world_size = getattr(
            blk,
            "world_size",
            getattr(blk_mlp, "world_size", _cfg_get(cfg, "moe_world_size", 1)),
        )
        if world_size is None or int(world_size) < 1:
            world_size = 1
        world_size = int(world_size)

        dense_hidden = None
        expert_hidden = None
        if blk_id is not None:
            k_fc1_w = f"blocks.{blk_id}.mlp.fc1.weight"
            if k_fc1_w in state_dict:
                dense_hidden = int(state_dict[k_fc1_w].shape[0])
            k_e1_w = f"blocks.{blk_id}.mlp.experts.htoh4.weight"
            if k_e1_w in model_sd:
                expert_hidden = int(model_sd[k_e1_w].shape[1])
            elif k_e1_w in state_dict:
                expert_hidden = int(state_dict[k_e1_w].shape[1])

        group_size = _auto_virtual_group_size(
            tot_experts,
            local_experts=local_experts,
            world_size=world_size,
            dense_hidden=dense_hidden,
            expert_hidden=expert_hidden,
        )

        new_w = build_grouped_w_gate(ref, group_size).cpu()
        state_dict[k] = new_w

        if verbose:
            if blk_id in sample_blocks:
                print(
                    f"[VGI] {k}: ref_src={ref_src}, shape={tuple(ref.shape)}, "
                    f"tot={tot_experts}, local={local_experts}, world={world_size}, G={group_size}"
                )
                if group_size == 1:
                    print("[VGI]   G=1 -> virtual grouping is no-op (normal init only)")
                print(f"[VGI]   new_w stats: mean={new_w.mean():.5f}, std={new_w.std():.5f}")
            else:
                suffix = " (virtual grouping no-op)" if group_size == 1 else ""
                print(f"[VGI] {k} <- initialized {tuple(new_w.shape)}, G={group_size}{suffix}")

    if verbose:
        print(f"[VGI] === _inject_virtual_group_init_for_gates DONE ===")

    return state_dict
