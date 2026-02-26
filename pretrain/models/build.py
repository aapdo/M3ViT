"""Model builder for ImageNet pretraining."""

import torch

from .moe_vit_cls import MoEViTConfig, MoEViTForImageNet


_MODEL_SPECS = {
    "moe_vit_tiny": {
        "model_name": "vit_tiny_patch16_224",
        "embed_dim": 192,
        "depth": 12,
        "num_heads": 3,
        "mlp_ratio": 4.0,
    },
    "moe_vit_small": {
        "model_name": "vit_small_patch16_224",
        "embed_dim": 384,
        "depth": 12,
        "num_heads": 6,
        "mlp_ratio": 4.0,
    },
    "moe_vit_base": {
        "model_name": "vit_base_patch16_224",
        "embed_dim": 768,
        "depth": 12,
        "num_heads": 12,
        "mlp_ratio": 4.0,
    },
}


def build_model(args):
    if args.model not in _MODEL_SPECS:
        raise ValueError(f"Unknown model '{args.model}'. Available: {list(_MODEL_SPECS.keys())}")

    spec = _MODEL_SPECS[args.model]
    model_name = args.model_name if getattr(args, "model_name", "") else spec["model_name"]
    embed_dim = int(args.embed_dim) if getattr(args, "embed_dim", -1) > 0 else spec["embed_dim"]
    depth = int(args.depth) if getattr(args, "depth", -1) > 0 else spec["depth"]
    num_heads = int(args.num_heads) if getattr(args, "num_heads", -1) > 0 else spec["num_heads"]
    mlp_ratio = float(args.mlp_ratio) if getattr(args, "mlp_ratio", -1.0) > 0 else spec["mlp_ratio"]

    world_size = torch.distributed.get_world_size() if args.distributed else 1
    if args.moe_experts % world_size != 0:
        raise ValueError(
            f"moe_experts ({args.moe_experts}) must be divisible by world size ({world_size})"
        )
    moe_experts_local = args.moe_experts // world_size

    cfg = MoEViTConfig(
        model_name=model_name,
        img_size=args.input_size,
        patch_size=args.patch_size,
        in_chans=args.in_chans,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=args.qkv_bias,
        distilled=getattr(args, "distilled", False),
        drop_rate=args.drop,
        attn_drop_rate=args.attn_drop_rate,
        drop_path_rate=args.drop_path,
        num_classes=args.nb_classes,
        random_init=args.random_init,
        pos_embed_interp=args.pos_embed_interp,
        align_corners=args.align_corners,
        moe_mlp_ratio=args.moe_mlp_ratio,
        moe_experts=moe_experts_local,
        moe_top_k=args.moe_top_k,
        moe_gate_type=args.moe_gate_type,
        vmoe_noisy_std=args.vmoe_noisy_std,
        gate_dim=embed_dim if args.gate_dim < 0 else args.gate_dim,
        gate_task_specific_dim=args.gate_task_specific_dim,
        multi_gate=args.multi_gate,
        world_size=world_size,
    )

    return MoEViTForImageNet(cfg)
