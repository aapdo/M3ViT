# ImageNet Pretrain Skeleton (DeiT-style)

This directory provides a DeiT-inspired pretraining pipeline for MoE ViT encoders in this repo.

## Goals

- Keep the training loop close to DeiT structure (`train.py`, `engine`, `datasets`, `samplers`, `losses`).
- Reuse this repository's encoder implementation (`models/ckpt_vision_transformer_moe.py`).
- Export checkpoints in a format directly loadable by the existing MTL flow (`--pretrained` + `cvt_state_dict`).

## Main entrypoints

- `pretrain/train.py`: ImageNet pretrain for MoE ViT classification.
- `pretrain/export_to_mtl.py`: Convert pretrain checkpoint to MTL-compatible checkpoint.

## Checkpoint outputs

Pretraining now writes two checkpoint families:

- Resume checkpoints (rank-local expert layout):
  - `checkpoint_latest.pth`
  - `checkpoint_best.pth`
- MTL-ready checkpoints (global expert layout):
  - `mtl_latest_global.pth`
  - `mtl_best_global.pth`

Use `mtl_*_global.pth` for downstream `--pretrained` in MTL training.

## Example

```bash
torchrun --nproc_per_node=8 pretrain/train.py \
  --config pretrain/configs/deit_moe_small.yaml \
  --data-path /path/to/imagenet \
  --eval-freq 10 \
  --dev-test true \
  --output-dir /path/to/output

python pretrain/export_to_mtl.py \
  --checkpoint /path/to/output/mtl_best_global.pth \
  --output /path/to/output/mtl_pretrained.pth
```

If `configs/path_env.yml` contains `dataset_roots.ImageNet1K`, you can omit `--data-path`:

```bash
torchrun --nproc_per_node=8 pretrain/train.py \
  --config pretrain/configs/deit_moe_small.yaml \
  --config-path configs/path_env.yml \
  --dataset-name ImageNet1K \
  --output-dir /path/to/output

# Enable Weights & Biases logging
torchrun --nproc_per_node=8 pretrain/train.py \
  --config pretrain/configs/deit_moe_small.yaml \
  --config-path configs/path_env.yml \
  --dataset-name ImageNet1K \
  --output-dir /path/to/output \
  --use-wandb \
  --wandb-project m3vit-pretrain \
  --wandb-name deit_moe_small
```

If that local path does not contain `train/` and `val/`, the loader will
automatically bootstrap ImageNet-1k from Hugging Face (`ILSVRC/imagenet-1k`)
and materialize it into that same local path as an `ImageFolder` layout.

Hugging Face auto-download is also supported by setting a HF dataset URI:

```bash
torchrun --nproc_per_node=8 pretrain/train.py \
  --config pretrain/configs/deit_moe_small.yaml \
  --config-path configs/path_env.yml \
  --data-path hf://ILSVRC/imagenet-1k \
  --output-dir /path/to/output
```

Notes:
- Make sure you already have access to the gated dataset on Hugging Face.
- Token is read from `HF_TOKEN`/`HUGGINGFACE_HUB_TOKEN` environment variables.
- If `configs/path_env.yml` has `huggingface_access_token`, it is auto-injected into those env vars.
- Install dependencies first: `pip install -U "datasets[vision]" huggingface_hub`.

V-MoE-style recipe (adapted to this codebase):

```bash
torchrun --nproc_per_node=8 pretrain/train.py \
  --config pretrain/configs/vmoe_style_moe_small.yaml \
  --config-path configs/path_env.yml \
  --dataset-name ImageNet1K \
  --output-dir /path/to/output
```

Available V-MoE-style presets:
- `pretrain/configs/vmoe_style_moe_tiny.yaml`
- `pretrain/configs/vmoe_style_moe_small.yaml`
- `pretrain/configs/vmoe_style_moe_base.yaml`

## DeiT Init Modes

The pretrain entrypoint supports three explicit initialization modes via
`--deit-init-mode` (alias `--init-mode`):

- `scratch`: random init pretraining (`random_init=True`)
- `deit_warm_start`: load DeiT weights and apply Nvidia-style expert upcycling
  (dense MLP split to experts, requires expert hidden = dense hidden / 4)
- `deit_upcycling`: load DeiT weights and initialize MoE experts from DeiT MLP
- `auto`: compatibility mode (maps to `scratch` when `random_init=true`, else `deit_upcycling`)

Examples:

```bash
# 1) Scratch pretrain
torchrun --nproc_per_node=8 pretrain/train.py \
  --config pretrain/configs/deit_moe_small.yaml \
  --deit-init-mode scratch \
  --output-dir /path/to/output

# 2) DeiT warm-start pretrain
torchrun --nproc_per_node=8 pretrain/train.py \
  --config pretrain/configs/deit_moe_small.yaml \
  --deit-init-mode deit_warm_start \
  --moe-mlp-ratio 1.0 \
  --output-dir /path/to/output

# 3) DeiT upcycling pretrain
torchrun --nproc_per_node=8 pretrain/train.py \
  --config pretrain/configs/deit_moe_small.yaml \
  --deit-init-mode deit_upcycling \
  --output-dir /path/to/output
```

If you have train_fastmoe-style shard directory (`0.pth`, `1.pth`, ...):

```bash
python pretrain/export_to_mtl.py \
  --checkpoint /path/to/shard_dir \
  --output /path/to/output/mtl_global.pth
```

## Notes

- Distillation flags are scaffolded but disabled by default (`--distillation-type none`).
- The default pipeline uses AdamW + warmup + cosine decay (DeiT-style pretrain setup).
- Evaluation runs every `--eval-freq` epochs (default: `10`, and always at final epoch).
- `--dev-test true` runs one validation pass before the training loop starts.
- Model construction `__init__` arguments are logged via `patch_and_log_initializations` by default (`--log-initializations true`).
- `--output-dir` is treated as a base path; each run writes into a timestamped subdirectory (`MMDD_HHMM`).
- W&B options: `--use-wandb`, `--wandb-project`, `--wandb-entity`, `--wandb-name`, `--wandb-mode [online|offline|disabled]`.
- Single-file rank-local MoE checkpoints are intentionally rejected by the MTL loader.
  If a checkpoint was produced in multi-GPU mode and only one local file is available,
  it cannot be fully reconstructed; use the full shard directory export path.
