#!/usr/bin/env bash
set -euo pipefail

# r1_a2 + num_heads=12 (300 epochs)
# moe_cv_weight=0.01, vmoe_noisy_std=0.5, mixup=0.4, cutmix=0.2, smoothing=0.05

CUDA_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC="${NPROC_PER_NODE:-8}"

RUN_NAME="deit_small_scratch_hard_r1_a2_h12"
OUT_DIR="output_dir/pretrain/${RUN_NAME}"
LOG_DIR="logs/pretrain"

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

cd "$(dirname "$0")/.."

CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" PYTHONPATH="$(pwd)" \
torchrun --nproc_per_node="${NPROC}" --module pretrain.train \
  --config pretrain/configs/deit_moe_small.yaml \
  --config-path configs/path_env.yml \
  --dataset-name ImageNet1K \
  --epochs 300 \
  --eval-freq 10 \
  --save-freq 10 \
  --output-dir "${OUT_DIR}" \
  --deit-init-mode scratch \
  --distillation-type hard \
  --distilled true \
  --distillation-alpha 0.5 \
  --moe-top-k 4 \
  --moe-data-distributed true \
  --moe-cv-weight 0.01 \
  --vmoe-noisy-std 0.5 \
  --mixup 0.4 \
  --cutmix 0.2 \
  --smoothing 0.05 \
  --use-wandb \
  --wandb-project pretrain_h12 \
  --wandb-name "${RUN_NAME}" \
  2>&1 | tee "${LOG_DIR}/${RUN_NAME}.log"
