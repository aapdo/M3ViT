#!/bin/bash
set -e

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCHELASTIC_ERROR_FILE=/home/jy/m3vit/torchelastic_error.json

# GPU 모니터링 시작
nvidia-smi dmon -i 0,1 -s um -d 5 > /home/jy/m3vit/gpu_monitor_batch8.log 2>&1 &
GPU_MON_PID=$!
trap "kill $GPU_MON_PID 2>/dev/null" EXIT

echo "=== EXP: std=0, trBatch=8, accumulation_steps=1 (원본 동일 설정) ==="
torchrun --nproc_per_node=2 train_fastmoe.py \
  --config_exp configs/nyud/vit_moe/pup_moe_vit_small_multi_task_baseline.yml \
  --use_checkpointing False --use_cv_loss True \
  --moe_gate_type noisy_vmoe --moe_experts 16 --moe_top_k 4 --moe_mlp_ratio 1 \
  --pos_emb_from_pretrained True --backbone_random_init False \
  --task_one_hot False --multi_gate True --save_dir output_dir --trBatch 8 --accumulation_steps 1 \
  --pretrained output_dir/pretrained_model/deit_small_scratch_hard_r2_a3/checkpoint_best.pth \
  --vmoe_noisy_std 0 --gate_task_specific_dim -1 \
  --use_wandb --wandb_project m3vit_pretrained_nyud --wandb_name nyud_r2a3_origin_std0_batch8 \
  > nohup_exp_batch8_std0.out 2>&1

echo "=== BATCH8 EXPERIMENT DONE ==="
