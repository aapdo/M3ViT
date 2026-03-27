#!/bin/bash
set -e

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# GPU는 컨테이너 레벨에서 device=1,2로 매핑되어 있으므로 별도 설정 불필요
export TORCHELASTIC_ERROR_FILE=/home/jy/m3vit/torchelastic_error.json

# GPU 모니터링 시작
nvidia-smi dmon -i 0,1 -s um -d 5 > /home/jy/m3vit/gpu_monitor.log 2>&1 &
GPU_MON_PID=$!
trap "kill $GPU_MON_PID 2>/dev/null" EXIT

echo "=== EXP 1: std=0 ==="
torchrun --nproc_per_node=2 train_fastmoe.py \
  --config_exp configs/nyud/vit_moe/pup_moe_vit_small_multi_task_baseline.yml \
  --use_checkpointing False --use_cv_loss True \
  --moe_gate_type noisy_vmoe --moe_experts 16 --moe_top_k 4 --moe_mlp_ratio 1 \
  --pos_emb_from_pretrained True --backbone_random_init False \
  --task_one_hot False --multi_gate True --save_dir output_dir --trBatch 4 --accumulation_steps 2 \
  --pretrained output_dir/pretrained_model/deit_small_scratch_hard_r2_a3/checkpoint_best.pth \
  --vmoe_noisy_std 0 --gate_task_specific_dim -1 \
  --use_wandb --wandb_project m3vit_pretrained_nyud --wandb_name nyud_r2a3_origin_std0 \
  > nohup_exp1_std0.out 2>&1

echo "=== EXP 2: std=0.5 ==="
torchrun --nproc_per_node=2 train_fastmoe.py \
  --config_exp configs/nyud/vit_moe/pup_moe_vit_small_multi_task_baseline.yml \
  --use_checkpointing False --use_cv_loss True \
  --moe_gate_type noisy_vmoe --moe_experts 16 --moe_top_k 4 --moe_mlp_ratio 1 \
  --pos_emb_from_pretrained True --backbone_random_init False \
  --task_one_hot False --multi_gate True --save_dir output_dir --trBatch 4 --accumulation_steps 2 \
  --pretrained output_dir/pretrained_model/deit_small_scratch_hard_r2_a3/checkpoint_best.pth \
  --vmoe_noisy_std 0.5 --gate_task_specific_dim -1 \
  --use_wandb --wandb_project m3vit_pretrained_nyud --wandb_name nyud_r2a3_origin_std05 \
  > nohup_exp2_std05.out 2>&1

echo "=== EXP 3: std=0.5 + drop_rate=0.1 ==="
torchrun --nproc_per_node=2 train_fastmoe.py \
  --config_exp configs/nyud/vit_moe/pup_moe_vit_small_multi_task_baseline_drop0.1.yml \
  --use_checkpointing False --use_cv_loss True \
  --moe_gate_type noisy_vmoe --moe_experts 16 --moe_top_k 4 --moe_mlp_ratio 1 \
  --pos_emb_from_pretrained True --backbone_random_init False \
  --task_one_hot False --multi_gate True --save_dir output_dir --trBatch 4 --accumulation_steps 2 \
  --pretrained output_dir/pretrained_model/deit_small_scratch_hard_r2_a3/checkpoint_best.pth \
  --vmoe_noisy_std 0.5 --gate_task_specific_dim -1 \
  --use_wandb --wandb_project m3vit_pretrained_nyud --wandb_name nyud_r2a3_origin_std05_drop01 \
  > nohup_exp3_std05_drop01.out 2>&1

echo "=== EXP 4: std=0.5 + drop_rate=0.1 + drop_path=0.1 ==="
torchrun --nproc_per_node=2 train_fastmoe.py \
  --config_exp configs/nyud/vit_moe/pup_moe_vit_small_multi_task_baseline_drop0.1_droppath0.1.yml \
  --use_checkpointing False --use_cv_loss True \
  --moe_gate_type noisy_vmoe --moe_experts 16 --moe_top_k 4 --moe_mlp_ratio 1 \
  --pos_emb_from_pretrained True --backbone_random_init False \
  --task_one_hot False --multi_gate True --save_dir output_dir --trBatch 4 --accumulation_steps 2 \
  --pretrained output_dir/pretrained_model/deit_small_scratch_hard_r2_a3/checkpoint_best.pth \
  --vmoe_noisy_std 0.5 --gate_task_specific_dim -1 \
  --use_wandb --wandb_project m3vit_pretrained_nyud --wandb_name nyud_r2a3_origin_std05_drop01_droppath01 \
  > nohup_exp4_std05_drop01_droppath01.out 2>&1

echo "=== ALL EXPERIMENTS DONE ==="
