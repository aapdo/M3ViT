# Checkpoint MoE (Gradient Checkpointing)

Origin MoE에 gradient checkpointing을 적용하여 GPU 메모리 효율을 높인 버전.
현재 `VisionTransformer_moe` backbone의 기본 구현.

## 파일

| 파일 | 설명 |
|------|------|
| `vision_transformer_moe.py` | Checkpointed MoE ViT backbone (`VisionTransformerMoE`) |
| `custom_moe_layer.py` | Checkpointed FMoE MLP layer (`FMoETransformerMLP`) |
| `noisy_gate_vmoe.py` | V-MoE 스타일 noisy gate (`NoisyGate_VMoE`) |

## Batch Size 설정

`trBatch`는 **GPU당 배치 사이즈**이다. 총 배치 = `trBatch × nproc_per_node(GPU 수)`.

| 환경 | trBatch | GPU 수 | 총 배치 |
|------|---------|--------|---------|
| 2080 Ti 11GB × 8 | 2 | 8 | 16 |
| 원본 M3ViT (nproc=2) | 8 | 2 | 16 |

## 실행 방법

### NYUD (semseg + depth)
```bash
torchrun --nproc_per_node=8 train_fastmoe.py \
    --config_exp configs/nyud/vit_moe/pup_moe_vit_small_multi_task_baseline.yml \
    --moe_experts 16 --moe_top_k 2 --moe_mlp_ratio 2 \
    --use_checkpointing True
```

### PASCAL Context (5 tasks)
```bash
torchrun --nproc_per_node=8 train_fastmoe.py \
    --config_exp configs/pascal/vit_moe/pup_moe_vit_small_multi_task_baseline.yml \
    --moe_experts 16 --moe_top_k 2 --moe_mlp_ratio 2 \
    --use_checkpointing True
```

### DeiT MoE 사전학습 모델 사용

사전학습된 checkpoint를 `--pretrained`로 지정하여 fine-tuning.

**NYUD (r2_a3 pretrained)**
```bash
torchrun --nproc_per_node=8 train_fastmoe.py \
    --config_exp configs/nyud/vit_moe/pup_moe_vit_small_multi_task_baseline.yml \
    --pretrained output_dir/pretrained_model/deit_small_scratch_hard_r2_a3/checkpoint_best.pth \
    --moe_gate_type "noisy_vmoe" \
    --moe_experts 16 --moe_top_k 4 --moe_mlp_ratio 1 \
    --moe_data_distributed \
    --vmoe_noisy_std 0 \
    --pos_emb_from_pretrained True \
    --backbone_random_init False \
    --task_one_hot False \
    --multi_gate True \
    --gate_task_specific_dim -1 \
    --save_dir output_dir \
    --use_checkpointing True \
    --use_wandb \
    --wandb_project m3vit-training \
    --wandb_name nyud_small_r2_a3_pretrained
```

**NYUD (r1_a3 pretrained)**
```bash
torchrun --nproc_per_node=8 train_fastmoe.py \
    --config_exp configs/nyud/vit_moe/pup_moe_vit_small_multi_task_baseline.yml \
    --pretrained output_dir/pretrained_model/deit_small_scratch_hard_r1_a3/checkpoint_best.pth \
    --moe_gate_type "noisy_vmoe" \
    --moe_experts 16 --moe_top_k 4 --moe_mlp_ratio 1 \
    --moe_data_distributed \
    --vmoe_noisy_std 0 \
    --pos_emb_from_pretrained True \
    --backbone_random_init False \
    --task_one_hot False \
    --multi_gate True \
    --gate_task_specific_dim -1 \
    --save_dir output_dir \
    --use_checkpointing True \
    --use_wandb \
    --wandb_project m3vit-training \
    --wandb_name nyud_small_r1_a3_pretrained
```

**PASCAL Context (r2_a3 pretrained)**
```bash
torchrun --nproc_per_node=8 train_fastmoe.py \
    --config_exp configs/pascal/vit_moe/pup_moe_vit_small_multi_task_baseline.yml \
    --pretrained output_dir/pretrained_model/deit_small_scratch_hard_r2_a3/checkpoint_best.pth \
    --moe_gate_type "noisy_vmoe" \
    --moe_experts 16 --moe_top_k 4 --moe_mlp_ratio 1 \
    --moe_data_distributed \
    --vmoe_noisy_std 0 \
    --pos_emb_from_pretrained True \
    --backbone_random_init False \
    --task_one_hot False \
    --multi_gate True \
    --gate_task_specific_dim -1 \
    --save_dir output_dir \
    --use_checkpointing True \
    --use_wandb \
    --wandb_project m3vit-training \
    --wandb_name pascal_small_r2_a3_pretrained
```

**PASCAL Context (r1_a3 pretrained)**
```bash
torchrun --nproc_per_node=8 train_fastmoe.py \
    --config_exp configs/pascal/vit_moe/pup_moe_vit_small_multi_task_baseline.yml \
    --pretrained output_dir/pretrained_model/deit_small_scratch_hard_r1_a3/checkpoint_best.pth \
    --moe_gate_type "noisy_vmoe" \
    --moe_experts 16 --moe_top_k 4 --moe_mlp_ratio 1 \
    --moe_data_distributed \
    --vmoe_noisy_std 0 \
    --pos_emb_from_pretrained True \
    --backbone_random_init False \
    --task_one_hot False \
    --multi_gate True \
    --gate_task_specific_dim -1 \
    --save_dir output_dir \
    --use_checkpointing True \
    --use_wandb \
    --wandb_project m3vit-training \
    --wandb_name pascal_small_r1_a3_pretrained
```

## 관련 Config

- `configs/nyud/vit_moe/pup_moe_vit_small_multi_task_baseline.yml`
- `configs/nyud/vit_moe/pup_moe_vit_base_multi_task_baseline.yml`
- `configs/pascal/vit_moe/pup_moe_vit_small_multi_task_baseline.yml`
- `configs/pascal/vit_moe/pup_moe_vit_base_multi_task_baseline.yml`

## Pretrained Model 경로

| 모델 | 경로 |
|------|------|
| DeiT-Small MoE (r2_a3, cv_weight=0.02) | `output_dir/pretrained_model/deit_small_scratch_hard_r2_a3/checkpoint_best.pth` |
| DeiT-Small MoE (r1_a3, cv_weight=0.01) | `output_dir/pretrained_model/deit_small_scratch_hard_r1_a3/checkpoint_best.pth` |
