## Git Rules

### Commit
- commit message에 "Claude", "AI", "Co-authored" 등 AI 관련 문구를 절대 포함하지 마라
- commit author는 항상 기존 git config의 user.name, user.email을 사용해라
- commit message는 일반적인 컨벤션(feat:, fix:, chore:, refactor: 등)만 사용해라

### Branch
- branch 이름에 "claude", "ai", "auto" 등 AI 관련 문구를 절대 포함하지 마라
- branch 이름은 컨벤션을 따라라: `<type>/<간결한-설명>` (예: feat/token-moe-shareability, fix/imagenet-label-mapping, refactor/models-directory)
- branch 이름만 보고 어떤 작업인지 알 수 있도록 의미 있는 이름을 사용해라
- type: feat, fix, refactor, chore, docs, test 등

## Project Structure

### 디렉토리 구조
- `configs/` — 데이터셋별 실험 설정 YAML (nyud, pascal, cityscapes)
- `data/` — 데이터셋 로더 및 전처리 (nyud.py, pascal_context.py, cityscapes.py, custom_transforms.py)
- `env_setup/` — Docker 및 환경 설정 파일
- `evaluation/` — 태스크별 평가 스크립트 (semseg, depth, normals, edge, sal, human_parts)
- `losses/` — 멀티태스크 loss 함수 및 loss scheme
- `models/` — 모델 코드 전체
  - `models/backbones/` — 백본 네트워크 (ResNet, HRNet, ViT, MoE ViT 등)
  - `models/heads/` — 태스크별 디코더 헤드 (ASPP, ViT upsampling head 등)
  - `models/mtl_methods/` — 멀티태스크 학습 방법론 (Cross-Stitch, MTAN, PAD-Net, MTI-Net 등)
  - `models/moe/` — MoE (Mixture of Experts) 관련 코드
    - `models/moe/origin` — M3ViT 공식 GitHub의 모델 코드를 그대로 가져온 것
    - `models/moe/ckpt` — origin 코드에 VRAM 부족 문제 해결을 위해 gradient checkpointing을 적용한 버전
    - `models/moe/token` — 현재 제안하는 논문에 대한 코드
- `pretrain/` — DeiT 사전학습 코드 (Dense_DeiT 기반)
- `resources/` — 논문 그림 등 참고 자료
- `train/` — 학습 유틸리티 (train_utils.py)
- `utils/` — 공통 유틸리티 (config, MoE 헬퍼, wandb 로거, 데이터 경로 등)
- `train_fastmoe.py` — 메인 학습 스크립트 (FastMoE 기반)
- `train_vit.py` — ViT 학습 스크립트 (non-MoE)
- `run_exps.sh`, `run_after_current.sh` — 실험 실행 셸 스크립트

### 사전학습 결과
- `output_dir/pretrain/deit_small_scratch_hard_r2_a3` — 사전학습 모델 학습 로그 및 결과
- `output_dir/pretrain/deit_small_scratch_hard_r1_a3` — 사전학습 모델 학습 로그 및 결과

## M3ViT 논문 벤치마크 (ViT-small 기준)

### NYUD-v2 (2 tasks)
| Model | Backbone | Seg. (mIoU)↑ | Depth (rmse)↓ | ∆m (%)↑ |
|---|---|---|---|---|
| STL-B | ResNet-50 | 43.9 | 0.585 | 0.00 |
| MTL-B | ResNet-50 | 44.4 | 0.587 | +0.41 |
| Cross-Stitch | ResNet-50 | 44.2 | 0.570 | +1.61 |
| M-ViT (MTL-B) | ViT-small | 40.9 | 0.631 | −6.27 |
| M3ViT-Single | MoE ViT-small | 45.3 | 0.600 | +0.31 |
| M3ViT-Multi. | MoE ViT-small | 45.6 | 0.589 | +1.59 |
| M3ViT-Task-cond. | MoE ViT-small | 45.3 | 0.595 | +0.74 |

### PASCAL-Context (5 tasks)
| Model | Backbone | Seg. (mIoU)↑ | Norm. (mErr)↓ | H.Parts (mIoU)↑ | Sal. (mIoU)↑ | Edge (odsF)↑ | ∆m (%)↑ |
|---|---|---|---|---|---|---|---|
| STL-B | ResNet-18 | 66.2 | 13.9 | 59.9 | 66.3 | 68.8 | 0.00 |
| MTL-B | ResNet-18 | 63.8 | 14.9 | 58.6 | 65.1 | 69.2 | −2.86 |
| Cross-Stitch | ResNet-18 | 66.1 | 13.9 | 60.6 | 66.8 | 69.9 | +0.60 |
| M-ViT (MTL-B) | ViT-small | 70.7 | 15.5 | 58.7 | 64.9 | 68.8 | −1.77 |
| M3ViT-Single | MoE ViT-small | 71.5 | 14.8 | 61.2 | 65.9 | 71.5 | +1.40 |
| M3ViT-Multi. | MoE ViT-small | 72.8 | 14.5 | 62.1 | 66.3 | 71.7 | +2.71 |
| M3ViT-Task-cond. | MoE ViT-small | 72.0 | 14.4 | 61.3 | 65.8 | 71.8 | +2.22 |

### 주요 설정
- Backbone: DeiT-small (ViT-small variant)
- MoE: K=4 (top-K), N=16 experts, expert 크기는 standard MLP 대비 1/4
- Load & importance balancing loss weight: 0.01
- Multi-gate 방식이 최종 보고 성능 (M2ViT/M3ViT)

## Current Issues

### M3ViT 재현 성능 차이
- M3ViT 논문에서 제안한 방법대로 사전학습 및 학습을 진행해도, 논문에서 보고한 벤치마크 성능과 큰 차이가 발생함
- 원인 분석 필요: 하이퍼파라미터, 학습 설정, 데이터 전처리, 모델 구현 차이 등
