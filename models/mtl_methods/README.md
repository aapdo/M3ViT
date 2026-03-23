# MTL Comparison Methods

Multi-Task Learning 비교 모델 (baseline 대비 성능 비교용).

## 파일 목록

| 파일 | 모델 | 논문 | Config 값 (`model`) |
|------|------|------|-------------------|
| `cross_stitch.py` | Cross-Stitch Networks | CVPR 2016 | `cross_stitch` |
| `mtan.py` | Multi-Task Attention Network | CVPR 2019 | `mtan` |
| `nddr_cnn.py` | NDDR-CNN | CVPR 2019 | `nddr_cnn` |
| `padnet.py` | PAD-Net | CVPR 2018 | `pad_net` |
| `mti_net.py` | MTI-Net | ECCV 2020 | `mti_net` |
| `papnet.py` | PAP-Net (ViT 버전) | - | `papnet` |
| `papnet_new.py` | PAP-Net 개선 | - | - |
| `Jtrl.py` | Joint Task Relationship Learning | - | `jtrl` |

## 실행 방법

```bash
# PAD-Net on PASCAL
python main.py --config_env configs/env.yml \
    --config_exp configs/pascal/hrnet18/pad_net.yml

# MTAN on NYUD
python main.py --config_env configs/env.yml \
    --config_exp configs/nyud/resnet50/mtan.yml
```
