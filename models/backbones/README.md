# Backbones

공통 backbone 인코더 모듈.

## 파일 목록

| 파일 | 설명 | Config 값 (`backbone`) |
|------|------|----------------------|
| `vit.py` | Vision Transformer (DeiT 기반) | `VisionTransformer` |
| `resnet.py` | ResNet-18/50, ResNeXt, Mixture ResNet | `resnet18`, `resnet50`, `mixture_inner_resnet_50` |
| `resnet_dilated.py` | Dilated ResNet (DeepLab용) | 내부적으로 사용 |
| `seg_hrnet.py` | HRNet-W18 | `hrnet_w18` |
| `mobilenetv3.py` | MobileNetV3 Large/Small | `mobilenetv3` |
| `vits_gate.py` | ViT + MoCo 스타일 + ConvStem + Gate | 내부적으로 사용 |

## 사용법

Config YAML에서 `backbone` 필드로 선택:
```yaml
backbone: VisionTransformer  # 또는 resnet50, hrnet_w18 등
backbone_kwargs:
  model_name: deit_small_distilled_patch16_224
  random_init: False
```

모델 인스턴스화는 `utils/common_config.py`의 `get_backbone()` 에서 처리.
