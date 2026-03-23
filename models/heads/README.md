# Decoder Heads

Dense prediction을 위한 decoder/prediction head 모듈.

## 파일 목록

| 파일 | 설명 | Config 값 (`head`) |
|------|------|-------------------|
| `decoder_head.py` | 모든 head의 base class (`BaseDecodeHead`) | - |
| `vit_up_head.py` | ViT feature를 원본 해상도로 upsample | `VisionTransformerUpHead` |
| `aspp.py` | DeepLab ASPP head (ResNet용) | `DeepLabHead` |

## 참고

- Token MoE 전용 head는 `models/moe/token/vit_up_head.py`에 위치
- HRNet head (`HighResolutionHead`)는 `models/backbones/seg_hrnet.py`에 포함

## 사용법

Config YAML에서 `head` 필드로 선택:
```yaml
head: VisionTransformerUpHead
head_kwargs:
  features: 384
  num_conv: 4
```
