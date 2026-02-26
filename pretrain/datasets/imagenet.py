"""ImageNet datasets and dataloaders."""

import io
import os

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, RandomSampler, SequentialSampler
from torchvision import datasets
from PIL import Image

from .samplers import RASampler
from .transforms import build_imagenet_transform


def _is_hf_dataset_path(data_path):
    return str(data_path or "").startswith("hf://")


def _get_hf_dataset_id(data_path):
    dataset_id = str(data_path or "")[len("hf://") :].strip()
    if dataset_id.startswith("/"):
        dataset_id = dataset_id[1:]
    return dataset_id


def _resolve_hf_token():
    for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        value = str(os.environ.get(key, "") or "").strip()
        if value:
            return value
    return None


def _load_hf_split(dataset_id, split, token=None, cache_dir=None):
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise ImportError(
            "Hugging Face dataset support requires `datasets` and `huggingface_hub`. "
            "Install with: pip install -U \"datasets[vision]\" huggingface_hub"
        ) from exc

    kwargs = {"split": split}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    if token:
        try:
            return load_dataset(dataset_id, token=token, **kwargs)
        except TypeError:
            # Backward compatibility for old `datasets` versions.
            return load_dataset(dataset_id, use_auth_token=token, **kwargs)
    return load_dataset(dataset_id, **kwargs)


class HFImageNetDataset(Dataset):
    """Wrap a Hugging Face image classification dataset as torchvision-style Dataset."""

    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform
        self.num_classes, self.classes = self._infer_classes(hf_dataset)

    @staticmethod
    def _infer_classes(hf_dataset):
        features = getattr(hf_dataset, "features", None) or {}
        label_feature = features.get("label", None)
        class_names = getattr(label_feature, "names", None)
        if class_names:
            names = list(class_names)
            return len(names), names
        num_classes = getattr(label_feature, "num_classes", None)
        if num_classes is not None:
            n = int(num_classes)
            return n, [str(i) for i in range(n)]
        # Fallback for unknown label schema.
        return 1000, [str(i) for i in range(1000)]

    @staticmethod
    def _ensure_pil_rgb(image):
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        if isinstance(image, dict) and image.get("bytes", None) is not None:
            return Image.open(io.BytesIO(image["bytes"])).convert("RGB")
        if hasattr(image, "ndim"):
            return Image.fromarray(image).convert("RGB")
        raise TypeError(f"Unsupported image type from Hugging Face dataset: {type(image)}")

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, index):
        sample = self.hf_dataset[index]
        image = self._ensure_pil_rgb(sample["image"])
        target = int(sample["label"])

        if self.transform is not None:
            image = self.transform(image)
        return image, target


def _build_hf_imagenet_datasets(args):
    dataset_id = _get_hf_dataset_id(getattr(args, "data_path", ""))
    if not dataset_id:
        raise ValueError(
            "Hugging Face mode expects --data-path in the form "
            "'hf://<dataset_id>' (e.g. hf://ILSVRC/imagenet-1k)."
        )

    train_split = str(getattr(args, "hf_train_split", "train") or "train")
    val_split = str(getattr(args, "hf_val_split", "validation") or "validation")
    cache_dir = str(getattr(args, "hf_cache_dir", "") or "").strip() or None
    token = _resolve_hf_token()

    dataset_train_hf = _load_hf_split(dataset_id, split=train_split, token=token, cache_dir=cache_dir)
    dataset_val_hf = _load_hf_split(dataset_id, split=val_split, token=token, cache_dir=cache_dir)

    dataset_train = HFImageNetDataset(dataset_train_hf, transform=build_imagenet_transform(True, args))
    dataset_val = HFImageNetDataset(dataset_val_hf, transform=build_imagenet_transform(False, args))
    return dataset_train, dataset_val, dataset_train.num_classes


def build_imagenet_datasets(args):
    if _is_hf_dataset_path(getattr(args, "data_path", "")):
        return _build_hf_imagenet_datasets(args)

    train_dir = os.path.join(args.data_path, "train")
    val_dir = os.path.join(args.data_path, "val")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"ImageNet train dir not found: {train_dir}")
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"ImageNet val dir not found: {val_dir}")

    dataset_train = datasets.ImageFolder(train_dir, transform=build_imagenet_transform(True, args))
    dataset_val = datasets.ImageFolder(val_dir, transform=build_imagenet_transform(False, args))
    return dataset_train, dataset_val, len(dataset_train.classes)


def build_imagenet_loaders(dataset_train, dataset_val, args):
    if args.distributed:
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train,
                num_replicas=args.world_size,
                rank=args.rank,
                shuffle=True,
                num_repeats=3,
            )
        else:
            sampler_train = DistributedSampler(
                dataset_train,
                num_replicas=args.world_size,
                rank=args.rank,
                shuffle=True,
            )

        if args.dist_eval:
            sampler_val = DistributedSampler(
                dataset_val,
                num_replicas=args.world_size,
                rank=args.rank,
                shuffle=False,
            )
        else:
            sampler_val = SequentialSampler(dataset_val)
    else:
        sampler_train = RandomSampler(dataset_train)
        sampler_val = SequentialSampler(dataset_val)

    loader_train = DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    loader_val = DataLoader(
        dataset_val,
        sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    return loader_train, loader_val
