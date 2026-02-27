"""ImageNet datasets and dataloaders."""

import io
import hashlib
import os
import tempfile
import time

import torch
from torch.utils.data import DataLoader, Dataset, DistributedSampler, RandomSampler, SequentialSampler
from torchvision import datasets
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader
from PIL import Image

from .samplers import RASampler
from .transforms import build_imagenet_transform


_HF_IMAGENET_DATASET_ID_DEFAULT = "ILSVRC/imagenet-1k"
_IMAGEFOLDER_INDEX_CACHE_VERSION = 1


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


def _normalize_class_name(name):
    text = str(name).strip().replace("/", "_").replace("\\", "_")
    return text if text else "unknown"


def _has_local_imagefolder_split(split_dir):
    if not os.path.isdir(split_dir):
        return False
    for class_dir in os.scandir(split_dir):
        if not class_dir.is_dir():
            continue
        for file_entry in os.scandir(class_dir.path):
            if file_entry.is_file():
                return True
    return False


def _infer_hf_class_names(hf_dataset):
    _, class_names = HFImageNetDataset._infer_classes(hf_dataset)
    return [_normalize_class_name(name) for name in class_names]


def _infer_image_extension(image_field):
    default_ext = ".jpg"
    if isinstance(image_field, dict):
        image_path = str(image_field.get("path", "") or "")
        ext = os.path.splitext(image_path)[1].lower()
        if ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            return ext
    return default_ext


def _save_image_field(image_field, image_path):
    if isinstance(image_field, dict) and image_field.get("bytes", None) is not None:
        with open(image_path, "wb") as f:
            f.write(image_field["bytes"])
        return

    image = HFImageNetDataset._ensure_pil_rgb(image_field)
    image.save(image_path, format="JPEG", quality=95)


def _save_hf_split_to_imagefolder(hf_dataset, split_dir, class_names):
    os.makedirs(split_dir, exist_ok=True)
    for idx in range(len(hf_dataset)):
        sample = hf_dataset[idx]
        label = int(sample["label"])
        class_name = class_names[label] if 0 <= label < len(class_names) else str(label)
        class_dir = os.path.join(split_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        image_ext = _infer_image_extension(sample["image"])
        image_path = os.path.join(class_dir, f"{idx:08d}{image_ext}")

        _save_image_field(sample["image"], image_path)

        if (idx + 1) % 10000 == 0:
            print(f"[HF->ImageFolder] {split_dir}: {idx + 1}/{len(hf_dataset)}")


def _maybe_bootstrap_local_imagenet_from_hf(args, train_dir, val_dir):
    has_train = _has_local_imagefolder_split(train_dir)
    has_val = _has_local_imagefolder_split(val_dir)
    if has_train and has_val:
        return

    dataset_name = str(getattr(args, "dataset_name", "ImageNet1K") or "ImageNet1K")
    if dataset_name != "ImageNet1K":
        missing = []
        if not has_train:
            missing.append(train_dir)
        if not has_val:
            missing.append(val_dir)
        raise FileNotFoundError(
            f"Missing dataset folders {missing} and auto-bootstrap is only supported for "
            "dataset_name=ImageNet1K."
        )

    dataset_id = str(getattr(args, "hf_dataset_id", _HF_IMAGENET_DATASET_ID_DEFAULT) or _HF_IMAGENET_DATASET_ID_DEFAULT)
    train_split = str(getattr(args, "hf_train_split", "train") or "train")
    val_split = str(getattr(args, "hf_val_split", "validation") or "validation")
    # Keep Hugging Face cache near dataset root if not explicitly set.
    cache_dir = str(getattr(args, "hf_cache_dir", "") or "").strip()
    if not cache_dir:
        cache_dir = os.path.join(args.data_path, ".hf_cache")
    token = _resolve_hf_token()

    os.makedirs(args.data_path, exist_ok=True)
    print(
        "[HF->ImageFolder] Local ImageNet folder not found. "
        f"Downloading {dataset_id} and materializing to {args.data_path}"
    )

    if not has_train:
        train_hf = _load_hf_split(dataset_id, split=train_split, token=token, cache_dir=cache_dir)
        class_names = _infer_hf_class_names(train_hf)
        _save_hf_split_to_imagefolder(train_hf, train_dir, class_names)

    if not has_val:
        val_hf = _load_hf_split(dataset_id, split=val_split, token=token, cache_dir=cache_dir)
        class_names = _infer_hf_class_names(val_hf)
        _save_hf_split_to_imagefolder(val_hf, val_dir, class_names)


def _dist_initialized():
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def _safe_torch_save(obj, path):
    cache_dir = os.path.dirname(path)
    try:
        os.makedirs(cache_dir, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=cache_dir, delete=False) as tmp:
            tmp_path = tmp.name
        try:
            torch.save(obj, tmp_path)
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        return True
    except Exception as exc:
        print(f"[ImageFolderCache] failed to write cache at {path}: {exc}")
        return False


def _imagefolder_cache_paths(split_dir):
    split_dir_abs = os.path.abspath(split_dir)
    split_name = os.path.basename(split_dir_abs.rstrip(os.sep))
    dataset_root = os.path.dirname(split_dir_abs)
    local_cache = os.path.join(dataset_root, ".cache", f"imagefolder_index_{split_name}.pt")

    split_hash = hashlib.sha1(split_dir_abs.encode("utf-8")).hexdigest()[:16]
    home_cache = os.path.join(
        os.path.expanduser("~/.cache/m3vit/imagefolder_index"),
        split_hash,
        f"imagefolder_index_{split_name}.pt",
    )
    return [local_cache, home_cache]


def _normalize_cached_samples(samples):
    out = []
    for item in samples:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return None
        path, target = item
        out.append((str(path), int(target)))
    return out


def _validate_cached_samples(split_dir, samples):
    if not samples:
        return False
    # Cheap stale-cache guard. We only probe a few entries.
    probe_count = min(4, len(samples))
    for idx in range(probe_count):
        path = samples[idx][0]
        abs_path = path if os.path.isabs(path) else os.path.join(split_dir, path)
        if not os.path.isfile(abs_path):
            return False
    return True


def _load_imagefolder_index_from_cache(split_dir, cache_path):
    if not os.path.isfile(cache_path):
        print(f"[ImageFolderCache] miss (not found): {cache_path}")
        return None
    try:
        payload = torch.load(cache_path, map_location="cpu")
    except Exception as exc:
        print(f"[ImageFolderCache] failed to read cache at {cache_path}: {exc}")
        return None

    if not isinstance(payload, dict):
        print(f"[ImageFolderCache] miss (invalid payload): {cache_path}")
        return None
    if int(payload.get("version", -1)) != _IMAGEFOLDER_INDEX_CACHE_VERSION:
        print(
            f"[ImageFolderCache] miss (version mismatch): {cache_path} "
            f"(found={payload.get('version')}, expected={_IMAGEFOLDER_INDEX_CACHE_VERSION})"
        )
        return None

    classes = payload.get("classes")
    class_to_idx = payload.get("class_to_idx")
    samples = payload.get("samples")
    if not isinstance(classes, list) or not isinstance(class_to_idx, dict):
        print(f"[ImageFolderCache] miss (bad classes/class_to_idx): {cache_path}")
        return None
    samples = _normalize_cached_samples(samples)
    if samples is None:
        print(f"[ImageFolderCache] miss (bad samples format): {cache_path}")
        return None
    if not _validate_cached_samples(split_dir, samples):
        print(f"[ImageFolderCache] miss (stale paths): {cache_path}")
        return None

    abs_samples = []
    for path, target in samples:
        abs_path = path if os.path.isabs(path) else os.path.join(split_dir, path)
        abs_samples.append((abs_path, int(target)))

    return {
        "classes": list(classes),
        "class_to_idx": dict(class_to_idx),
        "samples": abs_samples,
    }


def _serialize_imagefolder_index_for_cache(index, split_dir):
    samples_rel = []
    for path, target in index["samples"]:
        rel_path = str(path)
        if os.path.isabs(rel_path):
            try:
                rel_path = os.path.relpath(rel_path, split_dir)
            except Exception:
                rel_path = str(path)
        if rel_path.startswith(".."):
            rel_path = str(path)
        samples_rel.append((rel_path, int(target)))

    return {
        "version": _IMAGEFOLDER_INDEX_CACHE_VERSION,
        "split_dir": os.path.abspath(split_dir),
        "classes": list(index["classes"]),
        "class_to_idx": dict(index["class_to_idx"]),
        "samples": samples_rel,
        "num_samples": len(samples_rel),
    }


def _scan_imagefolder_index(split_dir):
    split_dir = os.path.abspath(split_dir)
    classes = sorted([d.name for d in os.scandir(split_dir) if d.is_dir()])
    if not classes:
        raise RuntimeError(f"No class folders found under: {split_dir}")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    valid_exts = tuple(e.lower() for e in IMG_EXTENSIONS)

    # Count phase with progress
    t0 = time.time()
    total_candidates = 0
    total_images = 0
    class_counts = {}
    print(f"[ImageFolderCache] counting image files under {split_dir} ...")
    for cls_name in classes:
        class_dir = os.path.join(split_dir, cls_name)
        img_count = 0
        for _root, _dirs, files in os.walk(class_dir, followlinks=True):
            total_candidates += len(files)
            for fname in files:
                if fname.lower().endswith(valid_exts):
                    img_count += 1
        class_counts[cls_name] = img_count
        total_images += img_count
    if total_images == 0:
        raise RuntimeError(f"No image files found under: {split_dir}")
    print(
        f"[ImageFolderCache] count done: classes={len(classes)}, "
        f"images={total_images}, candidates={total_candidates}, "
        f"elapsed={time.time()-t0:.1f}s"
    )

    # Build samples with periodic progress + ETA
    print(f"[ImageFolderCache] indexing {total_images} images ...")
    samples = []
    processed = 0
    last_log_t = time.time()
    log_every_sec = 5.0

    for cls_name in classes:
        class_dir = os.path.join(split_dir, cls_name)
        target = class_to_idx[cls_name]
        for root, _dirs, files in os.walk(class_dir, followlinks=True):
            files.sort()
            for fname in files:
                if not fname.lower().endswith(valid_exts):
                    continue
                path = os.path.join(root, fname)
                samples.append((path, target))
                processed += 1

                now = time.time()
                if (now - last_log_t) >= log_every_sec:
                    elapsed = max(now - t0, 1e-6)
                    speed = processed / elapsed
                    remain = max(total_images - processed, 0)
                    eta_sec = int(remain / max(speed, 1e-6))
                    pct = 100.0 * processed / max(total_images, 1)
                    print(
                        f"[ImageFolderCache] indexing progress: {processed}/{total_images} "
                        f"({pct:.2f}%), speed={speed:.1f} img/s, ETA={eta_sec}s"
                    )
                    last_log_t = now

    elapsed = time.time() - t0
    print(
        f"[ImageFolderCache] indexing done: samples={len(samples)}, "
        f"elapsed={elapsed:.1f}s, avg_speed={len(samples)/max(elapsed,1e-6):.1f} img/s"
    )
    return {
        "classes": classes,
        "class_to_idx": class_to_idx,
        "samples": [(str(path), int(target)) for path, target in samples],
    }


def _load_or_build_imagefolder_index(split_dir):
    cache_paths = _imagefolder_cache_paths(split_dir)
    print("[ImageFolderCache] candidate paths:")
    for p in cache_paths:
        print(f"  - {p}")

    for cache_path in cache_paths:
        index = _load_imagefolder_index_from_cache(split_dir, cache_path)
        if index is not None:
            print(f"[ImageFolderCache] loaded index: {cache_path}")
            serialized = _serialize_imagefolder_index_for_cache(index, split_dir)
            for other_path in cache_paths:
                if other_path != cache_path and not os.path.isfile(other_path):
                    _safe_torch_save(serialized, other_path)
            return index

    print(f"[ImageFolderCache] building index by scanning: {split_dir}")
    index = _scan_imagefolder_index(split_dir)
    serialized = _serialize_imagefolder_index_for_cache(index, split_dir)
    for cache_path in cache_paths:
        _safe_torch_save(serialized, cache_path)
    return index


class IndexedImageFolder(Dataset):
    """ImageFolder-compatible dataset built from cached index."""

    def __init__(self, split_dir, index, transform=None, target_transform=None):
        self.root = split_dir
        self.transform = transform
        self.target_transform = target_transform
        self.loader = default_loader

        self.classes = list(index["classes"])
        self.class_to_idx = dict(index["class_to_idx"])
        self.samples = list(index["samples"])
        self.imgs = self.samples
        self.targets = [int(target) for _, target in self.samples]

        if len(self.samples) == 0:
            raise RuntimeError(f"IndexedImageFolder has no samples: {split_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target


def _build_imagefolder_dataset(split_dir, transform):
    if _dist_initialized():
        rank = torch.distributed.get_rank()
        if rank == 0:
            print(f"[ImageFolderCache][rank{rank}] preparing index for {split_dir}")
            index = _load_or_build_imagefolder_index(split_dir)
            print(f"[ImageFolderCache][rank{rank}] index ready, releasing other ranks")
            torch.distributed.barrier()
        else:
            print(f"[ImageFolderCache][rank{rank}] waiting for rank0 to prepare index...")
            torch.distributed.barrier()
            print(f"[ImageFolderCache][rank{rank}] loading index after barrier")
            index = _load_or_build_imagefolder_index(split_dir)
    else:
        index = _load_or_build_imagefolder_index(split_dir)
    return IndexedImageFolder(split_dir, index, transform=transform)


def build_imagenet_datasets(args):
    if _is_hf_dataset_path(getattr(args, "data_path", "")):
        return _build_hf_imagenet_datasets(args)

    train_dir = os.path.join(args.data_path, "train")
    val_dir = os.path.join(args.data_path, "val")

    _maybe_bootstrap_local_imagenet_from_hf(args, train_dir, val_dir)
    if not _has_local_imagefolder_split(train_dir):
        raise FileNotFoundError(f"ImageNet train dir not found or empty: {train_dir}")
    if not _has_local_imagefolder_split(val_dir):
        raise FileNotFoundError(f"ImageNet val dir not found or empty: {val_dir}")

    dataset_train = _build_imagefolder_dataset(
        train_dir, transform=build_imagenet_transform(True, args)
    )
    dataset_val = _build_imagefolder_dataset(
        val_dir, transform=build_imagenet_transform(False, args)
    )
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
