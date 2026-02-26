"""Download ImageNet-1k from Hugging Face and materialize ImageFolder layout.

Example:
  python pretrain/datasets/download_imagenet.py \
    --output-dir /path/to/imagenet
"""

import argparse
import os
import shutil

from PIL import Image


HF_DATASET_ID_DEFAULT = "ILSVRC/imagenet-1k"


def parse_args():
    parser = argparse.ArgumentParser(description="Download ImageNet-1k to ImageFolder layout")
    parser.add_argument("--output-dir", required=True, type=str, help="Target root dir containing train/ and val/")
    parser.add_argument("--dataset-id", default=HF_DATASET_ID_DEFAULT, type=str)
    parser.add_argument("--train-split", default="train", type=str)
    parser.add_argument("--val-split", default="validation", type=str)
    parser.add_argument("--cache-dir", default="", type=str, help="HF cache dir (default: <output-dir>/.hf_cache)")
    parser.add_argument("--token", default="", type=str, help="HF token (optional; env vars also supported)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing split folders")
    parser.add_argument("--progress-every", default=10000, type=int)
    return parser.parse_args()


def resolve_hf_token(cli_token=""):
    token = str(cli_token or "").strip()
    if token:
        return token
    for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        value = str(os.environ.get(key, "") or "").strip()
        if value:
            return value
    return None


def load_hf_split(dataset_id, split, token=None, cache_dir=None):
    try:
        from datasets import load_dataset
    except Exception as exc:
        raise ImportError(
            "This script requires `datasets` and `huggingface_hub`. "
            "Install: pip install -U \"datasets[vision]\" huggingface_hub"
        ) from exc

    kwargs = {"split": split}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir

    if token:
        try:
            return load_dataset(dataset_id, token=token, **kwargs)
        except TypeError:
            return load_dataset(dataset_id, use_auth_token=token, **kwargs)
    return load_dataset(dataset_id, **kwargs)


def normalize_class_name(name):
    text = str(name).strip().replace("/", "_").replace("\\", "_")
    return text if text else "unknown"


def infer_class_names(hf_dataset):
    features = getattr(hf_dataset, "features", None) or {}
    label_feature = features.get("label", None)
    class_names = getattr(label_feature, "names", None)
    if class_names:
        return [normalize_class_name(name) for name in class_names]

    num_classes = getattr(label_feature, "num_classes", None)
    if num_classes is not None:
        return [str(i) for i in range(int(num_classes))]

    return [str(i) for i in range(1000)]


def has_local_imagefolder_split(split_dir):
    if not os.path.isdir(split_dir):
        return False
    for class_dir in os.scandir(split_dir):
        if not class_dir.is_dir():
            continue
        for entry in os.scandir(class_dir.path):
            if entry.is_file():
                return True
    return False


def infer_image_extension(image_field):
    default_ext = ".jpg"
    if isinstance(image_field, dict):
        image_path = str(image_field.get("path", "") or "")
        ext = os.path.splitext(image_path)[1].lower()
        if ext in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            return ext
    return default_ext


def ensure_pil_rgb(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, dict) and image.get("bytes", None) is not None:
        # Use bytes path in save_image_field to preserve original encoding.
        raise TypeError("Raw bytes image should be handled by save_image_field")
    if hasattr(image, "ndim"):
        return Image.fromarray(image).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image)}")


def save_image_field(image_field, image_path):
    if isinstance(image_field, dict) and image_field.get("bytes", None) is not None:
        with open(image_path, "wb") as f:
            f.write(image_field["bytes"])
        return
    image = ensure_pil_rgb(image_field)
    image.save(image_path, format="JPEG", quality=95)


def save_split_to_imagefolder(hf_dataset, split_dir, class_names, progress_every=10000):
    os.makedirs(split_dir, exist_ok=True)
    total = len(hf_dataset)
    for idx in range(total):
        sample = hf_dataset[idx]
        label = int(sample["label"])
        class_name = class_names[label] if 0 <= label < len(class_names) else str(label)
        class_dir = os.path.join(split_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

        image_ext = infer_image_extension(sample["image"])
        image_path = os.path.join(class_dir, f"{idx:08d}{image_ext}")
        save_image_field(sample["image"], image_path)

        if progress_every > 0 and (idx + 1) % progress_every == 0:
            print(f"[download] {split_dir}: {idx + 1}/{total}")
    print(f"[done] {split_dir}: {total} images")


def maybe_prepare_split(split_dir, overwrite=False):
    if overwrite and os.path.isdir(split_dir):
        shutil.rmtree(split_dir)
    return has_local_imagefolder_split(split_dir)


def main():
    args = parse_args()
    output_dir = os.path.expandvars(os.path.expanduser(args.output_dir))
    dataset_id = str(args.dataset_id).strip() or HF_DATASET_ID_DEFAULT
    cache_dir = str(args.cache_dir or "").strip()
    if not cache_dir:
        cache_dir = os.path.join(output_dir, ".hf_cache")
    cache_dir = os.path.expandvars(os.path.expanduser(cache_dir))
    token = resolve_hf_token(args.token)

    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "val")

    os.makedirs(output_dir, exist_ok=True)

    train_exists = maybe_prepare_split(train_dir, overwrite=args.overwrite)
    val_exists = maybe_prepare_split(val_dir, overwrite=args.overwrite)
    if train_exists and val_exists and not args.overwrite:
        print(f"[skip] train/ and val/ already exist under: {output_dir}")
        return 0

    print(f"[info] dataset_id={dataset_id}")
    print(f"[info] output_dir={output_dir}")
    print(f"[info] cache_dir={cache_dir}")
    print(f"[info] train_split={args.train_split}, val_split={args.val_split}")

    class_names = None
    if not train_exists or args.overwrite:
        train_hf = load_hf_split(dataset_id, split=args.train_split, token=token, cache_dir=cache_dir)
        class_names = infer_class_names(train_hf)
        save_split_to_imagefolder(train_hf, train_dir, class_names, progress_every=args.progress_every)
    else:
        print(f"[skip] existing split: {train_dir}")

    if not val_exists or args.overwrite:
        val_hf = load_hf_split(dataset_id, split=args.val_split, token=token, cache_dir=cache_dir)
        if class_names is None:
            class_names = infer_class_names(val_hf)
        save_split_to_imagefolder(val_hf, val_dir, class_names, progress_every=args.progress_every)
    else:
        print(f"[skip] existing split: {val_dir}")

    print("[ok] ImageNet dataset is ready in ImageFolder layout.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
