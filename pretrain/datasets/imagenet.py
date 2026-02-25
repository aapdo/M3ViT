"""ImageNet datasets and dataloaders."""

import os

import torch
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler
from torchvision import datasets

from .samplers import RASampler
from .transforms import build_imagenet_transform


def build_imagenet_datasets(args):
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
