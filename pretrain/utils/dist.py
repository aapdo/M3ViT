"""Distributed helpers adapted from DeiT-style training scripts."""

import builtins
import os

import torch
import torch.distributed as dist


def setup_for_distributed(is_master):
    """Disable printing when not in master process."""
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size <= 1:
        return x

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x_reduce = torch.tensor(x, dtype=torch.float32, device=device)
    dist.all_reduce(x_reduce)
    x_reduce /= world_size
    return x_reduce.item()


def init_distributed_mode(args):
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        env = os.environ
        args.rank = int(env["RANK"])
        args.world_size = int(env["WORLD_SIZE"])
        args.gpu = int(env.get("LOCAL_RANK", 0))
    else:
        args.distributed = False
        args.rank = 0
        args.world_size = 1
        args.gpu = 0
        setup_for_distributed(True)
        return

    args.distributed = True
    torch.cuda.set_device(args.gpu)
    dist_backend = "nccl"
    print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)
    dist.init_process_group(
        backend=dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )
    dist.barrier()
    setup_for_distributed(args.rank == 0)
