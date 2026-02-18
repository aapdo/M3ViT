#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import argparse
import cv2
import os
import numpy as np
import sys
import torch
from torch.nn.parallel import DistributedDataParallel
from models import ckpt_custom_moe_layer, ckpt_vision_transformer_moe
from models.gate_funs import ckpt_noisy_gate_vmoe
from utils.config import create_config
from utils.common_config import get_train_dataset, get_transformations,\
                                get_val_dataset, get_train_dataloader, get_val_dataloader,\
                                get_optimizer, get_model, adjust_learning_rate,\
                                get_criterion
from utils.logger import Logger
from utils.wandb_logger import WandbLogger, set_wandb_logger
from train.train_utils import train_vanilla,train_vanilla_distributed
from evaluation.evaluate_utils import eval_model, validate_results, save_model_predictions,\
                                    eval_all_results,validate_results_v2
from termcolor import colored

import torch.distributed as dist
import subprocess
import random
from utils.custom_collate import collate_mil
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils.common_config import build_train_dataloader,build_val_dataloader
from utils.moe_utils import sync_weights,save_checkpoint
import time
import fmoe
from thop import clever_format
from thop import profile
from utils.tracing import setup_forward_hooks, wrap_datasets_for_forward_hook, handle_forward_hook_data, \
                            patch_and_log_initializations, restore_original_initializations
import wandb

def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def str2bool(v):
    """
    Input:
        v - string
    output:
        True/False
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# Parser
parser = argparse.ArgumentParser(description='Vanilla Training')
parser.add_argument('--config_env',
                    help='Config file for the environment')
parser.add_argument('--config_exp',
                    help='Config file for the experiment')
parser.add_argument("--gpus",
        type=int,
        default=1,
        help="number of gpus to use " "(only applicable to non-distributed training)",
    )
parser.add_argument("--launcher",
        choices=["pytorch", "slurm"],
        default="pytorch",
        help="job launcher",
    )
parser.add_argument("--local_rank", type=int, default=-1)
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
parser.add_argument('--moe_data_distributed', action='store_true', help='if employ moe data distributed')
parser.add_argument('--moe_experts', default=16, type=int, help='moe experts number')
parser.add_argument('--moe_mlp_ratio', default=None, type=int, help='moe experts mlp ratio')
parser.add_argument('--moe_top_k', default=None, type=int, help='top k expert number')
parser.add_argument('--trBatch', default=None, type=int, help='train batch size')
parser.add_argument('--valBatch', default=None, type=int, help='validation batch size')
parser.add_argument('--moe_gate_arch', default="", type=str)
parser.add_argument('--moe_gate_type', default="noisy", type=str)
parser.add_argument('--vmoe_noisy_std', default=1, type=float)
parser.add_argument('--backbone_random_init',default=False, type=str2bool, help='whether randomly initialize backbone')
parser.add_argument('--pretrained', default='', type=str, help='path to moe pretrained checkpoint')
parser.add_argument('--moe_noisy_gate_loss_weight', default=0.01, type=float)


parser.add_argument('--pos_emb_from_pretrained', default=False, type=str, help='pos embedding load from pretrain weights')
parser.add_argument('--lr', default=None, type=float)
# parser.add_argument('--weight_decay', default=None, type=float)

parser.add_argument('--one_by_one',default=False, type=str2bool, help='whether train task one after another')
parser.add_argument('--task_one_hot',default=False, type=str2bool, help='whether use Task-conditioned MoE')
parser.add_argument('--multi_gate',default=False, type=str2bool, help='whether use Multi gate MoE')

parser.add_argument('--eval', action='store_true',help='if only do evaluation')
parser.add_argument('--flops', action='store_true',
        help='flops calculation')
parser.add_argument('--ckp',type=str,default=None,help='checkpoint path during evaluation')
parser.add_argument('--save_dir', type=str, default=None)
parser.add_argument('--gate_task_specific_dim', default=-1, type=int, help='gate task specific dims')

parser.add_argument('--regu_experts_fromtask',default=False, type=str2bool, help='whether use task id to guide expert selection')
parser.add_argument('--num_experts_pertask',default=-1, type=int)

parser.add_argument('--gate_input_ahead',default=False, type=str2bool, help='whether make gate input different from token')

parser.add_argument('--regu_sem',default=False, type=str2bool, help='whether use segmentation map to guide expert selection')
parser.add_argument('--semregu_loss_weight', default=0.01, type=float)

parser.add_argument('--sem_force',default=False, type=str2bool, help='whether use segmentation map to guide expert selection')
parser.add_argument('--warmup_epochs',default=5, type=int, help='whether need warmup train expert')

parser.add_argument('--epochs',default=None, type=int, help='number of train epochs')

parser.add_argument('--regu_subimage',default=False, type=str2bool, help='whether use subimage regulation for expert selection')
parser.add_argument('--subimageregu_weight', default=0.01, type=float)

parser.add_argument('--multi_level',default=None, type=str2bool, help='whether use multi level loss')

parser.add_argument('--opt', default=None, type=str, metavar='OPTIMIZER', help='Optimizer (default: "adamw"')
parser.add_argument('--weight_decay', type=float, default=0.0001,help='weight decay (default: 0.05)')

parser.add_argument('--expert_prune',default=False, type=str2bool, help='whether use expert pruning')

parser.add_argument('--tam_level0',default=None, type=str2bool, help='use tamlevel0 to boost training')
parser.add_argument('--tam_level1',default=None, type=str2bool, help='use tamlevel1 to boost training')
parser.add_argument('--tam_level2',default=None, type=str2bool, help='use tamlevel2 to boost training')

parser.add_argument('--resume', default='', help='resume from checkpoint')
parser.add_argument('--time', action='store_true', help='if wanna get inference time')
parser.add_argument('--forward_hook', default=False, type=str2bool, help='whether to enable forward hooks for layer output logging')

# Wandb arguments
parser.add_argument('--use_wandb', action='store_true', help='use wandb for logging')
parser.add_argument('--wandb_project', type=str, default='m3vit-training', help='wandb project name')
parser.add_argument('--wandb_entity', type=str, default=None, help='wandb entity (team name)')
parser.add_argument('--wandb_name', type=str, default=None, help='wandb run name')
parser.add_argument('--use_cv_loss', default=True, type=str2bool, help='whether model returns cv_loss (default: True for training)')
parser.add_argument('--use_checkpointing', default=True, type=str2bool, help='use gradient checkpointing version of VisionTransformer_moe (default: True)')
parser.add_argument('--use_weight_scaling', default=False, type=str2bool, help='whether to scale weights when initializing MoE experts from DeiT MLP')
parser.add_argument('--use_virtual_group_initialization', default=False, type=str2bool, help='whether to use virtual group initialization for MoE experts (split DeiT MLP into groups)')

args = parser.parse_args()

if args.task_one_hot:
    args.one_by_one = True

if "LOCAL_RANK" not in os.environ:
    os.environ["LOCAL_RANK"] = str(args.local_rank)
    # print(os.environ["LOCAL_RANK"])

print('os.environ["LOCAL_RANK"]: ',os.environ["LOCAL_RANK"],args.local_rank)


def main():
    cv2.setNumThreads(0)
    p = create_config(args.config_env, args.config_exp, local_rank=args.local_rank, args=args)
    args.num_tasks = len(p.TASKS.NAMES)
    p['multi_gate'] = args.multi_gate
    if args.tam_level0 is not None:
        p['model_kwargs']['tam_level0']=args.tam_level0
    if args.tam_level1 is not None:
        p['model_kwargs']['tam_level1']=args.tam_level1
    if args.tam_level2 is not None:
        p['model_kwargs']['tam_level2']=args.tam_level2
    if args.lr is not None:
        p['optimizer_kwargs']['lr'] = args.lr
    if args.opt is not None:
        p['optimizer'] = args.opt
    if args.weight_decay is not None:
        p['optimizer_kwargs']['weight_decay'] = args.weight_decay
    if args.epochs is not None:
        p['epochs'] = args.epochs
    if args.backbone_random_init is not None:
        p['backbone_kwargs']['random_init']=args.backbone_random_init
    if args.moe_mlp_ratio is not None:
        p['backbone_kwargs']['moe_mlp_ratio']=args.moe_mlp_ratio
    p['backbone_kwargs']['use_weight_scaling'] = args.use_weight_scaling
    p['backbone_kwargs']['use_virtual_group_initialization'] = args.use_virtual_group_initialization
    if args.moe_top_k is not None:
        p['backbone_kwargs']['moe_top_k']=args.moe_top_k
    if args.trBatch is not None:
        p['trBatch'] = args.trBatch
    if args.valBatch is not None:
        p['valBatch'] = args.valBatch
    if args.multi_level is not None:
        p['multi_level'] = args.multi_level
    
    if int(args.local_rank) < 0:
        args.local_rank = int(os.environ["LOCAL_RANK"])

    args.distributed = False
    if args.local_rank >=0:
        args.distributed = True
        print('os.environ["WORLD_SIZE"]: ', os.environ["WORLD_SIZE"])
        print('args.local_rank',args.local_rank)
        args.world_size = int(os.environ["WORLD_SIZE"])
    if args.local_rank >=0:
        logger = Logger(os.path.join(p['output_dir'], 'log_file.txt'),local_rank=args.local_rank)
        sys.stdout = logger
        sys.stderr = logger  # Also redirect stderr to capture errors
    if args.distributed:
        if args.launcher == "pytorch":
            torch.cuda.set_device(args.local_rank)
            dist.init_process_group(backend="nccl", init_method="env://")
            torch.distributed.barrier()
            p['local_rank'] = args.local_rank
        elif args.launcher == "slurm":
            proc_id = int(os.environ["SLURM_PROCID"])
            ntasks = int(os.environ["SLURM_NTASKS"])
            node_list = os.environ["SLURM_NODELIST"]
            num_gpus = torch.cuda.device_count()
            p['gpus'] = num_gpus
            torch.cuda.set_device(proc_id % num_gpus)
            addr = subprocess.getoutput(
                f"scontrol show hostname {node_list} | head -n1")
            # specify master port
            port = None
            if port is not None:
                os.environ["MASTER_PORT"] = str(port)
            elif "MASTER_PORT" in os.environ:
                pass  # use MASTER_PORT in the environment variable
            else:
                # 29500 is torch.distributed default port
                os.environ["MASTER_PORT"] = "29501"
            # use MASTER_ADDR in the environment variable if it already exists
            if "MASTER_ADDR" not in os.environ:
                os.environ["MASTER_ADDR"] = addr
            os.environ["WORLD_SIZE"] = str(ntasks)
            os.environ["LOCAL_RANK"] = str(proc_id % num_gpus)
            os.environ["RANK"] = str(proc_id)

            dist.init_process_group(backend="nccl")
            p['local_rank'] = int(os.environ["LOCAL_RANK"])

        p['gpus'] = dist.get_world_size()
    else:
        p['local_rank'] = args.local_rank 
    # CUDNN
    print(colored('Set CuDNN benchmark', 'blue')) 
    torch.backends.cudnn.benchmark = True

    # NOTE: currently all ranks use the same seed. For per-GPU randomness,
    # consider using args.seed + dist.get_rank() (dist is already initialized above).
    if args.seed is not None:
        print(f'Set random seed to {args.seed}, deterministic: '
                    f'{args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)

    print(colored(p, 'red'))
    print("Distributed training: {}".format(args.distributed))
    print(f"torch.backends.cudnn.benchmark: {torch.backends.cudnn.benchmark}")
    print(str(args))
    if args.distributed:
        args.rank = torch.distributed.get_rank()
    
    print(colored('Retrieve model', 'blue'))
    args.moe_use_gate = (args.moe_gate_arch != "")

    # Monkey patch to log all class initializations
    try:
        from models import vit_up_head, token_custom_moe_layer, token_vision_transformer_moe, token_vit_up_head, models as models_module
        from models.gate_funs import noisy_gate, token_noisy_gate_vmoe

        if args.use_checkpointing:
            vit_moe_module = ckpt_vision_transformer_moe
            moe_layer_module = ckpt_custom_moe_layer
            gate_vmoe_module = ckpt_noisy_gate_vmoe
        else:
            from models import origin_vision_transformer_moe
            from models import origin_custom_moe_layer
            from models.gate_funs import origin_noisy_gate_vmoe
            vit_moe_module = origin_vision_transformer_moe
            moe_layer_module = origin_custom_moe_layer
            gate_vmoe_module = origin_noisy_gate_vmoe

        modules_to_patch = [vit_moe_module, moe_layer_module, gate_vmoe_module, noisy_gate, token_noisy_gate_vmoe, vit_up_head, models_module, token_custom_moe_layer, token_vision_transformer_moe, token_vit_up_head]
        original_inits = patch_and_log_initializations(modules_to_patch, args)
    except Exception as e:
        print(f"Warning: Could not patch some modules for initialization logging: {e}")

    model = get_model(p,args)

    # Restore original __init__ methods
    if 'original_inits' in locals(): # Check if patching actually happened
        restore_original_initializations(modules_to_patch, original_inits)

    if not torch.cuda.is_available():
        raise NotImplementedError()
        log.info('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.local_rank is not None:
            torch.cuda.set_device(args.local_rank)
            model.cuda(args.local_rank)
            if (p['backbone'] == 'VisionTransformer_moe' or p['backbone'] == 'Token_VisionTransformer_moe') and (not args.moe_data_distributed):
                print('Use fast moe distributed learning==================>>')
                model = fmoe.DistributedGroupedDataParallel(model, device_ids=[args.local_rank],find_unused_parameters=True,)
                sync_weights(model, except_key_words=["mlp.experts.h4toh", "mlp.experts.htoh4"])
            else:
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],find_unused_parameters=True)
        else:
            model.cuda()
            if (p['backbone'] == 'VisionTransformer_moe' or p['backbone'] == 'Token_VisionTransformer_moe') and (not args.moe_data_distributed):
                model = fmoe.DistributedGroupedDataParallel(model)
            else:
                model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.local_rank is not None:
        model.cuda()
    else:
        raise NotImplementedError()

    # Get criterion
    print(colored('Get loss', 'blue'))
    criterion = get_criterion(p)
    criterion.cuda(args.local_rank)
    print(criterion)

    # Add forward hooks to print layer outputs (only if enabled)
    hook_handles = []
    if args.forward_hook:
        hook_handles, args.batch_counter = setup_forward_hooks(model, args, p['output_dir'], criterion)

    # Optimizer
    print(colored('Retrieve optimizer', 'blue'))
    optimizer = get_optimizer(p, model, args)
    print(optimizer)

    # Dataset
    print(colored('Retrieve dataset', 'blue'))
    # Transforms
    train_transforms, val_transforms = get_transformations(p)
    train_dataset = get_train_dataset(p, train_transforms)
    val_dataset = get_val_dataset(p, val_transforms)
    true_val_dataset = get_val_dataset(p, None) # True validation dataset without reshape

    if args.forward_hook:
        train_dataset, val_dataset = wrap_datasets_for_forward_hook(train_dataset, val_dataset)

    # Disable shuffle when forward_hook is enabled for reproducibility
    train_shuffle = not args.forward_hook
    train_dataloader = build_train_dataloader(
        train_dataset, p['trBatch'], p['nworkers'], dist=args.distributed, shuffle=train_shuffle)
    val_dataloader = build_val_dataloader(
        val_dataset, p['valBatch'], p['nworkers'], dist=args.distributed)

    print('Train samples %d - Val samples %d' %(len(train_dataset), len(val_dataset)))
    print('Train transformations:')
    print(train_transforms)
    print('Val transformations:')
    print(val_transforms)

    if args.flops:
        for ii, sample in enumerate(val_dataloader):
            inputs, meta = sample['image'].cuda(non_blocking=True), sample['meta']
            assert inputs.size(0)==1
            flops, params = profile(model, inputs=(inputs, ),)
            flops, params = clever_format([flops, params], "%.3f")
            print(flops,params)
            exit()

    if args.eval:
        if os.path.isdir(args.ckp):
            print("=> loading checkpoint '{}'".format(args.ckp))
            checkpoint = torch.load(os.path.join(args.ckp, "0.pth".format(torch.distributed.get_rank())),
                                    map_location="cpu")
            len_save = len([f for f in os.listdir(args.ckp) if "pth" in f])
            assert len_save % torch.distributed.get_world_size() == 0
            response_cnt = [i for i in range(
                torch.distributed.get_rank() * (len_save // torch.distributed.get_world_size()),
                (torch.distributed.get_rank() + 1) * (len_save // torch.distributed.get_world_size()))]
            # merge all ckpts
            for cnt, cnt_model in enumerate(response_cnt):
                if cnt_model != 0:
                    checkpoint_specific = torch.load(os.path.join(args.ckp, "{}.pth".format(cnt_model)),
                                                    map_location="cpu")
                    if cnt != 0:
                        for key, item in checkpoint_specific["state_dict"].items():
                            checkpoint["state_dict"][key] = torch.cat([checkpoint["state_dict"][key], item],
                                                                    dim=0)
                    else:
                        checkpoint["state_dict"].update(checkpoint_specific["state_dict"])
                moe_dir_read = True
        else:
            print("=> loading checkpoint '{}'".format(args.ckp))
            checkpoint = torch.load(args.ckp, map_location='cpu')
        state_dict = checkpoint['state_dict']
        # model = cvt_state_dict_(state_dict, model,args, linear_keyword, moe_dir_read)
        msg = model.load_state_dict(state_dict, strict=False)
        print('=================model unmatched keys:================',msg)
        save_model_predictions(p, val_dataloader, model, args)
        if args.distributed:
            torch.distributed.barrier()
        eval_stats = eval_all_results(p)
        exit()

    if args.resume:
        if os.path.isdir(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(os.path.join(args.resume, "0.pth".format(torch.distributed.get_rank())),
                                    map_location="cpu")
            len_save = len([f for f in os.listdir(args.resume) if "pth" in f])
            assert len_save % torch.distributed.get_world_size() == 0
            response_cnt = [i for i in range(
                torch.distributed.get_rank() * (len_save // torch.distributed.get_world_size()),
                (torch.distributed.get_rank() + 1) * (len_save // torch.distributed.get_world_size()))]
            # merge all ckpts
            for cnt, cnt_model in enumerate(response_cnt):
                if cnt_model != 0:
                    checkpoint_specific = torch.load(os.path.join(args.resume, "{}.pth".format(cnt_model)),
                                                    map_location="cpu")
                    if cnt != 0:
                        for key, item in checkpoint_specific["state_dict"].items():
                            checkpoint["state_dict"][key] = torch.cat([checkpoint["state_dict"][key], item],
                                                                    dim=0)
                    else:
                        checkpoint["state_dict"].update(checkpoint_specific["state_dict"])
                moe_dir_read = True
        else:
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
        state_dict = checkpoint['state_dict']
        # model = cvt_state_dict_(state_dict, model,args, linear_keyword, moe_dir_read)
        msg = model.load_state_dict(state_dict, strict=False)
        print('=================model unmatched keys:================',msg)

        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            for cnt, cnt_model in enumerate(response_cnt):
                print("=> loading checkpoint optimizer")
                if cnt_model != 0:
                    optimizer.load_state_dict(checkpoint_specific['optimizer'])
                else:
                    optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch']
        best_result = checkpoint['best_result']
    else:
        print(colored('No checkpoint file at {}'.format(p['checkpoint']), 'blue'))
        start_epoch = 0
        if args.distributed:
            torch.distributed.barrier()
        # Initialize best_result based on task configuration
        best_result = {'multi_task_performance': -200}
        if len(p.TASKS.NAMES) == 1:
            task = p.TASKS.NAMES[0]
            if task == 'semseg' or task == 'human_parts' or task == 'sal':
                best_result[task] = {'mIoU': 0.0}
            elif task == 'depth':
                best_result[task] = {'rmse': float('inf')}
            elif task == 'normals':
                best_result[task] = {'mean': float('inf')}
            elif task == 'edge':
                best_result[task] = {'odsF': 0.0}  
    
    # Initialize wandb logger (only on rank 0)
    wandb_logger = WandbLogger(enabled=(args.use_wandb and args.local_rank == 0))
    if wandb_logger.enabled:
        # Auto-generate run name in KST (UTC+9) if not explicitly provided
        from datetime import datetime, timezone, timedelta
        kst = timezone(timedelta(hours=9))
        timestamp = datetime.now(kst).strftime("%Y%m%d_%H%M")

        if args.wandb_name is None:
            wandb_run_name = timestamp
        else:
            wandb_run_name = f"{args.wandb_name}_{timestamp}"

        wandb_logger.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=wandb_run_name,
            config=None  # Will be set by log_config
        )
        wandb_logger.log_config(p, args)

        # Save config files to wandb if they exist
        if args.config_env and os.path.exists(args.config_env):
            wandb.save(args.config_env)
        if args.config_exp and os.path.exists(args.config_exp):
            wandb.save(args.config_exp)

    # Set as global instance
    set_wandb_logger(wandb_logger)

    # Main loop
    print(colored('Starting main loop', 'blue'))

    # Initial evaluation before training
    if start_epoch == 0:
        print(colored('Initial evaluation before training', 'blue'))
        print('Evaluate ...')
        # Temporarily disable forward hooks during evaluation
        if args.forward_hook and hasattr(args, 'batch_counter'):
            args.batch_counter['enabled'] = False

        save_model_predictions(p, val_dataloader, model, args)
        if args.distributed:
            torch.distributed.barrier()
        curr_result = eval_all_results(p)

        # Re-enable forward hooks after evaluation
        if args.forward_hook and hasattr(args, 'batch_counter'):
            args.batch_counter['enabled'] = True

        print(colored('Initial evaluation complete', 'blue'))
        if args.distributed:
            torch.distributed.barrier()

    for epoch in range(start_epoch, p['epochs']):
        print(colored('Epoch %d/%d' %(epoch+1, p['epochs']), 'yellow'))
        print(colored('-'*10, 'yellow'))

        # Adjust lr
        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))

        # Log learning rate to wandb
        wandb_logger.log_learning_rate(lr)

        # Train
        print('Train ...')
        # eval_train = train_vanilla_distributed(args, p, train_dataloader, model, criterion, optimizer, epoch)
        eval_train = train_vanilla_distributed(args, p, train_dataloader, model, criterion, optimizer, epoch, wandb_logger=wandb_logger)
        

        # Evaluate
        # Check if need to perform eval first
        if 'eval_final_10_epochs_only' in p.keys() and p['eval_final_10_epochs_only']: # To speed up -> Avoid eval every epoch, and only test during final 10 epochs.
            if epoch + 1 > p['epochs']-10:
                # Always evaluate during final 10 epochs
                eval_bool = True
            else:
                # Before final 10 epochs, evaluate according to eval_interval
                eval_interval = p.get('eval_interval', 1)  # Default to every epoch if not specified
                eval_bool = ((epoch + 1) % eval_interval == 0)
        else:
            eval_bool = True

        # Perform evaluation
        if eval_bool:
            print('Evaluate ...')
            # Temporarily disable forward hooks during evaluation
            if args.forward_hook and hasattr(args, 'batch_counter'):
                args.batch_counter['enabled'] = False

            save_model_predictions(p, val_dataloader, model, args)
            if args.distributed:
                torch.distributed.barrier()
            curr_result = eval_all_results(p)

            # Re-enable forward hooks after evaluation
            if args.forward_hook and hasattr(args, 'batch_counter'):
                args.batch_counter['enabled'] = True

            # Log validation results to wandb
            wandb_logger.log_val_performance(curr_result, p)

            # improves, best_result = validate_results_v2(p, curr_result, best_result)
            improves, best_result = validate_results(p, curr_result, best_result)

            # Log best results to wandb
            wandb_logger.log_best_results(best_result, p)

            print('Checkpoint ...')

            save_state_dict = model.state_dict()

            moe_save = (p['backbone'] == 'VisionTransformer_moe' or p['backbone'] == 'Token_VisionTransformer_moe') and (not args.moe_data_distributed)
            save_checkpoint({
                'epoch': epoch + 1,
                'backbone': p['backbone'],
                'state_dict': save_state_dict,
                'best_result': best_result,
                'optimizer' : optimizer.state_dict(),
                }, improves, p, moe_save=moe_save)
        if args.distributed:
            torch.distributed.barrier()

    torch.cuda.empty_cache()

    # Disable forward hooks for final evaluation
    if args.forward_hook and hasattr(args, 'batch_counter'):
        args.batch_counter['enabled'] = False

    # Evaluate best model at the end
    if (p['backbone'] == 'VisionTransformer_moe' or p['backbone'] == 'Token_VisionTransformer_moe') and (not args.moe_data_distributed):
        # state_dict = read_specific_group_experts(checkpoint['state_dict'], args.local_rank, args.moe_experts)
        checkpoint_specific = torch.load(os.path.join(p['best_model'], "{}.pth".format(torch.distributed.get_rank())), map_location="cpu")
        checkpoint = torch.load(os.path.join(p['best_model'], "0.pth".format(torch.distributed.get_rank())), map_location="cpu")
        checkpoint["state_dict"].update(checkpoint_specific["state_dict"])
        state_dict = checkpoint["state_dict"]
    else:
        # if args.local_rank==0:
        print(colored('Evaluating best model at the end', 'blue'))
        state_dict = torch.load(p['best_model'])['state_dict']
    if args.distributed:
        torch.distributed.barrier()
    model.load_state_dict(state_dict)
    save_model_predictions(p, val_dataloader, model, args)
    if args.distributed:
        torch.distributed.barrier()
    eval_stats = eval_all_results(p)

    # Finish wandb run
    wandb_logger.finish()


def sanity_check(state_dict, pretrained_weights, linear_keyword):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # only ignore linear layer
        if '%s.weight' % linear_keyword in k or '%s.bias' % linear_keyword in k:
            continue

        # name in pretrained model
        k_pre = 'module.base_encoder.' + k[len('module.'):] \
            if k.startswith('module.') else 'module.base_encoder.' + k

        assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")

if __name__ == "__main__":
    main()
