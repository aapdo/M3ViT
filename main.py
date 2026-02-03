#
# Authors: Simon Vandenhende
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

import argparse
import cv2
import os
import numpy as np
import sys
import torch

from utils.config import create_config
from utils.common_config import get_train_dataset, get_transformations,\
                                get_val_dataset, get_train_dataloader, get_val_dataloader,\
                                get_optimizer, get_model, adjust_learning_rate,\
                                get_criterion
from utils.logger import Logger
from utils.wandb_logger import WandbLogger, set_wandb_logger
from train.train_utils import train_vanilla
from evaluation.evaluate_utils import eval_model, validate_results, save_model_predictions,\
                                    eval_all_results,validate_results_v2
from termcolor import colored
import time
from thop import clever_format
from thop import profile
import wandb
# Parser
parser = argparse.ArgumentParser(description='Vanilla Training')
parser.add_argument('--config_env',
                    help='Config file for the environment')
parser.add_argument('--config_exp',
                    help='Config file for the experiment')
parser.add_argument('--flops', action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
parser.add_argument('--time', action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
parser.add_argument('--use_cv_loss', action='store_true', default=False,
        help='whether model returns cv_loss (default: False for evaluation)')
parser.add_argument('--save_dir', type=str, default=None, help='custom output directory')
# Wandb arguments
parser.add_argument('--use_wandb', action='store_true', help='use wandb for logging')
parser.add_argument('--wandb_project', type=str, default='m3vit-training', help='wandb project name')
parser.add_argument('--wandb_entity', type=str, default=None, help='wandb entity (team name)')
parser.add_argument('--wandb_name', type=str, default=None, help='wandb run name')
args = parser.parse_args()

def main():
    # Retrieve config file
    cv2.setNumThreads(0)
    p = create_config(args.config_env, args.config_exp, args=args)
    sys.stdout = Logger(os.path.join(p['output_dir'], 'log_file.txt'))
    print(colored(p, 'red'))

    # Get model
    print(colored('Retrieve model', 'blue'))
    model = get_model(p)
    # print('model_structure',model)
    if not args.flops:
        model = torch.nn.DataParallel(model)
    model = model.cuda()

    # Get criterion
    print(colored('Get loss', 'blue'))
    criterion = get_criterion(p)
    criterion.cuda()
    print(criterion)

    # CUDNN
    print(colored('Set CuDNN benchmark', 'blue')) 
    torch.backends.cudnn.benchmark = True

    # Optimizer
    print(colored('Retrieve optimizer', 'blue'))
    optimizer = get_optimizer(p, model)
    print(optimizer)

    # Dataset
    print(colored('Retrieve dataset', 'blue'))
    
    # Transforms 
    train_transforms, val_transforms = get_transformations(p)
    train_dataset = get_train_dataset(p, train_transforms)
    val_dataset = get_val_dataset(p, val_transforms)
    true_val_dataset = get_val_dataset(p, None) # True validation dataset without reshape 
    train_dataloader = get_train_dataloader(p, train_dataset)
    val_dataloader = get_val_dataloader(p, val_dataset)
    print('Train samples %d - Val samples %d' %(len(train_dataset), len(val_dataset)))
    print('Train transformations:')
    print(train_transforms)
    print('Val transformations:')
    print(val_transforms)
    
    if args.flops:
        model.eval()
        for ii, sample in enumerate(val_dataloader):
        # sample = val_dataloader[0]
            inputs, meta = sample['image'].cuda(non_blocking=True), sample['meta']
            assert inputs.size(0)==1
            flops, params = profile(model.backbone, inputs=(inputs, ),)
            flops, params = clever_format([flops, params], "%.3f")
            # if p['backbone']=='VisionTransformer':
            #     flops = int(flops)+int(model.backbone.flops())
            print(flops,params)
            exit()
    # Resume from checkpoint
    if os.path.exists(p['checkpoint']):
        print(colored('Restart from checkpoint {}'.format(p['checkpoint']), 'blue'))
        checkpoint = torch.load(p['checkpoint'], map_location='cpu')
        optimizer.load_state_dict(checkpoint['optimizer'])
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
        best_result = checkpoint['best_result']

    else:
        print(colored('No checkpoint file at {}'.format(p['checkpoint']), 'blue'))
        start_epoch = 0

        #### don't do it during debug
        # save_model_predictions(p, val_dataloader, model)
        # best_result = eval_all_results(p)
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

    # Initialize wandb logger
    wandb_logger = WandbLogger(enabled=args.use_wandb)
    if wandb_logger.enabled:
        # Auto-generate run name in KST (UTC+9) if not explicitly provided
        wandb_run_name = args.wandb_name
        if wandb_run_name is None:
            from datetime import datetime, timezone, timedelta
            kst = timezone(timedelta(hours=9))
            wandb_run_name = datetime.now(kst).strftime("%Y%m%d_%H%M")

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
        eval_train = train_vanilla(p, train_dataloader, model, criterion, optimizer, epoch, wandb_logger=wandb_logger)

        # Evaluate
            # Check if need to perform eval first
        eval_interval = p.get('eval_interval', 1)
        if 'eval_final_10_epochs_only' in p.keys() and p['eval_final_10_epochs_only']:
            # Eval every eval_interval epochs, and always during final 10 epochs
            if epoch + 1 > p['epochs'] - 10:
                eval_bool = True
            elif (epoch + 1) % eval_interval == 0:
                eval_bool = True
            else:
                eval_bool = False
        else:
            eval_bool = (epoch + 1) % eval_interval == 0

        # Perform evaluation
        if eval_bool:
            print('Evaluate ...')
            save_model_predictions(p, val_dataloader, model, args)
            curr_result = eval_all_results(p)

            # Log validation results to wandb
            wandb_logger.log_val_performance(curr_result, p)

            # improves, best_result = validate_results_v2(p, curr_result, best_result)
            improves, best_result = validate_results(p, curr_result, best_result)

            # Log best results to wandb
            wandb_logger.log_best_results(best_result, p)

            if improves:
                print('Save new best model')
                torch.save(model.state_dict(), p['best_model'])

            # Checkpoint
            print('Checkpoint ...')
            torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(),
                        'epoch': epoch + 1, 'best_result': best_result}, p['checkpoint'])

    # Evaluate best model at the end
    print(colored('Evaluating best model at the end', 'blue'))
    model.load_state_dict(torch.load(p['checkpoint'])['model'])
    save_model_predictions(p, val_dataloader, model, args)
    eval_stats = eval_all_results(p)

    # Finish wandb run
    wandb_logger.finish()

if __name__ == "__main__":
    main()
