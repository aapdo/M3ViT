"""
Weights & Biases Logger for M3ViT Training
"""
import wandb
from typing import Dict, Any, Optional


# Global singleton instance
_wandb_logger_instance = None


def get_wandb_logger():
    """
    Get the global WandbLogger instance.

    Returns:
        WandbLogger instance or None if not initialized
    """
    global _wandb_logger_instance
    return _wandb_logger_instance


def set_wandb_logger(logger):
    """
    Set the global WandbLogger instance.

    Args:
        logger: WandbLogger instance to set as global
    """
    global _wandb_logger_instance
    _wandb_logger_instance = logger


class WandbLogger:
    """
    Wrapper class for Weights & Biases logging
    """

    def __init__(self, enabled: bool = False):
        """
        Initialize wandb logger

        Args:
            enabled: Whether to enable wandb logging
        """
        self.enabled = enabled
        self.run = None

    def init(self,
             project: str,
             entity: Optional[str] = None,
             name: Optional[str] = None,
             config: Optional[Dict[str, Any]] = None,
             resume: Optional[str] = None,
             id: Optional[str] = None):
        """
        Initialize wandb run

        Args:
            project: wandb project name
            entity: wandb entity (team name)
            name: run name
            config: hyperparameters and config dict
            resume: resume mode ("allow", "must", "never", or None)
            id: run id for resuming
        """
        if not self.enabled:
            return

        self.run = wandb.init(
            project=project,
            entity=entity,
            name=name,
            config=config,
            resume=resume,
            id=id
        )

    def log_config(self, p: Dict, args):
        """
        Log hyperparameters and configuration

        Args:
            p: config dictionary
            args: argument parser results
        """
        if not self.enabled:
            return

        config = {
            # Basic hyperparameters
            "learning_rate": p['optimizer_kwargs']['lr'],
            "weight_decay": p['optimizer_kwargs']['weight_decay'],
            "optimizer": p['optimizer'],
            "epochs": p['epochs'],
            "train_batch_size": p['trBatch'],
            "val_batch_size": p['valBatch'],

            # Architecture
            "architecture": p['backbone'],
            "model": p['model'],
            "embed_dim": p['backbone_kwargs'].get('embed_dim', None),
            "depth": p['backbone_kwargs'].get('depth', None),
            "num_heads": p['backbone_kwargs'].get('num_heads', None),

            # MoE settings
            "moe_experts": args.moe_experts,
            "moe_top_k": args.moe_top_k,
            "moe_mlp_ratio": args.moe_mlp_ratio,
            "moe_gate_type": args.moe_gate_type,
            "vmoe_noisy_std": args.vmoe_noisy_std,
            "moe_noisy_gate_loss_weight": args.moe_noisy_gate_loss_weight,
            "multi_gate": args.multi_gate,
            "gate_task_specific_dim": args.gate_task_specific_dim,

            # Dataset and tasks
            "dataset": p['train_db_name'],
            "tasks": p.TASKS.NAMES,
            "num_tasks": len(p.TASKS.NAMES),

            # Training settings
            "seed": args.seed,
            "world_size": getattr(args, 'world_size', 1),
            "distributed": args.distributed,
        }

        # Update wandb config
        if self.run is not None:
            self.run.config.update(config)

    def log_train_losses(self, losses: Dict, p: Dict):
        """
        Log training losses

        Args:
            losses: dictionary of AverageMeter objects for losses
            p: config dictionary
        """
        if not self.enabled:
            return

        metrics = {}

        # Task-specific losses
        for task in p.TASKS.NAMES:
            if task in losses:
                metrics[f"train/loss_{task}"] = losses[task].avg

        # CV loss (gating loss)
        if 'gating' in losses:
            metrics["train/cv_loss"] = losses['gating'].avg

        # Total loss
        if 'total' in losses:
            metrics["train/total_loss"] = losses['total'].avg

        # Multi-level losses (if applicable)
        if p.get('multi_level', False):
            for task in p.TASKS.NAMES:
                for level in range(1, 4):
                    level_key = f'level{level}_{task}'
                    if level_key in losses:
                        metrics[f"train/level{level}_loss_{task}"] = losses[level_key].avg

        self.log(metrics)

    def log_train_performance(self, eval_results: Dict, p: Dict):
        """
        Log training performance metrics

        Args:
            eval_results: evaluation results from performance_meter.get_score()
            p: config dictionary
        """
        if not self.enabled:
            return

        metrics = {}

        for task in p.TASKS.NAMES:
            if task not in eval_results:
                continue

            task_result = eval_results[task]

            if task in ['semseg', 'human_parts']:
                metrics[f"train/{task}_mIoU"] = task_result.get('mIoU', 0)
                if 'acc' in task_result:
                    metrics[f"train/{task}_acc"] = task_result['acc']

            elif task == 'depth':
                metrics[f"train/{task}_rmse"] = task_result.get('rmse', 0)
                if 'abs_err' in task_result:
                    metrics[f"train/{task}_abs_err"] = task_result['abs_err']

            elif task == 'normals':
                metrics[f"train/{task}_mean"] = task_result.get('mean', 0)
                if 'median' in task_result:
                    metrics[f"train/{task}_median"] = task_result['median']
                for angle in ['11.25', '22.5', '30']:
                    if angle in task_result:
                        metrics[f"train/{task}_{angle}"] = task_result[angle]

            elif task == 'edge':
                metrics[f"train/{task}_odsF"] = task_result.get('odsF', 0)

            elif task == 'sal':
                if 'maxF' in task_result:
                    metrics[f"train/{task}_maxF"] = task_result['maxF']
                if 'mIoU' in task_result:
                    metrics[f"train/{task}_mIoU"] = task_result['mIoU']

        self.log(metrics)

    def log_val_performance(self, curr_result: Dict, p: Dict):
        """
        Log validation performance metrics

        Args:
            curr_result: evaluation results from eval_all_results()
            p: config dictionary
        """
        if not self.enabled:
            return

        metrics = {}

        for task in p.TASKS.NAMES:
            if task not in curr_result:
                continue

            task_result = curr_result[task]

            if task in ['semseg', 'human_parts']:
                metrics[f"val/{task}_mIoU"] = task_result.get('mIoU', 0)

            elif task == 'depth':
                metrics[f"val/{task}_rmse"] = task_result.get('rmse', 0)

            elif task == 'normals':
                metrics[f"val/{task}_mean"] = task_result.get('mean', 0)

            elif task == 'edge':
                metrics[f"val/{task}_odsF"] = task_result.get('odsF', 0)

            elif task == 'sal':
                if 'maxF' in task_result:
                    metrics[f"val/{task}_maxF"] = task_result['maxF']
                if 'mIoU' in task_result:
                    metrics[f"val/{task}_mIoU"] = task_result['mIoU']

        # Multi-task performance
        if 'multi_task_performance' in curr_result:
            metrics["val/multi_task_performance"] = curr_result['multi_task_performance']

        self.log(metrics)

    def log_best_results(self, best_result: Dict, p: Dict):
        """
        Log best results achieved so far

        Args:
            best_result: dictionary of best results
            p: config dictionary
        """
        if not self.enabled:
            return

        metrics = {}

        for task in p.TASKS.NAMES:
            if task in best_result and isinstance(best_result[task], dict):
                for metric_name, metric_value in best_result[task].items():
                    if isinstance(metric_value, (int, float)):
                        metrics[f"best/{task}_{metric_name}"] = metric_value

        if 'multi_task_performance' in best_result:
            metrics["best/multi_task_performance"] = best_result['multi_task_performance']

        self.log(metrics)

    def log_learning_rate(self, lr: float):
        """
        Log current learning rate

        Args:
            lr: learning rate value
        """
        if not self.enabled:
            return

        self.log({"train/lr": lr})

    def log_epoch(self, epoch: int):
        """
        Log current epoch

        Args:
            epoch: epoch number
        """
        if not self.enabled:
            return

        self.log({"epoch": epoch})

    def log_moe_stats(self, stats: Dict):
        """
        Log MoE statistics (reuse ratio, aggregation ratio, etc.)

        Args:
            stats: dictionary containing MoE statistics
                - reuse_ratio: ratio of reusable tokens
                - aggregation_ratio: ratio of aggregated tokens
                - shared_gate_ratio: ratio of tokens using shared gate
                - total_tokens: total number of tokens processed
                - moe_blocks: number of MoE blocks
        """
        if not self.enabled:
            return

        metrics = {}

        if 'reuse_ratio' in stats:
            metrics["moe/reuse_ratio"] = stats['reuse_ratio']
        if 'aggregation_ratio' in stats:
            metrics["moe/aggregation_ratio"] = stats['aggregation_ratio']
        if 'shared_gate_ratio' in stats:
            metrics["moe/shared_gate_ratio"] = stats['shared_gate_ratio']
        if 'total_tokens' in stats:
            metrics["moe/total_tokens"] = stats['total_tokens']
        if 'reusable_tokens' in stats:
            metrics["moe/reusable_tokens"] = stats['reusable_tokens']
        if 'aggregated_tokens' in stats:
            metrics["moe/aggregated_tokens"] = stats['aggregated_tokens']
        if 'shared_gate_tokens' in stats:
            metrics["moe/shared_gate_tokens"] = stats['shared_gate_tokens']
        if 'moe_blocks' in stats:
            metrics["moe/moe_blocks"] = stats['moe_blocks']

        self.log(metrics)

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """
        Log metrics to wandb

        Args:
            metrics: dictionary of metrics to log
            step: optional step number
        """
        if not self.enabled or self.run is None:
            return

        if step is not None:
            self.run.log(metrics, step=step)
        else:
            self.run.log(metrics)

    def finish(self):
        """
        Finish wandb run
        """
        if not self.enabled or self.run is None:
            return

        self.run.finish()

    def watch(self, model, log: str = "gradients", log_freq: int = 100):
        """
        Watch model gradients and parameters

        Args:
            model: PyTorch model to watch
            log: what to log ("gradients", "parameters", "all", or None)
            log_freq: logging frequency
        """
        if not self.enabled or self.run is None:
            return

        wandb.watch(model, log=log, log_freq=log_freq)
