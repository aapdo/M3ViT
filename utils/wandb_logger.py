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

            # MoE settings (optional - only for MoE models)
            "moe_experts": getattr(args, 'moe_experts', None),
            "moe_top_k": getattr(args, 'moe_top_k', None),
            "moe_mlp_ratio": getattr(args, 'moe_mlp_ratio', None),
            "moe_gate_type": getattr(args, 'moe_gate_type', None),
            "vmoe_noisy_std": getattr(args, 'vmoe_noisy_std', None),
            "moe_noisy_gate_loss_weight": getattr(args, 'moe_noisy_gate_loss_weight', None),
            "multi_gate": getattr(args, 'multi_gate', None),
            "gate_task_specific_dim": getattr(args, 'gate_task_specific_dim', None),

            # Dataset and tasks
            "dataset": p['train_db_name'],
            "tasks": p.TASKS.NAMES,
            "num_tasks": len(p.TASKS.NAMES),

            # Training settings
            "seed": getattr(args, 'seed', None),
            "world_size": getattr(args, 'world_size', 1),
            "distributed": getattr(args, 'distributed', False),
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

        # Semregu loss
        if 'semregu' in losses:
            metrics["train/semregu_loss"] = losses['semregu'].avg

        # Regu subimage loss
        if 'regu_subimage' in losses:
            metrics["train/regu_subimage_loss"] = losses['regu_subimage'].avg

        # Total loss
        if 'total' in losses:
            metrics["train/total_loss"] = losses['total'].avg

        # TAM (Task Attention Module) level losses
        for task in p.TASKS.NAMES:
            for level in range(3):  # tam_level0, tam_level1, tam_level2
                tam_key = f'tam_level{level}_{task}'
                if tam_key in losses:
                    metrics[f"train/tam_level{level}_loss_{task}"] = losses[tam_key].avg

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
        Log MoE statistics from TokenVisionTransformerMoE.forward() stats dict.

        Args:
            stats: dictionary containing MoE statistics produced by the model:
                - shared_position_ratio   : ratio of positions with ≥1 shared task
                - shared_tasktoken_ratio  : ratio of task-tokens routed through shared gate
                - reuse_ratio             : ratio of dispatched task-tokens reused from cache
                - compute_ratio           : ratio of dispatched task-tokens actually computed
                - computed_tokens         : absolute count of computed task-tokens
                - reused_tokens           : absolute count of reused task-tokens
                - total_positions         : B*N summed across MoE blocks
                - moe_blocks              : number of MoE blocks
                - analysis (optional):
                    - gate_entropy
                    - top1_prob_mean
                    - expert_load_hist
                    - dead_expert_ratio
                    - shared_bits_flip_rate
                    - partial_split_ratio
        """
        if not self.enabled:
            return

        metrics = {}

        # Routing ratios (position-scale)
        if 'shared_position_ratio' in stats:
            metrics["moe/shared_position_ratio"] = stats['shared_position_ratio']
        if 'shared_tasktoken_ratio' in stats:
            metrics["moe/shared_tasktoken_ratio"] = stats['shared_tasktoken_ratio']

        # Compute/reuse ratios (task-token-scale)
        if 'reuse_ratio' in stats:
            metrics["moe/reuse_ratio"] = stats['reuse_ratio']
        if 'compute_ratio' in stats:
            metrics["moe/compute_ratio"] = stats['compute_ratio']

        # Absolute counts
        if 'computed_tokens' in stats:
            metrics["moe/computed_tokens"] = stats['computed_tokens']
        if 'reused_tokens' in stats:
            metrics["moe/reused_tokens"] = stats['reused_tokens']
        if 'total_positions' in stats:
            metrics["moe/total_positions"] = stats['total_positions']
        if 'moe_blocks' in stats:
            metrics["moe/moe_blocks"] = stats['moe_blocks']

        analysis = stats.get("analysis", None)
        if isinstance(analysis, dict):
            if 'gate_entropy' in analysis:
                metrics["analysis/gate_entropy"] = analysis['gate_entropy']
            if 'top1_prob_mean' in analysis:
                metrics["analysis/top1_prob_mean"] = analysis['top1_prob_mean']
            if 'dead_expert_ratio' in analysis:
                metrics["analysis/dead_expert_ratio"] = analysis['dead_expert_ratio']
            if 'expert_load_cv' in analysis:
                metrics["analysis/expert_load_cv"] = analysis['expert_load_cv']
            if 'clean_logit_std' in analysis:
                metrics["analysis/clean_logit_std"] = analysis['clean_logit_std']
            if 'moe_out_norm_ratio' in analysis:
                metrics["analysis/moe_out_norm_ratio"] = analysis['moe_out_norm_ratio']
            if 'active_vs_dense_flops_ratio' in analysis:
                metrics["analysis/active_vs_dense_flops_ratio"] = analysis['active_vs_dense_flops_ratio']
            if 'expert_hidden_dim' in analysis:
                metrics["analysis/expert_hidden_dim"] = analysis['expert_hidden_dim']
            if 'shared_bits_flip_rate' in analysis:
                metrics["analysis/shared_bits_flip_rate"] = analysis['shared_bits_flip_rate']
            if 'partial_split_ratio' in analysis:
                metrics["analysis/partial_split_ratio"] = analysis['partial_split_ratio']
            if 'expert_load_hist' in analysis and isinstance(analysis['expert_load_hist'], (list, tuple)):
                for i, v in enumerate(analysis['expert_load_hist']):
                    metrics[f"analysis/expert_load_hist/e{i}"] = v

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
