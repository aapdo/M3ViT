"""Training loop (DeiT-like)."""

import math

import torch

from pretrain.utils import MetricLogger


def train_one_epoch(
    model,
    criterion,
    data_loader,
    optimizer,
    device,
    epoch,
    scaler,
    mixup_fn,
    args,
    model_ema=None,
):
    model.train(True)
    metric_logger = MetricLogger(delimiter="  ")
    header = f"Epoch: [{epoch}]"

    for samples, targets in metric_logger.log_every(data_loader, args.print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast(enabled=args.amp):
            out = model(samples)
            logits = out["logits"] if isinstance(out, dict) else out
            cv_loss = out.get("cv_loss", None) if isinstance(out, dict) else None

            loss = criterion(samples, logits, targets)
            criterion_stats = getattr(criterion, "last_stats", None)
            cv = 0.0
            cv_weighted = 0.0
            if cv_loss is not None:
                cv = cv_loss.mean() if torch.is_tensor(cv_loss) else torch.tensor(float(cv_loss), device=device)
                cv_weighted = float(args.moe_cv_weight) * float(cv.detach().item())
                loss = loss + args.moe_cv_weight * cv

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            raise RuntimeError(f"Loss is {loss_value}, stopping training")

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()

        if args.clip_grad is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

        scaler.step(optimizer)
        scaler.update()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        if isinstance(criterion_stats, dict):
            metric_logger.update(**criterion_stats)
        if cv_loss is not None:
            metric_logger.update(cv_loss=float(cv.detach().item()))
            metric_logger.update(cv_loss_weighted=cv_weighted)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
