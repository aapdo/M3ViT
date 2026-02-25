"""Evaluation loop."""

import torch
import torch.nn as nn
from timm.utils import accuracy

from pretrain.utils import MetricLogger


@torch.no_grad()
def evaluate(data_loader, model, device, args):
    criterion = nn.CrossEntropyLoss()
    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    model.eval()
    loss_sum = 0.0
    correct1_sum = 0.0
    correct5_sum = 0.0
    sample_count = 0

    for images, target in metric_logger.log_every(data_loader, args.print_freq, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=args.amp):
            out = model(images)
            logits = out["logits"] if isinstance(out, dict) else out
            loss = criterion(logits, target)

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))

        batch_size = images.shape[0]
        loss_sum += float(loss.item() * batch_size)
        correct1_sum += float(acc1.item() * batch_size / 100.0)
        correct5_sum += float(acc5.item() * batch_size / 100.0)
        sample_count += int(batch_size)

        metric_logger.update(loss=loss.item())
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        stats = torch.tensor(
            [loss_sum, correct1_sum, correct5_sum, float(sample_count)],
            device=device,
            dtype=torch.float64,
        )
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
        loss_sum, correct1_sum, correct5_sum, sample_count = stats.tolist()

    sample_count = max(sample_count, 1.0)
    acc1_global = 100.0 * correct1_sum / sample_count
    acc5_global = 100.0 * correct5_sum / sample_count
    loss_global = loss_sum / sample_count

    print(
        "* Acc@1 {top1:.3f} Acc@5 {top5:.3f} loss {loss:.4f}".format(
            top1=acc1_global,
            top5=acc5_global,
            loss=loss_global,
        )
    )

    return {"loss": loss_global, "acc1": acc1_global, "acc5": acc5_global}
