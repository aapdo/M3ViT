"""Losses following DeiT's distillation interface."""

import torch
import torch.nn as nn
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy


class DistillationLoss(nn.Module):
    """
    Distillation loss compatible with DeiT-style training loop.

    Inputs:
    - inputs: raw image tensor
    - outputs: logits tensor or tuple(logits, logits_dist)
    - labels: hard labels or soft labels from mixup
    """

    def __init__(self, base_criterion, teacher_model, distillation_type, alpha, tau):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau
        # Latest scalar loss terms for logging.
        self.last_stats = {}

    def forward(self, inputs, outputs, labels):
        if not isinstance(outputs, tuple):
            outputs = (outputs, None)

        outputs, outputs_kd = outputs
        base_loss = self.base_criterion(outputs, labels)
        base_loss_val = float(base_loss.detach().item())

        if self.distillation_type == "none":
            self.last_stats = {
                "loss_base": base_loss_val,
                "loss_distill": 0.0,
                "loss_no_cv": base_loss_val,
            }
            return base_loss

        if outputs_kd is None:
            raise ValueError(
                "Distillation is enabled but model did not return distillation logits. "
                "Either disable distillation or add a distillation head."
            )

        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)

        if self.distillation_type == "soft":
            t = self.tau
            distillation_loss = nn.functional.kl_div(
                nn.functional.log_softmax(outputs_kd / t, dim=1),
                nn.functional.log_softmax(teacher_outputs / t, dim=1),
                reduction="sum",
                log_target=True,
            ) * (t * t) / outputs_kd.numel()
        elif self.distillation_type == "hard":
            distillation_loss = nn.functional.cross_entropy(
                outputs_kd, teacher_outputs.argmax(dim=1)
            )
        else:
            raise ValueError(f"Unknown distillation type: {self.distillation_type}")

        total_loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        self.last_stats = {
            "loss_base": base_loss_val,
            "loss_distill": float(distillation_loss.detach().item()),
            "loss_no_cv": float(total_loss.detach().item()),
        }
        return total_loss


def build_criterion(args, teacher_model=None):
    if args.mixup > 0.0 or args.cutmix > 0.0 or args.cutmix_minmax is not None:
        base_criterion = SoftTargetCrossEntropy()
    elif args.smoothing > 0.0:
        base_criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        base_criterion = nn.CrossEntropyLoss()

    return DistillationLoss(
        base_criterion=base_criterion,
        teacher_model=teacher_model,
        distillation_type=args.distillation_type,
        alpha=args.distillation_alpha,
        tau=args.distillation_tau,
    )
