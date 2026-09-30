from __future__ import annotations

import torch
import torch.nn.functional as F


def smooth_focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 0.1,
    gamma: float = 2.0,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    probs = torch.sigmoid(logits).clamp(1e-6, 1.0 - 1e-6)
    labels_smooth = (1.0 - epsilon) * labels + epsilon / 2.0
    focal_factor = (1.0 - torch.abs(probs - labels_smooth)).pow(gamma)

    positive_term = labels_smooth * torch.log(probs)
    negative_term = (1.0 - labels_smooth) * torch.log(1.0 - probs)
    if pos_weight is not None:
        positive_term = positive_term * pos_weight.view(1)
    loss = -(focal_factor * (positive_term + negative_term))
    return loss.mean()


def binary_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    hard_labels: torch.Tensor,
    temperature: float,
    alpha: float,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    teacher_prob = torch.sigmoid(teacher_logits / temperature).clamp(1e-6, 1.0 - 1e-6)
    student_prob = torch.sigmoid(student_logits / temperature).clamp(1e-6, 1.0 - 1e-6)

    kl = teacher_prob * (torch.log(teacher_prob) - torch.log(student_prob))
    kl += (1.0 - teacher_prob) * (torch.log(1.0 - teacher_prob) - torch.log(1.0 - student_prob))
    kl = kl.mean() * (temperature ** 2)

    bce = F.binary_cross_entropy_with_logits(student_logits, hard_labels, pos_weight=pos_weight)
    return alpha * kl + (1.0 - alpha) * bce
