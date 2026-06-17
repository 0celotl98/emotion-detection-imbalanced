"""Loss functions for imbalanced multi-label classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedLoss(nn.Module):
    """Weighted BCE + focal loss for imbalanced multi-label classification.

        total = bce_weight * weighted_BCE + (1 - bce_weight) * focal_BCE

    - **Weighted BCE** uses a per-class ``pos_weight`` (neg/pos ratio) so that
      rare labels contribute more to the loss.
    - **Focal loss** applies the ``(1 - p_t) ** gamma`` modulation, down-weighting
      easy, confident predictions and focusing learning on hard examples.

    Combining both is a simple, robust recipe for skewed label distributions and
    is the core idea behind the imbalance handling explored in the SemEval-2025
    Task 11 work this repository is based on.
    """

    def __init__(self, pos_weight=None, gamma: float = 2.0, bce_weight: float = 0.5):
        super().__init__()
        self.gamma = gamma
        self.bce_weight = bce_weight
        if pos_weight is not None and not torch.is_tensor(pos_weight):
            pos_weight = torch.tensor(pos_weight, dtype=torch.float)
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        pos_weight = self.pos_weight.to(logits.device) if self.pos_weight is not None else None

        weighted_bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pos_weight, reduction="mean"
        )

        bce_none = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal = ((1.0 - p_t) ** self.gamma) * bce_none
        focal = focal.mean()

        return self.bce_weight * weighted_bce + (1.0 - self.bce_weight) * focal
