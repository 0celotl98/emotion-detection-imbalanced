"""Custom HF Trainer that plugs in the combined imbalance-aware loss."""

from dataclasses import dataclass

import torch
from transformers import PreTrainedTokenizerBase, Trainer


@dataclass
class MultilabelCollator:
    """Pads token inputs and stacks multi-hot float labels into a batch."""

    tokenizer: PreTrainedTokenizerBase

    def __call__(self, features):
        labels = torch.tensor([f["labels"] for f in features], dtype=torch.float)
        inputs = [{k: f[k] for k in f if k != "labels"} for f in features]
        batch = self.tokenizer.pad(inputs, padding=True, return_tensors="pt")
        batch["labels"] = labels
        return batch


class MultilabelTrainer(Trainer):
    """Trainer subclass that uses an external loss function on the logits."""

    def __init__(self, *args, loss_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = self.loss_fn(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss
