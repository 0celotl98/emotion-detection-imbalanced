"""GoEmotions loading and multi-hot preprocessing.

GoEmotions (Demszky et al., 2020) is a public, multi-label emotion dataset of
Reddit comments with 27 emotion categories plus ``neutral`` (28 labels total).
The label distribution is heavily imbalanced, which makes it a good public
stand-in for the imbalance challenges seen in multilingual emotion detection.
"""

import numpy as np
import torch
from datasets import load_dataset


def load_goemotions(tokenizer, max_length: int = 128):
    """Load and tokenize GoEmotions, returning multi-hot float labels.

    Returns
    -------
    tokenized : DatasetDict with ``input_ids``, ``attention_mask`` and ``labels``
    label_names : list[str] of the 28 label names
    """
    raw = load_dataset("go_emotions", "simplified")
    label_names = raw["train"].features["labels"].feature.names
    num_labels = len(label_names)

    def encode(batch):
        enc = tokenizer(batch["text"], truncation=True, max_length=max_length)
        multi_hot = []
        for label_ids in batch["labels"]:
            vec = [0.0] * num_labels
            for idx in label_ids:
                vec[idx] = 1.0
            multi_hot.append(vec)
        enc["labels"] = multi_hot
        return enc

    # Remove raw text columns; our multi-hot "labels" overwrites the original.
    remove_cols = [c for c in raw["train"].column_names if c != "labels"]
    tokenized = raw.map(encode, batched=True, remove_columns=remove_cols)
    return tokenized, label_names


def compute_pos_weight(dataset_split, num_labels: int) -> torch.Tensor:
    """Per-class pos_weight = (# negatives / # positives) from the training split."""
    labels = np.asarray(dataset_split["labels"], dtype=np.float32)
    positives = labels.sum(axis=0)
    negatives = labels.shape[0] - positives
    pos_weight = negatives / np.clip(positives, 1.0, None)
    return torch.tensor(pos_weight, dtype=torch.float)
