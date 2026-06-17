"""Evaluation metrics for multi-label emotion classification."""

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def multilabel_metrics(logits: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> dict:
    probs = _sigmoid(np.asarray(logits))
    preds = (probs >= threshold).astype(int)
    labels = np.asarray(labels).astype(int)
    return {
        "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
    }


def per_class_f1(logits: np.ndarray, labels: np.ndarray, label_names, threshold: float = 0.5) -> dict:
    probs = _sigmoid(np.asarray(logits))
    preds = (probs >= threshold).astype(int)
    labels = np.asarray(labels).astype(int)
    scores = f1_score(labels, preds, average=None, zero_division=0)
    return {name: float(s) for name, s in zip(label_names, scores)}


def build_compute_metrics(threshold: float = 0.5):
    """Return a ``compute_metrics`` callable compatible with the HF Trainer."""

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        return multilabel_metrics(logits, labels, threshold)

    return compute_metrics
