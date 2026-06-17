"""Evaluate a trained checkpoint on the GoEmotions test split.

Example
-------
    python evaluate.py --model_dir outputs --threshold 0.5
"""

import argparse

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
)

from src.data import load_goemotions
from src.losses import CombinedLoss
from src.metrics import build_compute_metrics, per_class_f1
from src.trainer import MultilabelCollator, MultilabelTrainer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True, help="Path to a trained checkpoint.")
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--batch_size", type=int, default=32)
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    tokenized, label_names = load_goemotions(tokenizer, args.max_length)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)

    training_args = TrainingArguments(
        output_dir="tmp_eval",
        per_device_eval_batch_size=args.batch_size,
        report_to="none",
    )
    trainer = MultilabelTrainer(
        model=model,
        args=training_args,
        data_collator=MultilabelCollator(tokenizer),
        compute_metrics=build_compute_metrics(args.threshold),
        loss_fn=CombinedLoss(),
    )

    preds = trainer.predict(tokenized["test"])

    print("=== Test metrics ===")
    for key, value in preds.metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")

    print("\n=== Per-class F1 (sorted) ===")
    pc = per_class_f1(preds.predictions, preds.label_ids, label_names, args.threshold)
    for name, score in sorted(pc.items(), key=lambda kv: kv[1]):
        print(f"{name:15s} {score:.3f}")


if __name__ == "__main__":
    main()
