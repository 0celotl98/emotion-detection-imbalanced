"""Fine-tune BERT on GoEmotions with a combined imbalance-aware loss.

Example
-------
Quick sanity check on a small subset (runs on CPU in a few minutes):

    python train.py --max_train_samples 2000 --max_eval_samples 500 --epochs 1

Full run:

    python train.py --epochs 3 --batch_size 16 --lr 2e-5
"""

import argparse

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
)

from src.data import compute_pos_weight, load_goemotions
from src.losses import CombinedLoss
from src.metrics import build_compute_metrics, per_class_f1
from src.trainer import MultilabelCollator, MultilabelTrainer


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune BERT on GoEmotions (multi-label).")
    p.add_argument("--model_name", default="bert-base-uncased")
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--gamma", type=float, default=2.0, help="Focal loss focusing parameter.")
    p.add_argument("--bce_weight", type=float, default=0.5,
                   help="Weight of the weighted-BCE term vs. the focal term (0-1).")
    p.add_argument("--threshold", type=float, default=0.5, help="Sigmoid decision threshold.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_eval_samples", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenized, label_names = load_goemotions(tokenizer, args.max_length)
    num_labels = len(label_names)

    if args.max_train_samples:
        tokenized["train"] = tokenized["train"].select(range(args.max_train_samples))
    if args.max_eval_samples:
        tokenized["validation"] = tokenized["validation"].select(range(args.max_eval_samples))

    pos_weight = compute_pos_weight(tokenized["train"], num_labels)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification",
    )

    loss_fn = CombinedLoss(pos_weight=pos_weight, gamma=args.gamma, bce_weight=args.bce_weight)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=50,
        seed=args.seed,
        report_to="none",
    )

    trainer = MultilabelTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=MultilabelCollator(tokenizer),
        compute_metrics=build_compute_metrics(args.threshold),
        loss_fn=loss_fn,
    )

    trainer.train()

    print("\n=== Test set evaluation ===")
    preds = trainer.predict(tokenized["test"])
    for key, value in preds.metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")

    print("\n=== Per-class F1 (sorted) ===")
    pc = per_class_f1(preds.predictions, preds.label_ids, label_names, args.threshold)
    for name, score in sorted(pc.items(), key=lambda kv: kv[1]):
        print(f"{name:15s} {score:.3f}")

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nModel saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
