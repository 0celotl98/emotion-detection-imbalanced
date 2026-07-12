# Imbalance-Aware Multi-Label Emotion Detection

Fine-tuning **BERT** for **multi-label emotion classification** on
[**GoEmotions**](https://huggingface.co/datasets/go_emotions), with a
**combined loss** (weighted BCE + focal loss) designed to handle the dataset's
heavy class imbalance.

This is a small, self-contained reference implementation built with PyTorch and
the Hugging Face `transformers` / `datasets` libraries. It focuses on doing one
thing clearly: showing how imbalance-aware loss functions improve performance on
rare emotion labels.

> **Background.** I worked on imbalanced multilingual emotion detection as a
> co-author of *LATE-GIL-NLP at SemEval-2025 Task 11*
> ([ACL Anthology](https://aclanthology.org/2025.semeval-1.93/)). This repo does
> **not** contain that shared task's code or data; it is a clean, standalone demo
> of the same core idea — transformer fine-tuning with combined imbalance-aware
> losses — on the public GoEmotions dataset, so anyone can run it end to end.

## Method

GoEmotions has 28 labels (27 emotions + `neutral`) and a long-tailed
distribution: a handful of labels are common while many are rare. Two
complementary techniques address this:

- **Weighted BCE** — each class gets a `pos_weight = (# negatives / # positives)`
  computed from the training split, so rare labels contribute more to the loss.
- **Focal loss** — the `(1 - p_t) ** gamma` term down-weights easy, confident
  predictions and concentrates learning on hard examples.

The training objective combines them:

```
loss = bce_weight * weighted_BCE + (1 - bce_weight) * focal_BCE
```

Both `gamma` and `bce_weight` are exposed as command-line flags so you can
ablate each component.

## Project structure

```
.
├── train.py            # fine-tuning entry point (argparse)
├── evaluate.py         # evaluate a saved checkpoint on the test split
├── requirements.txt
└── src/
    ├── data.py         # GoEmotions loading + multi-hot preprocessing
    ├── losses.py       # CombinedLoss (weighted BCE + focal)
    ├── metrics.py      # micro/macro F1 + per-class F1
    └── trainer.py      # custom Trainer + multi-label collator
```

## Setup

```bash
git clone https://github.com/0celotl98/emotion-detection-imbalanced.git
cd emotion-detection-imbalanced
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Quick sanity check on a small subset (runs on CPU in a few minutes):

```bash
python train.py --max_train_samples 2000 --max_eval_samples 500 --epochs 1
```

Full training run (GPU recommended):

```bash
python train.py --epochs 3 --batch_size 16 --lr 2e-5 --gamma 2.0 --bce_weight 0.5
```

Evaluate a saved checkpoint:

```bash
python evaluate.py --model_dir outputs --threshold 0.5
```
## Reproducibility

The reported results use the following configuration:

* **Base checkpoint:** `bert-base-uncased`
* **Dataset:** GoEmotions, `simplified` configuration
* **Training data:** full `train` split
* **Model selection:** best checkpoint by macro-F1 on the `validation` split
* **Final evaluation:** full `test` split
* **Epochs:** `3`
* **Learning rate:** `2e-5`
* **Per-device batch size:** `16`
* **Maximum sequence length:** `128`
* **Focal-loss gamma:** `2.0`
* **Weighted-BCE contribution:** `0.5`
* **Global decision threshold:** `0.5`
* **Random seed:** `42`

```bash
python train.py \
  --model_name bert-base-uncased \
  --epochs 3 \
  --batch_size 16 \
  --lr 2e-5 \
  --max_length 128 \
  --gamma 2.0 \
  --bce_weight 0.5 \
  --threshold 0.5 \
  --seed 42
```

The training script evaluates once per epoch, reloads the checkpoint with the best validation macro-F1 and reports the final metrics on the test split.

> Exact numerical reproduction may also depend on the installed PyTorch, Transformers, CUDA and hardware versions. No original training log or trained checkpoint is currently included in this repository.

## Results

Run on the GoEmotions test split (threshold = 0.5).

### Aggregate

| Metric            | Score   |
|-------------------|---------|
| F1 (micro)        | 0.4476  |
| F1 (macro)        | 0.4100  |
| Precision (macro) | 0.3019  |
| Recall (macro)    | 0.7601  |

### Per-class F1

| Emotion        | F1    |   | Emotion        | F1    |
|----------------|-------|---|----------------|-------|
| gratitude      | 0.831 |   | annoyance      | 0.351 |
| amusement      | 0.800 |   | caring         | 0.330 |
| love           | 0.753 |   | disgust        | 0.325 |
| neutral        | 0.679 |   | confusion      | 0.320 |
| admiration     | 0.641 |   | pride          | 0.298 |
| remorse        | 0.530 |   | excitement     | 0.271 |
| curiosity      | 0.525 |   | disappointment | 0.240 |
| fear           | 0.509 |   | nervousness    | 0.218 |
| surprise       | 0.469 |   | embarrassment  | 0.213 |
| sadness        | 0.414 |   | realization    | 0.192 |
| desire         | 0.409 |   | relief         | 0.161 |
| anger          | 0.407 |   | grief          | 0.133 |
| joy            | 0.404 |   |                |       |
| optimism       | 0.400 |   |                |       |
| approval       | 0.362 |   |                |       |
| disapproval    | 0.352 |   |                |       |

**Takeaway:** high-frequency, lexically distinctive classes (`gratitude`,
`amusement`, `love`, `neutral`) are learned well, while rare, context-dependent
classes (`grief`, `relief`, `realization`) remain hard — the long-tail pattern
the weighted-BCE + focal loss is meant to mitigate. The high macro-recall
(0.76) vs. low macro-precision (0.30) shows the loss is trading precision for
coverage on rare labels.

<!-- TODO (reproducibility): base checkpoint, decision threshold,
     epochs / LR, train/val split. -->

Per-class F1 is printed at the end of training/evaluation, sorted ascending, so
you can see exactly which rare emotions benefit most from the combined loss.

## Notes & possible extensions

- **Decision threshold.** A single global `0.5` threshold is used by default.
  Tuning per-class thresholds on the validation split typically improves macro-F1.
- **Multilingual.** Swapping `--model_name bert-base-uncased` for
  `xlm-roberta-base` and a multilingual dataset extends this to non-English
  emotion detection.
- **Loss ablations.** Set `--bce_weight 1.0` for weighted BCE only or
  `--bce_weight 0.0` for focal only to compare against the combination.

## License

[MIT](LICENSE)

## Acknowledgements

GoEmotions: Demszky et al., 2020. Built with
[Hugging Face Transformers](https://github.com/huggingface/transformers) and
[Datasets](https://github.com/huggingface/datasets).
