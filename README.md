# fruit-freshness-cnn

**Fruit ripeness detection.** Classifying fruit as **fresh** or **rotten** from a
photograph, using an EfficientNet-B0 convolutional neural network fine-tuned in PyTorch.

Covers three fruit types — apples, bananas and oranges.

![accuracy](https://img.shields.io/badge/accuracy-97.02%25-brightgreen)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![pytorch](https://img.shields.io/badge/pytorch-2.8-ee4c2c)

---

## Results

Evaluated with 5-fold cross-validation over **1,511 source photographs**. Every photograph is tested exactly once by a model that never saw it during training.

| Metric | Value |
|---|---|
| **Accuracy** | **97.02%** (95% CI 96.04–97.77%) |
| Macro F1 | 0.9698 |
| ROC-AUC | 0.9956 |
| Fold spread | ±0.76 pp across 5 folds |

Measured against three reference points:

| Reference | Score | What it tells us |
|---|---|---|
| Always guess "rotten" | 56.59% | The floor — anything near this has learned nothing |
| Colour histogram + logistic regression | 88.48% | A deliberately simple model, using colour only |
| **EfficientNet-B0** | **97.02%** | Beats the simple baseline by 8.5 pp |
| Shuffled-label control | 47.52% | Collapses to chance — confirms no data leakage |

**Per class**

| Class | Precision | Recall | F1 | Photographs |
|---|---|---|---|---|
| fresh | 0.9513 | 0.9817 | 0.9662 | 656 |
| rotten | 0.9856 | 0.9614 | 0.9734 | 855 |

**Per fruit** — bananas 99.05%, apples 96.42%, oranges 95.33%.

<p align="center">
  <img src="results/confusion_matrix.png" width="45%" alt="Confusion matrix">
  <img src="results/accuracy_curve.png" width="45%" alt="Accuracy curves">
</p>

Full numbers are in [`results/metrics.json`](results/metrics.json).

---

## Quick start

```bash
git clone https://github.com/a19simru/fruit-freshness-cnn.git
cd fruit-freshness-cnn
pip install -r requirements.txt
```

Add the dataset under `data/` (see [Dataset](#dataset)), then:

```bash
python fruit.py
```

The script runs the full 5-fold cross-validation and writes everything it
produces into `results/` — `metrics.json`, `confusion_matrix.png`,
`accuracy_curve.png` and `loss_curve.png`. Model weights go to `checkpoints/`.

Runtime is roughly 20–25 minutes on an Apple M-series Mac. The device is picked
automatically: Apple GPU (MPS) → NVIDIA (CUDA) → CPU.

---

## Dataset

`data/` is not committed — it is ~3.8 GB. Arrange it as:

```
data/
├── train/
│   ├── freshapples/      freshbanana/      freshoranges/
│   └── rottenapples/     rottenbanana/     rottenoranges/
└── test/
    └── (same six folders)
```

This matches the "Fruits fresh and rotten for classification" dataset on Kaggle.
The script reads `train/` and `test/` together and re-splits them itself — see
below for why.

### One thing to know about this data

The folders hold 13,599 image files, but only **1,511 distinct photographs**.
Each photograph was saved nine times: the original plus eight pre-generated
variants (five rotations, a vertical flip, a translation, and a salt-and-pepper
noise version), identifiable by filename prefix.

The supplied `train/` and `test/` folders split those nine copies **at random**,
so every test photograph also appeared in training:

| Folder | Test photographs | Also in training |
|---|---|---|
| freshapples | 198 | 198 (100%) |
| rottenapples | 300 | 300 (100%) |
| rottenbanana | 259 | 259 (100%) |
| freshoranges | 175 | 175 (100%) |

Training and testing on rotations of the same photograph measures memorisation,
not generalisation, and inflates the reported accuracy. This project therefore
discards the supplied split.

---

## How it works

### Grouping

Every file is mapped back to its source photograph by stripping the augmentation
prefix from its filename, so all nine copies share one group key. Splits are made
over the **1,511 groups**, never over individual files — a photograph and all its
variants always land on the same side of the split. Assertions in the code fail
loudly if a group is ever found on both sides.

### Evaluation

- **5-fold cross-validation** over the groups. With only 1,511 photographs, a
  single hold-out test set would contain ~227 images, giving a ±3 pp confidence
  interval. Pooling predictions across five folds uses all 1,511 and halves that.
- **Validation and test use original photographs only.** Six of the nine variants
  have black corners, an artifact of rotating a rectangular image, which never
  occurs in a real photograph.
- Model selection is on **macro F1** rather than accuracy, since the classes are
  imbalanced 656 : 855.

### Training

EfficientNet-B0 pretrained on ImageNet, fine-tuned in two stages:

| Stage | Epochs | Trainable | Learning rate |
|---|---|---|---|
| 1 — warm-up | 5 | Classifier head only | 1e-3 |
| 2 — fine-tune | up to 25 | Head + last two feature blocks | 1e-5 backbone / 1e-4 head |

AdamW, cosine annealing, class-weighted cross-entropy with 0.05 label smoothing,
early stopping (patience 6) on validation macro F1. BatchNorm layers in the frozen
backbone are kept in eval mode so their running statistics do not drift.

Augmentation is applied at training time — horizontal and vertical flips, rotation
up to 75°, colour jitter, affine shift and salt-and-pepper noise — with **white
fill** rather than the default black, matching the white-background photographs.

### Verifying there is no leakage

The headline number is only meaningful if the grouping actually worked, so the
script re-runs training with the 1,511 labels randomly shuffled. There is then
nothing genuine to learn, and an honest model must fail. It scores **47.52%** —
below the 56.59% you would get by always guessing "rotten".

Had any leakage survived, the model could have memorised the shuffled labels and
scored highly. It could not.

---

## Project structure

```
fruit.py                     Training, evaluation and baselines
requirements.txt             Pinned dependencies
data/                        Dataset (not committed — see Dataset)
checkpoints/                 Per-fold model weights (not committed)
notebooks/
└── eda.ipynb                Exploratory analysis of the dataset
results/
├── metrics.json             Full results: per fold, per class, per fruit, baselines
├── confusion_matrix.png     Pooled out-of-fold confusion matrix
├── accuracy_curve.png       Accuracy, mean ± std across folds
├── loss_curve.png           Loss, mean ± std across folds
└── train.log                Raw training output (not committed)
docs/
└── Fruit_Ripeness_Detection.pptx
```

Everything the training run produces lands in `results/`; nothing is written to
the project root.

Key settings live at the top of `fruit.py`:

```python
N_FOLDS = 5
TRAIN_VARIANTS = "originals"   # or "all", to train on the 9 pre-generated variants
RUN_PERMUTATION_CONTROL = True
RUN_COLOR_BASELINE = True
```

---

## Limitations

**The images are stock product photographs, not real-world photographs.** Every
image is a single piece of fruit, studio-lit against a plain white background.
Real photographs have shadows, clutter, varied lighting and several fruits at
once. **97% is an honest figure for this dataset and an upper bound for
real-world use.**

**Errors are asymmetric.** 33 rotten photographs were classified as fresh, versus
12 the other way. For a consumer application the model is more likely to pass off
spoiled fruit than to reject good fruit.

**Only two classes.** An earlier version attempted a third "unripe" class, but
those images came from a different source: 100% JPEG at a fixed 162 px width
(search-result thumbnails), against 100% PNG at 520–840 px for fresh and rotten.
A model can separate them on compression artifacts alone, without ever looking at
the fruit, producing a high score that means nothing. Restoring the third class
would require unripe images matched in format and resolution to the rest.

---

## Requirements

Python 3.9+, and the packages in [`requirements.txt`](requirements.txt).
Runs on Apple Silicon (MPS), NVIDIA (CUDA) or CPU.
