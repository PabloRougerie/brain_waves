# Brain Activity Classification

> A deep-learning diagnostic aid for harmful brain activity patterns — developed as a full ML engineering pipeline from raw EEG spectrograms to a **language-agnostic, 0.46 MB, sub-millisecond ONNX model** ready for embedded / edge deployment.

The project uses the EEG spectrogram data from the Kaggle **[HMS — Harmful Brain Activity Classification](https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification)** dataset.

---

## Overview

The model classifies six harmful brain activity types — Seizure, LPD, GPD, LRDA, GRDA, Other — from 50-second EEG spectrogram windows. Each window is represented as four regional channels (LL, RL, LP, RP), 100 frequency bins, 25 time steps → tensor shape `(4, 100, 25)`.

The full pipeline covers:

1. **Preprocessing** — crop raw parquet spectrograms to a standardised 50-second window, save as `.npy`
2. **Training** — KL divergence against soft expert-vote targets, patient-aware cross-validation
3. **Architecture search** — four model families compared (CNN, two CNN+LSTM hybrids, Optuna-tuned CNN)
4. **Hyperparameter optimisation** — two Optuna studies (TPE sampler) over architecture and training parameters
5. **Compression** — structured magnitude pruning + post-training static quantisation
6. **Export** — ONNX export (float32 and int8) for low-latency edge inference

---

## Deliverables

| Artifact | Location | Notes |
|---|---|---|
| Best PyTorch checkpoint | `notebooks/checkpoints/final/epoch=26-val_loss=0.677.ckpt` | 13 MB (.ckpt includes optimizer state; model weights only: 4.20 MB), KL = 0.677 |
| Pruned checkpoint (40%) | `notebooks/checkpoints/pruning_0.4/epoch=06-val_loss=0.686.ckpt` | 5.1 MB, KL = 0.685 |
| ONNX float32 model | `notebooks/checkpoints/export/model_brain_pruned040.onnx` | 2.2 MB |
| ONNX int8 model | `notebooks/checkpoints/export/model_brain_pruned040_int8.onnx` | **0.46 MB, ~0.4 ms** |
| Result tables | `data/pruning_results.csv`, `data/export_results.csv`, `data/quantize_results.csv` | |
| Figures | `figs/` | Learning curves, pruning analysis, spectrogram examples |

---

## Repository layout

```
brain_waves/
│
├── src/                        # Python modules
│   ├── params.py               # Central config: paths, best model hyperparameters
│   ├── preprocess.py           # Spectrogram cropping pipeline (parquet → .npy)
│   ├── utils.py                # I/O helpers (load_spectrogram)
│   ├── dataset.py              # BrainDataset, BrainDataModule (StratifiedGroupKFold)
│   ├── models.py               # All model architectures (CNN, Hybrid, OptunaModel)
│   ├── lightning.py            # BrainLightning training wrapper (KLDiv, mixup, AdamW)
│   └── deployment.py           # Pruning, quantisation, ONNX export utilities
│
├── notebooks/                  # Notebooks — run in order
│   ├── 1_eda.ipynb             # EDA + spectrogram preprocessing (prepare_spectrograms)
│   ├── 2_dataset_objects.ipynb # Dataset / datamodule validation
│   ├── 3_model_cnn.ipynb       # [exploration] Baseline VGG-style CNN
│   ├── 4_hybrid_parallel_model.ipynb  # [exploration] CNN + 4 parallel LSTMs
│   ├── 5_hybrid_sequential_model.ipynb  # [exploration] CNN → 2-layer LSTM
│   ├── 6_dual_training.ipynb   # [exploration] Two-phase curriculum training
│   ├── 7_optuna_v1.ipynb       # [exploration] Optuna study v1 (broad architecture search)
│   ├── 8_optuna_v2.ipynb       # [exploration] Optuna study v2 (up to 300 trials)
│   ├── 8_bis_final_training.ipynb  # ★ Final training with best Optuna config
│   ├── 9_pruning.ipynb         # ★ Magnitude pruning sweep (0–40%)
│   ├── 10_quantization.ipynb   # ★ Post-training static quantisation
│   ├── 11_onnx_export.ipynb    # ★ ONNX export, quantisation, runtime benchmark
│   └── checkpoints/
│       ├── final/              # Best model weights
│       ├── pruning_0.4/        # Pruned + fine-tuned weights
│       └── export/             # ONNX artifacts
│
├── data/
│   ├── cache/                  # Cached fold splits and normalisation statistics
│   ├── train.csv               # Metadata (106 800 annotated windows)
│   ├── rating_per_patient.csv  # Annotation statistics per patient
│   ├── pruning_results.csv     # KL / size / latency across pruning ratios
│   ├── export_results.csv      # ONNX pipeline benchmark
│   └── quantize_results.csv    # PyTorch int8 quantisation results
│
├── figs/                       # Summary figures
│   ├── spec_example.png
│   ├── expert_consensus_per_patient.png
│   ├── lc_*.png                # Learning curves (one per model/experiment)
│   ├── pruning_kl.png
│   ├── pruning_size_latency.png
│   └── summary_model_optimization.png
│
├── pyproject.toml              # Dependencies (managed with uv)
└── README.md
```

> **Not tracked by git** (local only): raw EEG parquets (`data/train_eegs/`, `data/train_spectrograms/`), processed `.npy` arrays (`data/processed/`), Optuna study databases.

---

## Getting started

### Prerequisites

- Python ≥ 3.12
- [`uv`](https://docs.astral.sh/uv/) for dependency management

```bash
git clone https://github.com/<your-handle>/brain_waves.git
cd brain_waves
uv sync
```

---

### Option A — Run ONNX inference (no data required)

The int8 ONNX model and normalisation statistics are included in the repo. Only `onnxruntime` and `numpy` are required.

**Step 1 — Load a preprocessed spectrogram** (shape `(4, 100, 25)`, values in linear power scale):

```python
import json
import numpy as np
import onnxruntime as ort

# Load a preprocessed .npy spectrogram (4 channels × 100 freq bins × 25 time steps)
spec = np.load("data/processed/cropped/files/<spectrogram_id>-<subsample_id>.npy")
# spec.shape == (4, 100, 25), dtype float32
```

**Step 2 — Apply the same normalisation used during training:**

```python
# log1p compression (same as training-time transform)
spec = np.log1p(spec)

# z-score using fold-0 training statistics (global scalars, cached)
# filename encodes: 5 splits, seed=273, min_votes=1
with open("data/cache/5_at_seed_273_min_vote_1.json") as f:
    stats = json.load(f)["0"]           # fold 0
mean = stats["mean"]   # float scalar
std  = stats["std"]    # float scalar
spec = (spec - mean) / std
```

**Step 3 — Run inference:**

```python
session = ort.InferenceSession(
    "notebooks/checkpoints/export/model_brain_pruned040_int8.onnx"
)

x      = spec[np.newaxis].astype(np.float32)  # add batch dim → (1, 4, 100, 25)
logits = session.run(["class_logits"], {"spectrogram": x})[0]
proba  = np.exp(logits)                        # model outputs log-softmax

classes = ["Seizure", "LPD", "GPD", "LRDA", "GRDA", "Other"]
print(dict(zip(classes, proba[0].round(3))))
```

---

### Option B — Reproduce the full training pipeline

**Step 1 — Download the raw data from Kaggle**

```bash
# requires kaggle CLI: pip install kaggle
kaggle competitions download -c hms-harmful-brain-activity-classification
unzip hms-harmful-brain-activity-classification.zip -d data/
```

The relevant files are `data/train.csv` and `data/train_spectrograms/` (~20 GB of parquet files).

**Step 2 — Generate preprocessed spectrograms** (run inside notebook `1_eda.ipynb`, or standalone):

```python
import pandas as pd
from src.preprocess import prepare_spectrograms

metadata = pd.read_csv("data/train.csv")
prepare_spectrograms(metadata, save_dir="cropped/files")
# Output: ~4 GB of .npy files in data/processed/cropped/files/
# Runtime: ~30–60 min depending on hardware
```

**Step 3 — Run the notebooks**

Notebooks 3–8 are **exploratory** (architecture experiments, Optuna studies). You can skip them and go straight to the production path:

| Notebook | Purpose | Required? |
|---|---|---|
| `1_eda.ipynb` | EDA + preprocessing | Yes |
| `2_dataset_objects.ipynb` | Dataset validation | Recommended |
| `3_model_cnn.ipynb` … `8_optuna_v2.ipynb` | Architecture exploration | No (results already in `data/*.csv`) |
| `8_bis_final_training.ipynb` | Train the final model | Yes |
| `9_pruning.ipynb` | Pruning sweep | Yes |
| `10_quantization.ipynb` | PyTorch int8 quantisation | Yes |
| `11_onnx_export.ipynb` | ONNX export + benchmark | Yes |

Minimum reproduction path: **1 → 2 → 8_bis → 9 → 10 → 11**

---

## Tech stack

| Library | Role |
|---|---|
| PyTorch 2.11 | Model definition and training |
| PyTorch Lightning 2.6 | Training loop, callbacks, checkpointing |
| torch-pruning 1.6 | Structured magnitude pruning |
| ONNX 1.21 | Model serialisation |
| ONNX Runtime 1.24 | Optimised CPU inference, static int8 quantisation |
| Optuna 4.8 | Bayesian hyperparameter optimisation |
| scikit-learn 1.8 | StratifiedGroupKFold cross-validation |
| pandas / numpy / pyarrow | Data loading and preprocessing |
| matplotlib / seaborn | Visualisation |
| uv | Python environment and dependency management |

---

## Results summary

| Model | Val KL loss | Size (MB) | Latency (ms) |
|---|---|---|---|
| Baseline CNN (VGG-style) | 0.709 | 1.55 | — |
| Hybrid Parallel (CNN + 4 LSTMs) | 0.852 | — | — |
| Hybrid Sequential (CNN → LSTM) | 0.755 | — | — |
| **Optuna best (final training)** | **0.677** | **4.20** | **~4** |
| + Pruned 40% + fine-tuned | 0.685 | 1.67 | ~3 |
| + ONNX int8 (final artifact) | **0.687** | **0.46** | **0.40** |

The Optuna model (4.20 MB) is larger than the hardcoded baseline (1.55 MB) because the search found a wider architecture (1.1M vs 406K parameters). The compression pipeline brings it down to 0.46 MB — **~89% smaller** and **~10× faster** than the uncompressed Optuna model, with only +0.010 KL increase.
