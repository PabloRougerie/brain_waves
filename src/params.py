"""
Project-wide constants: paths, model architecture, and training hyperparameters.

All other modules import from this file via `from src.params import *`.
Path layout adapts automatically between local development and Kaggle environments.
"""

from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

KAGGLE = Path('/kaggle').exists()

if KAGGLE:
    DATA_DIR       = Path('/kaggle/input/competitions/hms-harmful-brain-activity-classification')
    SPEC_DIR       = DATA_DIR / 'train_spectrograms'
    PROCESSED_DIR  = Path('/kaggle/input/hms-processed-spectrograms')
    CACHE_DIR      = Path('/kaggle/working/cache')
    CHECKPOINTS_DIR = Path('/kaggle/working/checkpoints')
else:
    ROOT           = Path(__file__).resolve().parent.parent
    DATA_DIR       = ROOT / 'data'
    SPEC_DIR       = DATA_DIR / 'train_spectrograms'
    PROCESSED_DIR  = DATA_DIR / 'processed'
    CACHE_DIR      = DATA_DIR / 'cache'
    CHECKPOINTS_DIR = ROOT / "notebooks" / "checkpoints"


# ─────────────────────────────────────────────────────────────────────────────
# Dataset constants
# ─────────────────────────────────────────────────────────────────────────────

# Column names for the 6 expert vote counts in train.csv
VOTE_COL = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]


# ─────────────────────────────────────────────────────────────────────────────
# Best model — architecture & training hyperparameters (from Optuna v2, trial 59)
# ─────────────────────────────────────────────────────────────────────────────

BEST_MODEL_PARAMS = {
    "n_classes":   6,
    "n_channels":  4,
    "hidden_dims": [117, 52, 60, 64, 103, 226],
    "kernel_size": 5,
    "dropout":     0.085,
}

BEST_TRAINING_PARAMS = {
    "lr":          0.0028273119832751066,
    "mixup_alpha": 0.35,
    "weight_decay": 1.2e-5,
    "n_epochs":    35,
    "t_max":       35,
}


# ─────────────────────────────────────────────────────────────────────────────
# Deployment constants
# ─────────────────────────────────────────────────────────────────────────────

# Checkpoint produced by 8_bis_final_training
BEST_WEIGHTS_CHECKPOINT = "epoch=26-val_loss=0.677.ckpt"
BEST_WEIGHTS_PATH = CHECKPOINTS_DIR / "final" / BEST_WEIGHTS_CHECKPOINT

# Layers excluded from magnitude pruning (first input conv + classifier head)
PRUNER_IGNORED_LAYERS_NAMES = ["backbone[0][0].block[0]", "head"]

# Conv-BN-ReLU triplets to fuse before static quantization
FUSE_LIST = [
    ["model.backbone.0.0.block.0", "model.backbone.0.0.block.1", "model.backbone.0.0.block.2"],
    ["model.backbone.0.1.block.0", "model.backbone.0.1.block.1", "model.backbone.0.1.block.2"],
    ["model.backbone.1.0.block.0", "model.backbone.1.0.block.1", "model.backbone.1.0.block.2"],
    ["model.backbone.1.1.block.0", "model.backbone.1.1.block.1", "model.backbone.1.1.block.2"],
    ["model.backbone.2.0.block.0", "model.backbone.2.0.block.1", "model.backbone.2.0.block.2"],
    ["model.backbone.2.1.block.0", "model.backbone.2.1.block.1", "model.backbone.2.1.block.2"],
]
