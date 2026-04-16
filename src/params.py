from pathlib import Path

KAGGLE = Path('/kaggle').exists()

if KAGGLE:
    DATA_DIR = Path('/kaggle/input/competitions/hms-harmful-brain-activity-classification')
    SPEC_DIR = DATA_DIR / 'train_spectrograms'
    PROCESSED_DIR = Path('/kaggle/input/hms-processed-spectrograms')
    CACHE_DIR = Path('/kaggle/working/cache')
else:
    ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = ROOT / 'data'
    SPEC_DIR = DATA_DIR / 'train_spectrograms'
    PROCESSED_DIR = DATA_DIR / 'processed'
    CACHE_DIR = DATA_DIR / 'cache'
    CHECKPOINTS_DIR = ROOT  / "notebooks" / "checkpoints" / "final"

VOTE_COL = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]


BEST_MODEL_PARAMS = {
        "n_classes" : 6,
        "n_channels" : 4,
        "hidden_dims" : [117, 52, 60, 64, 103, 226],
        "kernel_size" : 5,
        "dropout" : 0.085}

BEST_TRAINING_PARAMS = {
        "lr": 0.0028273119832751066,
        "mixup_alpha": 0.35,
        "weight_decay": 1.2e-5,
        "n_epochs": 35,
        "t_max": 35,
    
}
    

