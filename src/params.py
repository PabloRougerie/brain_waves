from pathlib import Path

KAGGLE = Path('/kaggle').exists()

if KAGGLE:
    DATA_DIR = Path('/kaggle/input/competitions/hms-harmful-brain-activity-classification')
    SPEC_DIR = DATA_DIR / 'train_spectrograms'
    PROCESSED_DIR = Path('/kaggle/working/processed')
    CACHE_DIR = Path('/kaggle/working/cache')
else:
    ROOT = Path(__file__).resolve().parent.parent
    DATA_DIR = ROOT / 'data'
    SPEC_DIR = DATA_DIR / 'train_spectrograms'
    PROCESSED_DIR = DATA_DIR / 'processed'
    CACHE_DIR = DATA_DIR / 'cache'

VOTE_COL = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
