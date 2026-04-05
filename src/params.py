from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # src/ -> project root
DATA_DIR = ROOT / "data"
SPEC_DIR = DATA_DIR / "train_spectrograms"
PROCESSED_DIR = DATA_DIR / "processed"
VOTE_COL = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
CACHE_DIR = DATA_DIR / "cache"
