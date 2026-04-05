from pathlib import Path

DATA_DIR = Path("data")
SPEC_DIR = DATA_DIR / "train_spectrograms"
PROCESSED_DIR = DATA_DIR / "processed"
VOTE_COL = ["seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote"]
