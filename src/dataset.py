"""
PyTorch Dataset and Lightning DataModule for EEG spectrogram classification.

Splits are produced by StratifiedGroupKFold (grouped by patient_id) and cached
to disk so that mean/std statistics are computed only once per configuration.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, Subset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from pytorch_lightning import LightningDataModule

from src.params import VOTE_COL, CACHE_DIR
from src.utils import load_spectrogram


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class BrainDataset(Dataset):
    """PyTorch Dataset for EEG spectrograms.

    Loads pre-processed .npy spectrograms, applies log1p transform and
    z-score normalisation, and returns (spectrogram_tensor, vote_distribution)
    pairs suitable for KL divergence training.
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        spec_dir: Path,
        mean: float,
        std: float,
        augment: bool = False,
    ):
        """
        Args:
            metadata: Metadata DataFrame (train.csv rows for this split).
            spec_dir: Directory containing the pre-processed .npy files.
            mean:     Per-fold training-set mean (after log1p transform).
            std:      Per-fold training-set std  (after log1p transform).
            augment:  If True, adds small Gaussian noise to the spectrogram.
        """
        super().__init__()
        self.metadata = metadata
        self.spec_dir = spec_dir
        self.mean     = mean
        self.std      = std
        self.augment  = augment

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int):
        """Load, transform, and return a (spectrogram, vote_distribution) pair.

        Returns:
            spec:  Float32 tensor of shape (4, 100, 25).
            votes: Float32 tensor of shape (6,) summing to 1.0.
        """
        spec = load_spectrogram(base_path=self.spec_dir, df=self.metadata, idx=idx)

        # log(1+x) to reduce dynamic range, then z-score with train-set statistics
        spec = np.log1p(spec)
        spec = (spec - self.mean) / self.std

        if self.augment:
            spec = spec + np.random.normal(0, 0.05, spec.shape)

        spec = torch.tensor(spec, dtype=torch.float32)

        # normalize raw vote counts to a probability distribution
        votes = self.metadata.iloc[idx][VOTE_COL].values.astype(int)
        votes = votes / votes.sum()
        votes = torch.from_numpy(votes).float()

        return spec, votes

    def xy_masking(
        self,
        spec: np.ndarray,
        num_masks_x: int = 1,
        mask_size_x: int = 10,
        num_masks_y: int = 1,
        mask_size_y: int = 30,
    ) -> np.ndarray:
        """Apply SpecAugment-style frequency and time masking to a spectrogram.

        Args:
            spec:         Array of shape (channels, freq, time).
            num_masks_x:  Number of frequency-band masks to apply.
            mask_size_x:  Width of each frequency mask (in frequency bins).
            num_masks_y:  Number of time-window masks to apply.
            mask_size_y:  Width of each time mask (in time steps).

        Returns:
            Masked copy of `spec` with selected bands set to 0.
        """
        spec = spec.copy()
        _, n_freq, n_time = spec.shape

        for _ in range(num_masks_x):
            f_start = np.random.randint(0, n_freq - mask_size_x)
            spec[:, f_start:f_start + mask_size_x, :] = 0

        for _ in range(num_masks_y):
            t_start = np.random.randint(0, n_time - mask_size_y)
            spec[:, :, t_start:t_start + mask_size_y] = 0

        return spec


# ─────────────────────────────────────────────────────────────────────────────
# Lightning DataModule
# ─────────────────────────────────────────────────────────────────────────────

class BrainDataModule(LightningDataModule):
    """
    LightningDataModule for BrainDataset.

    Handles StratifiedGroupKFold splitting (grouped by patient_id to prevent
    patient leakage), streaming mean/std computation on the training fold, and
    caching of split indices and normalisation statistics to disk.

    Cache files are stored under `CACHE_DIR` and keyed by (n_split, seed, min_votes).
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        spec_dir: Path,
        batch_size: int = 32,
        num_workers: int = 4,
        seed: int = 273,
        n_split: int = 5,
        n_fold: int = 0,
        min_votes: int = 1,
        verbose: bool = True,
    ):
        """
        Args:
            metadata:    Metadata DataFrame (train.csv).
            spec_dir:    Directory containing the pre-processed .npy files.
            batch_size:  Samples per batch.
            num_workers: DataLoader worker processes.
            seed:        Random seed for reproducible splits.
            n_split:     Total number of folds.
            n_fold:      Which fold to use as the validation set (0-indexed).
            min_votes:   Minimum total expert votes required to keep a sample.
                         Higher values filter out low-consensus annotations.
            verbose:     If True, print split statistics and distribution checks.
        """
        super().__init__()
        self.metadata    = metadata
        self.spec_dir    = spec_dir
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.seed        = seed
        self.n_split     = n_split
        self.n_fold      = n_fold
        self.min_votes   = min_votes
        self.verbose     = verbose

    def setup(self, stage: str = None):
        """Compute or load splits, then instantiate train/val BrainDataset subsets.

        On first call for a given (n_split, seed, min_votes) configuration, runs
        StratifiedGroupKFold and computes per-fold mean/std via a streaming pass
        over the training spectrograms, then writes results to a JSON cache file.
        Subsequent calls load from cache instantly.

        Args:
            stage: Ignored (test stage is not supported).
        """
        if stage == "test":
            raise NotImplementedError("Test stage is not supported in BrainDataModule.")

        # keep only the subsample with the most votes per unique EEG recording
        self.metadata["total_votes"] = self.metadata[VOTE_COL].sum(axis=1)
        self.metadata = self.metadata.loc[
            self.metadata.groupby("eeg_id")["total_votes"].idxmax()
        ].reset_index(drop=True)

        # filter by minimum vote count
        before_filtering = len(self.metadata)
        self.metadata = self.metadata[
            self.metadata["total_votes"] >= self.min_votes
        ].reset_index(drop=True)

        if self.verbose:
            print(f"Filtered by min_votes={self.min_votes}: "
                  f"{len(self.metadata)}/{before_filtering} kept")

        cache_path = Path(
            CACHE_DIR /
            f"{self.n_split}_at_seed_{self.seed}_min_vote_{self.min_votes}.json"
        )

        # loading data from cache if exists (splits and normalization data)
        if cache_path.exists():
            print(f"[setup] Loading splits from cache: {cache_path}")
            with open(cache_path) as f:
                splits_dict = json.load(f)
            print(f"[setup] Cache loaded — {len(splits_dict)} folds found")

        else:
            print(f"[setup] No cache found — computing {self.n_split}-fold split "
                  f"(seed={self.seed})")

            sgkf   = StratifiedGroupKFold(n_splits=self.n_split, random_state=self.seed, shuffle=True)
            groups = self.metadata["patient_id"]
            X      = range(len(self.metadata))
            y      = self.metadata["expert_consensus"]

            splits_dict = {}

            for i, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
                print(f"\n[setup] Fold {i+1}/{self.n_split} — "
                      f"{len(train_idx)} train / {len(val_idx)} val")
                splits_dict[str(i)] = {
                    "train_idx": train_idx.tolist(),
                    "val_idx":   val_idx.tolist(),
                }

                # streaming mean/std over training spectrograms (after log1p)
                mean = 0
                var  = 0
                for idx in tqdm(train_idx, desc=f"  Fold {i+1} mean/std", unit="spec"):
                    spec  = load_spectrogram(base_path=self.spec_dir, df=self.metadata, idx=idx)
                    spec  = np.log1p(spec)
                    mean += np.mean(spec)
                    var  += np.var(spec)

                final_mean = mean / len(train_idx)
                final_std  = np.sqrt(var / len(train_idx))
                print(f"  → mean={final_mean:.4f}, std={final_std:.4f}")

                splits_dict[str(i)]["mean"] = final_mean.tolist()
                splits_dict[str(i)]["std"]  = final_std.tolist()

            print(f"\n[setup] Saving cache to {cache_path}")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(splits_dict, f)

        # load the selected fold
        fold            = splits_dict[str(self.n_fold)]
        self.train_idx  = np.array(fold["train_idx"])
        self.val_idx    = np.array(fold["val_idx"])
        self.mean       = np.array(fold["mean"], dtype=np.float32)
        self.std        = np.array(fold["std"],  dtype=np.float32)

        # distribution checks
        if self.verbose:
            train_dist = self.metadata.iloc[self.train_idx]["expert_consensus"].value_counts(normalize=True).sort_index()
            val_dist   = self.metadata.iloc[self.val_idx  ]["expert_consensus"].value_counts(normalize=True).sort_index()
            print("[Check] Train set distribution:\n", train_dist.to_string())
            print("[Check] Val distribution:\n",       val_dist.to_string())

            train_patients = self.metadata.iloc[self.train_idx]["patient_id"].nunique()
            val_patients   = self.metadata.iloc[self.val_idx  ]["patient_id"].nunique()
            print(f"[Check] Train: {len(self.train_idx)} samples, {train_patients} patients")
            print(f"[Check] Val:   {len(self.val_idx)}   samples, {val_patients} patients")

            train_top = self.metadata.iloc[self.train_idx]["patient_id"].value_counts().head(5)
            val_top   = self.metadata.iloc[self.val_idx  ]["patient_id"].value_counts().head(5)
            print(f"[Check] Top 5 patients train:\n{train_top.to_string()}")
            print(f"[Check] Top 5 patients val:\n{val_top.to_string()}")

        self.dataset_train = BrainDataset(
            metadata=self.metadata, spec_dir=self.spec_dir,
            mean=self.mean, std=self.std, augment=False,
        )
        self.dataset_val = BrainDataset(
            metadata=self.metadata, spec_dir=self.spec_dir,
            mean=self.mean, std=self.std, augment=False,
        )

        if stage in ("fit", None):
            self.train_ds = Subset(self.dataset_train, self.train_idx)
            self.val_ds   = Subset(self.dataset_val,   self.val_idx)

    def train_dataloader(self) -> DataLoader:
        """Return the training DataLoader (shuffled)."""
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation DataLoader (not shuffled)."""
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=self.num_workers > 0,
        )
