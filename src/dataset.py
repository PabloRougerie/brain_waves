import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent))
import pandas as pd
import numpy as np
from src.utils import *
from src.params import *
import json
from src.preprocess import *

import torch
from torch import nn
from torch.utils.data import Dataset, Subset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from pytorch_lightning import LightningDataModule, LightningModule


#==========
# DATASET
#==========

class BrainDataset(Dataset):
    """PyTorch Dataset for EEG spectrograms. Returns (spec, votes) pairs."""

    def __init__(self, metadata: pd.DataFrame, mean: float, std: float):
        """
        Args:
            metadata: train.csv DataFrame.
            mean: train-set mean (computed streaming in DataModule).
            std:  train-set std  (computed streaming in DataModule).
        """
        super().__init__()
        self.metadata = metadata
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx: int):
        """Load, transform and return (spec tensor, vote distribution tensor)."""

        #load spectrogram from disk
        spec = load_spectrogram(df=self.metadata, idx=idx)

        #log(1+x) transform to reduce dynamic range — applied before normalization
        spec = np.log1p(spec)  # should this be before norm?

        #z-score normalization using train-set statistics
        spec = (spec - self.mean) / self.std  # add epsilon if self.std == 0?

        #convert to tensor
        spec = torch.tensor(spec, dtype=torch.float32)

        #get raw expert votes for this sample
        votes = self.metadata.iloc[idx][VOTE_COL].values.astype(int)

        #normalize to probability distribution
        votes = votes / votes.sum()

        #convert to tensor
        votes = torch.from_numpy(votes).float()

        return spec, votes


#======================
# LIGHTNING DATA MODULE
#======================

class BrainDataModule(LightningDataModule):
    """
    LightningDataModule for BrainDataset.

    Handles stratified group k-fold splitting (grouped by patient_id),
    streaming mean/std computation, and caching of split metadata to disk.
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        batch_size: int = 32,
        num_workers: int = 4,
        seed: int = 273,
        n_split: int = 5,
        n_fold: int = 0,
    ):
        """
        Args:
            metadata:    train.csv DataFrame — needed for both dataset and splits.
            batch_size:  samples per batch.
            num_workers: DataLoader worker processes.
            seed:        random seed for reproducible splits.
            n_split:     number of folds.
            n_fold:      which fold to use as validation set.
        """
        super().__init__()
        self.metadata = metadata
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.n_split = n_split
        self.n_fold = n_fold

    def setup(self, stage: str = None):
        """Compute or load splits, then instantiate train/val subsets."""

        #temporary safety check
        if stage == "test":
            raise NotImplementedError("Test stage is not supported in BrainDataModule.")

        #check if splits at this config are already cached
        cache_path = Path(CACHE_DIR / f"{self.n_split}_at_seed_{self.seed}.json")
        if cache_path.exists():

            print(f"[setup] Loading splits from cache: {cache_path}")
            with open(cache_path) as f:
                splits_dict = json.load(f)
            print(f"[setup] Cache loaded — {len(splits_dict)} folds found")

        else:
            #no cache data, need to do that split and calculate mean and std on train set
            print(f"[setup] No cache found — computing {self.n_split}-fold split (seed={self.seed})")

            sgkf = StratifiedGroupKFold(n_splits=self.n_split, random_state=self.seed, shuffle=True)
            groups = self.metadata["patient_id"]
            X = range(len(self.metadata))
            y = self.metadata["expert_consensus"]

            #dict to hold data for each fold
            splits_dict = {}

            #iterate through folds
            for i, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):

                print(f"\n[setup] Fold {i+1}/{self.n_split} — {len(train_idx)} train / {len(val_idx)} val")
                splits_dict[str(i)] = {}  #keys: fold index (str), values: fold metadata
                splits_dict[str(i)]["train_idx"] = train_idx.tolist()
                splits_dict[str(i)]["val_idx"] = val_idx.tolist()

                #calculate train-set mean and std in streaming fashion (approximation)
                mean = 0
                var = 0
                for idx in tqdm(train_idx, desc=f"  Fold {i+1} mean/std", unit="spec"):
                    spec = load_spectrogram(self.metadata, idx)
                    spec = np.log1p(spec)
                    mean += np.mean(spec)
                    var += np.var(spec)

                final_mean = mean / len(train_idx)
                final_std = np.sqrt(var / len(train_idx))
                print(f"  → mean={final_mean:.4f}, std={final_std:.4f}")

                splits_dict[str(i)]["mean"] = final_mean.tolist()
                splits_dict[str(i)]["std"] = final_std.tolist()

            #save splits to cache
            print(f"\n[setup] Saving cache to {cache_path}")
            with open(cache_path, "w") as f:
                json.dump(splits_dict, f)

        #load data for the desired fold
        self.train_idx = np.array(splits_dict[str(self.n_fold)]["train_idx"])
        self.val_idx   = np.array(splits_dict[str(self.n_fold)]["val_idx"])
        self.mean      = np.array(splits_dict[str(self.n_fold)]["mean"])
        self.std       = np.array(splits_dict[str(self.n_fold)]["std"])

        print(f"train_idx dtype: {self.train_idx.dtype}")
        print(f"val_idx   dtype: {self.val_idx.dtype}")
        print(f"mean      dtype: {self.mean.dtype}")
        print(f"std       dtype: {self.std.dtype}")

        self.dataset = BrainDataset(metadata=self.metadata, mean=self.mean, std=self.std)

        if stage in ("fit", None):
            self.train_ds = Subset(self.dataset, self.train_idx)
            self.val_ds   = Subset(self.dataset, self.val_idx)

    def train_dataloader(self) -> DataLoader:
        """Return the training DataLoader (shuffled)."""
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        """Return the validation DataLoader (not shuffled)."""
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
