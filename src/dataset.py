
import pandas as pd
import numpy as np
from src.utils import *
from src.params import *
import json

from tqdm import tqdm

import torch
from torch.utils.data import Dataset, Subset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from pytorch_lightning import LightningDataModule


#==========
# DATASET
#==========

class BrainDataset(Dataset):
    """PyTorch Dataset for EEG spectrograms. Returns (spec, votes) pairs."""

    def __init__(self, metadata: pd.DataFrame, spec_dir: Path, mean: float, std: float, augment= False):
        """
        Args:
            metadata: train.csv DataFrame.
            spec_dir: path to the folder containing .npy spectrogram files.
            mean: train-set mean (computed streaming in DataModule).
            std:  train-set std  (computed streaming in DataModule).
        """
        super().__init__()
        self.metadata = metadata
        self.spec_dir = spec_dir
        self.mean = mean
        self.std = std
        self.augment = augment

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx: int):
        """Load, transform and return (spec tensor, vote distribution tensor)."""

        #load spectrogram from disk
        spec = load_spectrogram(base_path=self.spec_dir, df=self.metadata, idx=idx)

        #log(1+x) transform to reduce dynamic range — applied before normalization
        spec = np.log1p(spec)  # should this be before norm?

        #z-score normalization using train-set statistics
        spec = (spec - self.mean) / self.std  # add epsilon if self.std == 0?

        if self.augment:
            spec = spec + np.random.normal(0, 0.05, spec.shape)
        #convert to tensor
        spec = torch.tensor(spec, dtype=torch.float32)

        #get raw expert votes for this sample
        votes = self.metadata.iloc[idx][VOTE_COL].values.astype(int)

        #normalize to probability distribution
        votes = votes / votes.sum()

        #convert to tensor
        votes = torch.from_numpy(votes).float()

        return spec, votes

    def xy_masking(self, spec, num_masks_x=1, mask_size_x=10,
                num_masks_y=1, mask_size_y=30):

        spec = spec.copy()
        _, n_freq, n_time = spec.shape

        # Mask frequency bands
        for _ in range(num_masks_x):
            f_start = np.random.randint(0, n_freq - mask_size_x)
            spec[:, f_start:f_start + mask_size_x, :] = 0

        # Mask time windows
        for _ in range(num_masks_y):
            t_start = np.random.randint(0, n_time - mask_size_y)
            spec[:, :, t_start:t_start + mask_size_y] = 0

        return spec


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
        spec_dir: Path,
        batch_size: int = 32,
        num_workers: int = 4,
        seed: int = 273,
        n_split: int = 5,
        n_fold: int = 0,
    ):
        """
        Args:
            metadata:    train.csv DataFrame — needed for both dataset and splits.
            spec_dir:    path to the folder containing .npy spectrogram files.
            batch_size:  samples per batch.
            num_workers: DataLoader worker processes.
            seed:        random seed for reproducible splits.
            n_split:     number of folds.
            n_fold:      which fold to use as validation set.
        """
        super().__init__()
        self.metadata = metadata
        self.spec_dir = spec_dir
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

        #select the eeg subsample with most vote for each recording
        self.metadata["total_votes"] = self.metadata[VOTE_COL].sum(axis=1)
        self.metadata = self.metadata.loc[
        self.metadata.groupby("eeg_id")["total_votes"].idxmax()
        ].reset_index(drop=True)

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
                    spec = load_spectrogram(base_path=self.spec_dir, df=self.metadata, idx=idx)
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
        self.mean      = np.array(splits_dict[str(self.n_fold)]["mean"], dtype= np.float32)
        self.std       = np.array(splits_dict[str(self.n_fold)]["std"], dtype= np.float32)

        #check train vs val
        train_dist = self.metadata.iloc[self.train_idx]["expert_consensus"].value_counts(normalize=True).sort_index()
        val_dist = self.metadata.iloc[self.val_idx]["expert_consensus"].value_counts(normalize=True).sort_index()
        print("[Check] Train set distribution:\n", train_dist.to_string())
        print("[Check] Val distribution:\n", val_dist.to_string())

        train_patients = self.metadata.iloc[self.train_idx]["patient_id"].nunique()
        val_patients = self.metadata.iloc[self.val_idx]["patient_id"].nunique()
        print(f"[Check] Train: {len(self.train_idx)} samples, {train_patients} patients")
        print(f"[Check] Val:   {len(self.val_idx)} samples, {val_patients} patients")

        train_top = self.metadata.iloc[self.train_idx]["patient_id"].value_counts().head(5)
        val_top = self.metadata.iloc[self.val_idx]["patient_id"].value_counts().head(5)

        print(f"[Check] Top 5 patients train:\n{train_top.to_string()}")
        print(f"[Check] Top 5 patients val:\n{val_top.to_string()}")


        #create the same dataset but with different augmentation settings
        self.dataset_train = BrainDataset(metadata=self.metadata, spec_dir=self.spec_dir, mean=self.mean, std=self.std, augment=False)
        self.dataset_val   = BrainDataset(metadata=self.metadata, spec_dir=self.spec_dir, mean=self.mean, std=self.std, augment=False)

        if stage in ("fit", None):
            self.train_ds = Subset(self.dataset_train, self.train_idx)
            self.val_ds   = Subset(self.dataset_val, self.val_idx)

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
