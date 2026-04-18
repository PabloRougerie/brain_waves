"""Utility functions shared across the project (spectrogram loading, etc.)."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.params import PROCESSED_DIR  # noqa: F401 — re-exported for notebook convenience


def load_spectrogram(base_path: Path, df: pd.DataFrame, idx: int) -> np.ndarray:
    """Load a pre-processed spectrogram array from disk for a given metadata row.

    Args:
        base_path: Directory containing the .npy spectrogram files.
        df:        Metadata DataFrame (must have `spectrogram_id` and `spectrogram_sub_id` columns).
        idx:       Integer row index into `df`.

    Returns:
        Float32 numpy array of shape (4, 100, 25). NaN values are replaced with 0.
    """
    spec_id      = df.iloc[idx]["spectrogram_id"]
    subsample_id = df.iloc[idx]["spectrogram_sub_id"]

    path = Path(base_path) / f"{spec_id}-{subsample_id}.npy"
    spec = np.load(path)

    # replace NaN with 0 (rare edge case from preprocessing)
    spec = np.nan_to_num(spec, nan=0.0)

    return spec
