"""
Preprocessing pipeline: raw parquet spectrograms → cropped .npy arrays.

Pipeline per subsample:
  1. Load raw parquet spectrogram for the spectrogram_id.
  2. Crop to a `duration`-second window starting at offset + 1 s.
  3. Extract the central 50-second window (25 time steps at 2 s/step).
  4. Reshape into (time=25, region=4, freq=100).
  5. Transpose to (region=4, freq=100, time=25) and save as .npy.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.params import PROCESSED_DIR, SPEC_DIR


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_parquet(spec_id: str) -> pd.DataFrame:
    """Load the raw spectrogram parquet file for a given spectrogram ID."""
    return pd.read_parquet(SPEC_DIR / f"{spec_id}.parquet")


def save_spec(spec: np.ndarray, path: Path) -> None:
    """Save a spectrogram array to disk as a .npy file."""
    np.save(path, spec)


# ─────────────────────────────────────────────────────────────────────────────
# Cropping helpers
# ─────────────────────────────────────────────────────────────────────────────

def crop_subsample(df: pd.DataFrame, offset: float, duration: float) -> pd.DataFrame:
    """Crop spectrogram to [offset+1, offset+1+duration] seconds.

    Args:
        df:       Raw spectrogram DataFrame with a `time` column.
        offset:   Label offset in seconds from the metadata.
        duration: Window length in seconds.

    Returns:
        Filtered DataFrame containing only rows within the window.
    """
    start = offset + 1
    return df[(df["time"] >= start) & (df["time"] < start + duration)]


def crop_central_window(df: pd.DataFrame, duration: float = 50) -> pd.DataFrame:
    """Crop a spectrogram DataFrame to a centered window of `duration` seconds.

    With duration=50 s and 2 s/step resolution this yields 25 time steps,
    corresponding to the 50-second window reviewed by the expert annotators.

    Args:
        df:       Spectrogram DataFrame already cropped to a coarse window.
        duration: Target window length in seconds (default 50).

    Returns:
        DataFrame with exactly (or at most) `duration / 2` time steps.
    """
    start  = df["time"].min()
    end    = df["time"].max()
    center = (end + start) / 2
    return df[
        (df["time"] >= center - duration // 2) &
        (df["time"] <  center + duration // 2)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Reshape helpers
# ─────────────────────────────────────────────────────────────────────────────

def reshape_by_region(
    df: pd.DataFrame,
    regions: list[str] = ["LL_", "RL_", "LP_", "RP_"],
    n_freq: int = 100,
) -> np.ndarray:
    """Reshape a flat spectrogram DataFrame into a 3D array (time, region, freq).

    Each region's frequency columns are stacked along a dedicated dimension.

    Args:
        df:      Cropped spectrogram DataFrame.
        regions: List of region prefixes to extract (default: 4 EEG regions).
        n_freq:  Number of frequency bins per region (default 100).

    Returns:
        Float32 array of shape (n_time, n_regions, n_freq).
    """
    n_times   = df.shape[0]
    n_regions = len(regions)
    spec      = np.zeros((n_times, n_regions, n_freq), dtype=np.float32)

    for i, region in enumerate(regions):
        region_cols   = [c for c in df.columns if c.startswith(region)]
        spec[:, i, :] = df[region_cols].to_numpy()

    return spec


def reorder_dimensions(arr: np.ndarray, axes: tuple = (1, 2, 0)) -> np.ndarray:
    """Transpose a spectrogram array to a new axis order.

    Default (1, 2, 0) converts (time, region, freq) → (region, freq, time).
    """
    return arr.transpose(axes)


# ─────────────────────────────────────────────────────────────────────────────
# Batch pipeline
# ─────────────────────────────────────────────────────────────────────────────

def prepare_spectrograms(
    df: pd.DataFrame,
    save_dir: str,
    duration: float = 600,
    architecture: str = "cnn",
    force: bool = False,
) -> None:
    """Crop, reshape, and save all spectrogram subsamples from a metadata DataFrame.

    For each row in `df`, loads the corresponding raw parquet spectrogram, applies
    the two-stage crop (coarse window → central 50 s window), reshapes to
    (4, 100, 25), and saves as a .npy file under `PROCESSED_DIR / save_dir /`.

    Subsamples that produce fewer than 25 time steps after cropping (edge cases near
    the spectrogram boundary) are skipped and logged to `PROCESSED_DIR/bad_cases.json`.

    Args:
        df:           Metadata DataFrame (train.csv).
        save_dir:     Subdirectory name under `PROCESSED_DIR` for the output files.
        duration:     Coarse crop window length in seconds (default 600).
        architecture: Reserved for future architecture-specific reshaping
                      ('cnn', 'lstm', or 'transformer'). Currently all produce the
                      same (4, 100, 25) output.
        force:        If True, recompute and overwrite existing files (default False).
    """
    if architecture not in ["cnn", "lstm", "transformer"]:
        raise ValueError(
            f"architecture must be 'cnn', 'lstm', or 'transformer', got '{architecture}'"
        )

    spec_groups = list(df.groupby("spectrogram_id"))
    n_total     = len(df)
    n_saved     = 0
    n_skipped   = 0
    bad_cases   = []  # (spec_id, subsample_id, offset, actual_len)

    print(f"Starting preprocessing: {len(spec_groups)} spectrograms / {n_total} subsamples")

    for spec_id, group in tqdm(spec_groups, desc="Spectrograms", unit="spec"):
        df_spec = load_parquet(spec_id)

        for _, row in group.iterrows():
            offset       = row["spectrogram_label_offset_seconds"]
            subsample_id = row["spectrogram_sub_id"]
            save_path    = Path(PROCESSED_DIR / f"{save_dir}" / f"{spec_id}-{subsample_id}")

            if save_path.exists() and not force:
                n_skipped += 1
                continue

            df_cropped     = crop_subsample(df_spec, offset, duration)
            central_window = crop_central_window(df_cropped)

            # expect exactly 25 time steps (50 s window / 2 s per step)
            if len(central_window) != 25:
                bad_cases.append((spec_id, subsample_id, offset, len(central_window)))
                continue  # skip edge cases near the spectrogram boundary

            spec_array = reshape_by_region(df=central_window)
            spec_out   = reorder_dimensions(arr=spec_array, axes=(1, 2, 0))

            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_spec(spec=spec_out, path=save_path)
            n_saved += 1

    if bad_cases:
        print(f"WARNING: {len(bad_cases)} subsamples skipped (unexpected crop size)")
        log_path = PROCESSED_DIR / "bad_cases.json"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            json.dump(bad_cases, f, indent=2)
        print(f"Bad cases logged to {log_path}")

    print(f"Done — {n_saved} saved, {n_skipped} skipped (already existed)")
