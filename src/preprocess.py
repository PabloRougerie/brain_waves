import pandas as pd
import numpy as np
import json
from src.params import *
from pathlib import Path
from tqdm import tqdm


def load_parquet(spec_id: str) -> pd.DataFrame:
    """Load raw spectrogram parquet file for a given spectrogram ID."""
    return pd.read_parquet(SPEC_DIR / f"{spec_id}.parquet")


def crop_subsample(df: pd.DataFrame, offset: float, duration: float) -> pd.DataFrame:
    """Crop spectrogram to [offset+1, offset+1+duration] seconds."""
    start = offset + 1
    return df[(df["time"] >= start) & (df["time"] < start + duration)]

def crop_central_window(df: pd.DataFrame, duration: float = 50) -> pd.DataFrame:
    """
    Crop a spectrogram DataFrame to a centered window of `duration` seconds.

    Used to extract a fixed-length window around the center of a subsample.
    With duration=50s and a 2s/step resolution, this yields 25 time steps.
    """
    start = df["time"].min()
    end = df["time"].max()
    center = (end + start) / 2
    # keep only time steps within [center - half, center + half)
    central_window = df[(df["time"] >= center - duration//2) & (df["time"] < center + duration//2)]

    return central_window

def reshape_by_region(
    df: pd.DataFrame,
    regions: list[str] = ["LL_", "RL_", "LP_", "RP_"],
    n_freq: int = 100,
) -> np.ndarray:
    """
    Reshape a flat spectrogram DataFrame into a 3D array (time, region, freq).

    Each region's frequency columns are stacked along a dedicated dimension.
    """
    n_times = df.shape[0]
    n_regions = len(regions)
    spec = np.zeros((n_times, n_regions, n_freq), dtype= np.float32)

    for i, region in enumerate(regions):
        region_cols = [c for c in df.columns if c.startswith(region)]
        region_arr = df[region_cols].to_numpy()
        spec[:, i, :] = region_arr

    return spec


def reorder_dimensions(arr: np.ndarray, axes: tuple = (1, 2, 0)) -> np.ndarray:
    """Transpose a spectrogram array to a new axis order."""
    return arr.transpose(axes)


def save_spec(spec: np.ndarray, path: Path) -> None:
    """Save a spectrogram array to disk as a .npy file."""
    np.save(path, spec)


def prepare_spectrograms(df: pd.DataFrame, save_dir, duration: float = 600, architecture: str = "cnn", force= False) -> None:
    """
    Crop, reshape, and save all spectrogram subsamples from a metadata DataFrame.

    Supports 'cnn', 'lstm', and 'transformer' architectures.
    Output shape: (region, freq, time) for all architectures at this stage.
    """


    if architecture not in ["cnn", "lstm", "transformer"]:
        raise ValueError(f"architecture must be 'cnn', 'lstm', or 'transformer', got '{architecture}'")

    spec_groups = list(df.groupby("spectrogram_id"))
    n_total = len(df)
    n_saved = 0
    n_skipped = 0
    bad_cases = []  # (spec_id, subsample_id, offset, actual_len)

    print(f"Starting preprocessing: {len(spec_groups)} spectrograms / {n_total} subsamples")

    for spec_id, group in tqdm(spec_groups, desc="Spectrograms", unit="spec"):
        df_spec = load_parquet(spec_id)

        for _, row in group.iterrows():
            offset = row["spectrogram_label_offset_seconds"]
            subsample_id = row["spectrogram_sub_id"]
            save_path = Path(PROCESSED_DIR / f"{save_dir}" / f"{spec_id}-{subsample_id}")

            if save_path.exists() and not force:
                n_skipped += 1
                continue

            df_cropped = crop_subsample(df_spec, offset, duration)
            central_window = crop_central_window(df_cropped)


            # expect exactly 25 time steps (50s window / 2s per step)
            if len(central_window) != 25:
                bad_cases.append((spec_id, subsample_id, offset, len(central_window)))
                continue  # skip edge cases near the boundary of the spectrogram

            spec_array = reshape_by_region(df=central_window)

            # placeholder: same reshape for all architectures at this stage
            spec_out = reorder_dimensions(arr=spec_array, axes=(1, 2, 0))

            save_path.parent.mkdir(parents=True, exist_ok=True)
            save_spec(spec=spec_out, path=save_path)
            n_saved += 1

    if bad_cases:
        print(f"WARNING: {len(bad_cases)} subsamples skipped (unexpected crop size)")
        # log bad cases to disk for later inspection
        log_path = PROCESSED_DIR / "bad_cases.json"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w") as f:
            json.dump(bad_cases, f, indent=2)
        print(f"Bad cases logged to {log_path}")

    print(f"Done — {n_saved} saved, {n_skipped} skipped (already existed)")
