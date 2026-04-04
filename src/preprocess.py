import pandas as pd
import numpy as np
from src.params import *
from pathlib import Path


def load_parquet(spec_id: str) -> pd.DataFrame:
    """Load raw spectrogram parquet file for a given spectrogram ID."""
    return pd.read_parquet(SPEC_DIR / f"{spec_id}.parquet")


def crop(df: pd.DataFrame, offset: float, duration: float) -> pd.DataFrame:
    """Crop spectrogram to [offset+1, offset+1+duration] seconds."""
    start = offset + 1
    return df[(df["time"] >= start) & (df["time"] < start + duration)]


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
    spec = np.zeros((n_times, n_regions, n_freq))

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


def prepare_spectrograms(df: pd.DataFrame, duration: float = 600, architecture: str = "cnn", force= False) -> None:
    """
    Crop, reshape, and save all spectrogram subsamples from a metadata DataFrame.

    Supports 'cnn', 'lstm', and 'transformer' architectures.
    Output shape: (region, freq, time) for all architectures at this stage.
    """


    if architecture not in ["cnn", "lstm", "transformer"]:
        raise ValueError(f"architecture must be 'cnn', 'lstm', or 'transformer', got '{architecture}'")

    #iterates over spec
    for spec_id, group in df.groupby("spectrogram_id"):
        df_spec = load_parquet(spec_id)

        #iterates over subsamples
        for _, row in group.iterrows():

            offset = row["spectrogram_label_offset_seconds"]
            subsample_id = row["spectrogram_sub_id"]

            save_path = Path(PROCESSED_DIR / f"{spec_id}-{subsample_id}")

            #avoid re-generating existing spec, unless forced
            if save_path.exists() and force is not False:
                print(f"spectrogram {spec_id} - subsample {subsample_id} already exists: skipped")
                continue

            #crop to right start and duration
            df_cropped = crop(df_spec, offset, duration)
            #reshape to shape (time, region, frequencies)
            spec_array = reshape_by_region(df=df_cropped)

            if architecture == "cnn":
                #reorder to (canal, frequencies, times) as needed for CNN
                spec_out = reorder_dimensions(arr=spec_array, axes=(1, 2, 0))
            else:
                spec_out = reorder_dimensions(arr=spec_array, axes=(1, 2, 0))

            #save subsampled spectrograms with explicit names.
            save_path.parent.mkdir(parents=True, exist_ok=True) #create parent dir if doesnt exists
            save_spec(spec=spec_out, path=save_path)
