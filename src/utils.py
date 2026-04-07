from src.params import *
import numpy as np
import pandas as pd



def load_spectrogram(df: pd.DataFrame, idx: int) -> np.ndarray:
    """Load a preprocessed spectrogram array from disk for a given metadata row index."""
    spec_id = df.iloc[idx]["spectrogram_id"]
    subsample_id = df.iloc[idx]["spectrogram_sub_id"]

    #load file
    path = PROCESSED_DIR / f"{spec_id}-{subsample_id}.npy"
    spec = np.load(path)

    #impute zero to nan values
    spec= np.nan_to_num(spec, nan=0.0)

    return spec
