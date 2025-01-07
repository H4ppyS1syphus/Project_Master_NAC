"""
data_loading.py

Utilities for loading datasets from .npz files.
"""

import numpy as np

def load_waveforms(file_path, key="data_array", fraction=1.0):
    """
    Load a fraction of the waveforms from an .npz file.

    Args:
        file_path (str): Path to the .npz file.
        key (str): Key to access the data in the .npz file.
        fraction (float): Fraction of the data to load (0 < fraction <= 1.0).

    Returns:
        numpy.ndarray: Subset of the loaded waveforms.
    """
    with np.load(file_path, mmap_mode="r") as data:
        waveforms = data[key]
        total_samples = waveforms.shape[0]

        if fraction < 1.0:
            sampled_indices = np.random.choice(
                total_samples, 
                size=int(total_samples * fraction), 
                replace=False
            )
            waveforms = waveforms[sampled_indices]

    return waveforms
