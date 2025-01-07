"""
data_loading.py

Utilities for loading datasets from .npz files.
"""

import numpy as np

def load_waveforms(file_path, key="data_array", fraction=1.0, time_start=200, time_end=800):
    """
    Load a fraction of the waveforms from an .npz file, 
    then slice the time dimension [time_start:time_end].

    Args:
        file_path (str): Path to the .npz file.
        key (str): Key to access the data in the .npz file.
        fraction (float): Fraction of the data to load (0 < fraction <= 1.0).
        time_start (int): Start index for time slicing.
        time_end (int): End index (non-inclusive) for time slicing.

    Returns:
        numpy.ndarray: shape (N, C, time_end - time_start), 
                       Subset of waveforms in both event and time dimensions.
    """
    import numpy as np

    with np.load(file_path, mmap_mode="r") as data:
        waveforms = data[key]  # shape: (N, Channels, Timepoints)
        total_samples = waveforms.shape[0]

        # 1) Optional fraction sampling of events
        if fraction < 1.0:
            sampled_indices = np.random.choice(
                total_samples, 
                size=int(total_samples * fraction), 
                replace=False
            )
            waveforms = waveforms[sampled_indices]

    # 2) Slice the time dimension => waveforms[:, :, time_start:time_end]
    waveforms = waveforms[:, :, time_start:time_end]

    return waveforms