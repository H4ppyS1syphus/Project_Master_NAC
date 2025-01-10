# data_loading.py

import numpy as np

def load_waveforms(file_path, key="data_array", fraction=1.0, time_start=200, time_end=800):
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

    waveforms = waveforms[:, :, time_start:time_end]

    return waveforms
