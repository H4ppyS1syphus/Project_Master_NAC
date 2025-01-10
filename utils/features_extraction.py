import numpy as np
from scipy.fft import fft, fftfreq

def extract_features(sample, sampling_rate=10000.0):
    """
    Extract features from a single sample (4 signals),
    using only the time indices [200..700].
    """
    sliced_sample = sample[:, :]  # shape (4, 500) if 700-200=500
    n_points = sliced_sample.shape[1]
    freqs_full = fftfreq(n_points, d=1.0 / sampling_rate)
    half_len   = n_points // 2
    freqs      = freqs_full[:half_len]

    features = []
    for wave in sliced_sample:
        # 1) Mean & Peak
        mean_val = np.mean(wave)
        peak_val = np.max(wave)

        # 2) FFT & Magnitude (positive half)
        spectrum = np.abs(fft(wave))[:half_len]
        max_idx  = np.argmax(spectrum)
        dom_freq = freqs[max_idx]

        # 3) Spectral Entropy
        spectral_entropy = -np.sum(spectrum * np.log(spectrum + 1e-8))

        features.extend([mean_val, peak_val, dom_freq, spectral_entropy])

    return np.array(features)

def extract_features_dataset(waveforms, sampling_rate=10000.0):
    """
    Extracts features for the entire dataset,
    slicing each waveform from index 200..700.
    """
    all_features = [extract_features(sample, sampling_rate) for sample in waveforms]
    return np.array(all_features)
