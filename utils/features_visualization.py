# In features_visualization.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

sns.set_theme(style="whitegrid", context="notebook")

def plot_features_by_base_feature(
    features_Li6, 
    features_Po, 
    base_feature_names, 
    save_path="plots",
    show=True
):
    """
    For each base feature (e.g., 'Mean', 'Peak', 'Dominant_Freq', 'Spectral_Entropy'),
    create a figure with 4 subplots, each subplot representing one detector (channel).

    Args:
        features_Li6 (np.ndarray): shape (N_Li6, 16) => 4 base features x 4 channels
        features_Po  (np.ndarray): shape (N_Po, 16)  => same
        base_feature_names (list): e.g. ["Mean", "Peak", "Dominant_Freq", "Spectral_Entropy"]
        save_path (str): directory path to save PNGs
        show (bool): if True, display inline; if False, save to file then close.
    """
    os.makedirs(save_path, exist_ok=True)

    # We assume exactly 4 base features => total columns = 16 (4 features * 4 channels)
    # The columns order is typically: 
    #   (Mean_Ch0, Peak_Ch0, DomFreq_Ch0, SpecEnt_Ch0,
    #    Mean_Ch1, Peak_Ch1, DomFreq_Ch1, SpecEnt_Ch1,
    #    Mean_Ch2, Peak_Ch2, DomFreq_Ch2, SpecEnt_Ch2,
    #    Mean_Ch3, Peak_Ch3, DomFreq_Ch3, SpecEnt_Ch3)

    num_channels = 4  # 4 detectors
    # We'll create one figure per base feature.
    for i, base_feat in enumerate(base_feature_names):
        # For base_feat 'Mean' => columns = 0, 4, 8, 12
        # For base_feat 'Peak' => columns = 1, 5, 9, 13
        # etc.
        
        fig, axes = plt.subplots(1, num_channels, figsize=(14, 3), sharey=True)
        if num_channels == 1:
            axes = [axes]  # ensure it's iterable

        for ch in range(num_channels):
            col_idx = i + 4 * ch
            sns.histplot(features_Li6[:, col_idx], kde=True, color="blue",
                         alpha=0.5, label="Li6", ax=axes[ch])
            sns.histplot(features_Po[:, col_idx],  kde=True, color="red",
                         alpha=0.5, label="Po",  ax=axes[ch])

            axes[ch].set_title(f"{base_feat} (Detector {ch})")
            axes[ch].set_xlabel(base_feat)
            axes[ch].set_ylabel("Count")
            axes[ch].legend()

        plt.suptitle(f"{base_feat} Distributions by Detector", y=1.02, fontsize=12)
        plt.tight_layout()
        if base_feat != "Dominant_Freq":
            if show:
                plt.show()
        else:
            out_file = f"{base_feat.lower()}_by_detector.png"
            plt.savefig(os.path.join(save_path, out_file))
            plt.close()
