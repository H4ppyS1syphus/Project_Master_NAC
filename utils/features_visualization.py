import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    create a figure with subplots, each representing one detector (channel).
    
    Args:
        features_Li6 (np.ndarray): Shape (N_Li6, 16) => 4 base features x 4 channels
        features_Po  (np.ndarray): Shape (N_Po, 16) => same structure
        base_feature_names (list): e.g. ["Mean", "Peak", "Dominant_Freq", "Spectral_Entropy"]
        save_path (str): Directory path to save PNGs
        show (bool): If True, display inline; if False, save to file and close
    """
    os.makedirs(save_path, exist_ok=True)

    num_channels = 4  # 4 detectors
    
    # Create one figure per base feature
    for i, base_feat in enumerate(base_feature_names):
        fig, axes = plt.subplots(1, num_channels, figsize=(14, 3), sharey=True)
        axes = np.array(axes)  # Ensure axes is iterable even if there's only 1 channel

        for ch in range(num_channels):
            col_idx = i + 4 * ch  # Determine the column index for each feature/channel

            # Plot the histograms for Li6 and Po data
            sns.histplot(features_Li6[:, col_idx], kde=True, color="blue", alpha=0.5, label="Li6", ax=axes[ch])
            sns.histplot(features_Po[:, col_idx], kde=True, color="red", alpha=0.5, label="Po", ax=axes[ch])

            # Set plot details
            axes[ch].set_title(f"{base_feat} (Detector {ch})")
            axes[ch].set_xlabel(base_feat)
            axes[ch].set_ylabel("Count")
            axes[ch].legend()

        plt.suptitle(f"{base_feat} Distributions by Detector", y=1.02, fontsize=12)
        plt.tight_layout()

        # Handle saving or displaying the plot based on base feature and show flag
        if base_feat != "Dominant_Freq":
            if show:
                plt.show()
            else:
                out_file = f"{base_feat.lower()}_by_detector.png"
                plt.savefig(os.path.join(save_path, out_file))
                plt.close()
        else:
            out_file = f"{base_feat.lower()}_by_detector.png"
            plt.savefig(os.path.join(save_path, out_file))
            plt.close()
