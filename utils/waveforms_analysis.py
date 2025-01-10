import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set_theme(style="whitegrid", context="notebook")

def compute_statistics(waveforms):
    """
    Compute basic statistics (mean, max) for each sample/channel.
    Args:
        waveforms (numpy.ndarray): shape (N, channels, timepoints).
    Returns:
        dict: {"mean": array, "max": array}
    """
    mean_vals = np.mean(waveforms, axis=2)  # shape (N, channels)
    max_vals  = np.max(waveforms, axis=2)   # shape (N, channels)

    return {"mean": mean_vals, "max": max_vals}

def plot_stat_distributions(stat_Li6, stat_Po, stat_name, dataset_names, show=True):
    """
    Plot distributions of a given statistic (e.g. "Mean Amplitude") for Li6 vs Po.
    Each channel is plotted side-by-side.

    Args:
        stat_Li6 (np.ndarray): shape (N_Li6, channels)
        stat_Po (np.ndarray): shape (N_Po, channels)
        stat_name (str): e.g. "Mean Amplitude"
        dataset_names (tuple): e.g. ("Li6", "Po")
        show (bool): Show inline or save to file
    """
    n_channels = stat_Li6.shape[1]

    fig, axes = plt.subplots(1, n_channels, figsize=(5 * n_channels, 4), sharey=True)
    if n_channels == 1:
        axes = [axes]

    for ch in range(n_channels):
        sns.histplot(stat_Li6[:, ch], kde=True, color="blue", label=dataset_names[0],
                     ax=axes[ch], alpha=0.6)
        sns.histplot(stat_Po[:, ch], kde=True, color="red", label=dataset_names[1],
                     ax=axes[ch], alpha=0.6)
        axes[ch].set_title(f"{stat_name} - Ch {ch}")
        axes[ch].set_xlabel(stat_name)
        axes[ch].set_ylabel("Frequency")
        axes[ch].legend()

    plt.tight_layout()
    if show:
        plt.show()
    else:
        os.makedirs("plots/statistics", exist_ok=True)
        plt.savefig(f"plots/statistics/{stat_name.replace(' ', '_')}_distributions.png")
        plt.close()

def compute_average_waveform(waveforms):
    """
    Compute average waveform across all samples, per channel.
    Returns shape (channels, timepoints).
    """
    return np.mean(waveforms, axis=0)

def plot_overall_average_overlay(avg_Li6, avg_Po, show=True):
    """
    Overlays overall average waveforms for Li6 vs Po.
    """
    overall_Li6 = np.mean(avg_Li6, axis=0)
    overall_Po  = np.mean(avg_Po, axis=0)

    plt.figure(figsize=(8, 4))
    plt.plot(overall_Li6, label="Li6 Overall Avg", color="blue")
    plt.plot(overall_Po,  label="Po Overall Avg",  color="red")
    plt.title("Overall Average Waveform Overlay")
    plt.xlabel("Time Index")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if show:
        plt.show()
    else:
        os.makedirs("plots", exist_ok=True)
        plt.savefig("plots/overall_average_overlay.png")
        plt.close()

def plot_random_waveforms(waveforms, dataset_name="Dataset", num_samples=3, show=True):
    """
    Plot a few random waveforms from the dataset.
    """
    fig, axes = plt.subplots(num_samples, 1, figsize=(10, 2 * num_samples), sharex=True)
    if num_samples == 1:
        axes = [axes]

    for i in range(num_samples):
        idx = np.random.randint(0, waveforms.shape[0])
        for ch in range(waveforms.shape[1]):
            axes[i].plot(waveforms[idx, ch], alpha=0.7, label=f"Ch{ch}")
        axes[i].legend(loc="upper right")
        axes[i].set_title(f"{dataset_name} - Sample {idx}")

    plt.xlabel("Time Index")
    plt.tight_layout()

    if show:
        plt.show()
    else:
        os.makedirs("plots", exist_ok=True)
        plt.savefig(f"plots/{dataset_name.lower()}_random_waveforms.png")
        plt.close()

def plot_average_waveforms_per_detector(avg_Li6, avg_Po, show=True):
    """
    Plots average waveform for each channel, plus a separate subplot for the
    overall mean across channels.
    """
    n_channels = avg_Li6.shape[0]
    mean_Li6 = np.mean(avg_Li6, axis=0)
    mean_Po  = np.mean(avg_Po,  axis=0)

    fig, axes = plt.subplots(2, 3, figsize=(10, 6), constrained_layout=True)
    axes = axes.flatten()

    for ch in range(n_channels):
        axes[ch].plot(avg_Li6[ch], label="Li6", color="blue")
        axes[ch].plot(avg_Po[ch],  label="Po",  color="red")
        axes[ch].set_title(f"Detector {ch + 1}")
        axes[ch].legend()

    axes[n_channels].plot(mean_Li6, label="Li6 Mean", color="blue", linestyle="--")
    axes[n_channels].plot(mean_Po,  label="Po Mean",  color="red",  linestyle="--")
    axes[n_channels].set_title("Mean of All Detectors")
    axes[n_channels].legend()

    for ax in axes[n_channels + 1:]:
        ax.axis("off")

    if show:
        plt.show()
    else:
        os.makedirs("plots", exist_ok=True)
        plt.savefig("plots/average_waveforms_per_detector.png")
        plt.close()

def plot_difference_waveforms_per_detector(avg_Li6, avg_Po, show=True):
    """
    Plots average waveforms for Li6 vs Po in subplots, also an overall subplot
    for the means. (Same layout as above, but for differences if needed.)
    """
    n_channels = avg_Li6.shape[0]
    mean_Li6 = np.mean(avg_Li6, axis=0)
    mean_Po  = np.mean(avg_Po,  axis=0)

    fig, axes = plt.subplots(2, 3, figsize=(10, 6), constrained_layout=True)
    axes = axes.flatten()

    for ch in range(n_channels):
        axes[ch].plot(avg_Li6[ch], label="Li6", color="blue")
        axes[ch].plot(avg_Po[ch],  label="Po",  color="red")
        axes[ch].set_title(f"Detector {ch + 1}")
        axes[ch].legend()

    axes[n_channels].plot(mean_Li6, label="Li6 Mean", color="blue", linestyle="--")
    axes[n_channels].plot(mean_Po,  label="Po Mean",  color="red",  linestyle="--")
    axes[n_channels].set_title("Mean of All Detectors")
    axes[n_channels].legend()

    for ax in axes[n_channels + 1:]:
        ax.axis("off")

    if show:
        plt.show()
    else:
        os.makedirs("plots", exist_ok=True)
        plt.savefig("plots/difference_waveforms_per_detector.png")
        plt.close()

def plot_heatmap_avg_difference(avg_li6, avg_po, dataset_names=("Li6", "Po"), show=True):
    """
    Create a 2D heatmap showing absolute differences between avg Li6 and Po waveforms.
    Shape: (channels x timepoints)
    """
    diff = np.abs(avg_li6 - avg_po)

    plt.figure(figsize=(8, 8))
    sns.heatmap(diff, cmap="viridis", cbar=True)
    plt.title(f"2D Heatmap of Abs Differences ({dataset_names[0]} vs {dataset_names[1]})")
    plt.xlabel("Time Index")
    plt.ylabel("Channel Index")
    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/average_difference_heatmap.png", bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()
