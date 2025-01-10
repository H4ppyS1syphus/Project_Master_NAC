import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix


def enable_dropout(model):
    """
    Enable dropout layers during inference (test time).
    This function will set all dropout layers to train mode
    so dropout remains active during multiple forward passes.
    """
    for module in model.modules():
        if module.__class__.__name__.startswith('Dropout'):
            module.train()


def mc_dropout_predict(model, dataloader, device, num_forward_passes=20):
    """
    Estimates epistemic uncertainty via Monte Carlo Dropout by performing
    multiple forward passes with dropout enabled, then aggregating results.

    Args:
        model (torch.nn.Module): Trained PyTorch model (with dropout layers).
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset
                                                  we want predictions on.
        device (torch.device): Device to run the forward passes on.
        num_forward_passes (int): Number of stochastic passes through the model.

    Returns:
        predictions_mean (np.ndarray): shape (N, C), where N is the number
                                       of samples and C is number of classes,
                                       representing the mean predicted probability.
        predictions_std  (np.ndarray): shape (N, C), standard deviation
                                       of predicted probabilities across MC samples.
        true_labels      (np.ndarray): ground-truth labels (if available).
    """
    model.eval()             # Put model in eval mode for layers like BN
    enable_dropout(model)    # Force dropout layers to remain active

    all_preds  = []  # Will hold predicted probabilities for each forward pass
    all_labels = []

    with torch.no_grad():
        for _ in range(num_forward_passes):
            preds_for_pass = []
            labels_for_pass = []

            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                # Convert logits to probabilities
                probs = torch.softmax(outputs, dim=1)
                preds_for_pass.append(probs.cpu().numpy())
                labels_for_pass.append(labels.cpu().numpy())

            # Concatenate for this pass
            preds_for_pass = np.concatenate(preds_for_pass, axis=0)
            all_preds.append(preds_for_pass)

            if len(all_labels) == 0:
                # Collect ground-truth only once (assuming order is consistent)
                all_labels = np.concatenate(labels_for_pass, axis=0)

    # Convert list of arrays => shape = (num_forward_passes, N, C)
    all_preds = np.stack(all_preds, axis=0)

    # Mean & Std over the MC dimension
    predictions_mean = np.mean(all_preds, axis=0)  # (N, C)
    predictions_std  = np.std(all_preds,  axis=0)  # (N, C)

    return predictions_mean, predictions_std, all_labels


def plot_uncertainty_distribution(predictions_mean, predictions_std, bins=50, title="Uncertainty Distribution"):
    """
    Plots a histogram of per-sample uncertainty, defined as the standard deviation
    of the predicted probability for the *predicted class*.

    Args:
        predictions_mean (np.ndarray): shape (N, C), mean predicted probabilities.
        predictions_std  (np.ndarray): shape (N, C), std dev of predicted probabilities.
        bins (int): Number of histogram bins.
        title (str): Plot title.
    """
    # predicted_class => argmax over mean probabilities
    predicted_class = np.argmax(predictions_mean, axis=1)

    # Gather standard deviation for each sample's predicted class
    sample_uncertainties = []
    for i, c in enumerate(predicted_class):
        sample_uncertainties.append(predictions_std[i, c])

    sample_uncertainties = np.array(sample_uncertainties)

    # Plot histogram
    plt.figure(figsize=(6,4))
    plt.hist(sample_uncertainties, bins=bins, color='skyblue', edgecolor='black')
    plt.xlabel("Std Dev (Predicted Class)")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Optionally print some stats
    print(f"Mean uncertainty: {sample_uncertainties.mean():.4f}")
    print(f"Std uncertainty:  {sample_uncertainties.std():.4f}")
    print(f"Max uncertainty:  {sample_uncertainties.max():.4f}")


def plot_sample_prediction(predictions_mean, predictions_std, sample_idx=0, class_names=None):
    """
    Plots a bar chart showing the mean predicted probability for each class
    (for a single sample), along with an error bar for the std dev.

    Args:
        predictions_mean (np.ndarray): shape (N, C) of mean predicted probabilities.
        predictions_std  (np.ndarray): shape (N, C) of std dev for predicted probabilities.
        sample_idx (int): Which sample index to visualize.
        class_names (list): Optional list of class names. If None, use [0, 1, 2, ...].
    """
    num_classes = predictions_mean.shape[1]
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    mean_probs = predictions_mean[sample_idx]
    std_probs  = predictions_std[sample_idx]

    plt.figure(figsize=(6,4))
    plt.bar(range(num_classes), mean_probs, yerr=std_probs, capsize=5, alpha=0.7, color='orange')
    plt.xticks(range(num_classes), class_names)
    plt.ylim([0, 1])
    plt.xlabel("Class")
    plt.ylabel("Predicted Probability")
    plt.title(f"Sample {sample_idx} - MC Dropout Prediction")
    plt.grid(True, axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
