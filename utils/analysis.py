"""
analysis.py

Utility functions for:
1) Finding an optimal threshold to cap Po contamination at 5%.
2) Classifying an unlabeled dataset (e.g., Phys) to estimate Li6 vs Po counts.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def find_threshold_for_contamination(
    probabilities_li6, 
    labels, 
    max_po_contamination=0.05,
    plot=True
):
    """
    Finds a probability threshold that caps the contamination of Po signals at a desired level
    and optionally plots the confusion matrix and probability distributions.
    
    Args:
        probabilities_li6 (np.ndarray): Shape (N,) => Predicted probability that a sample is Li6.
        labels (np.ndarray): Shape (N,) => True labels (0=Li6, 1=Po).
        max_po_contamination (float, optional): Maximum allowed fraction of Po mislabeled as Li6. Defaults to 0.05.
        plot (bool, optional): Whether to plot the confusion matrix and probability distributions. Defaults to True.
    
    Returns:
        float: The threshold T such that only 'max_po_contamination' fraction of Po gets classified as Li6.
               If no threshold can achieve that, returns None.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    
    # 1) Sort all probabilities in descending order so we can sweep thresholds
    sort_indices = np.argsort(probabilities_li6)[::-1]
    sorted_probs = probabilities_li6[sort_indices]
    sorted_labels = labels[sort_indices]
    
    # Count total Po
    total_po = np.sum(sorted_labels == 1)
    if total_po == 0:
        print("Warning: No Po samples in dataset. Returning threshold=0.5 by default.")
        threshold = 0.5
        if plot:
            # Plotting the probability distribution
            plt.figure(figsize=(8,6))
            sns.histplot(probabilities_li6[labels == 0], color='blue', label='Li6', kde=True, stat="density", bins=30, alpha=0.6)
            plt.title('Probability Distribution for Li6')
            plt.xlabel('P(Li6)')
            plt.ylabel('Density')
            plt.legend()
            plt.show()
        return threshold
    
    # 2) Sweep down from highest to lowest probability
    #    For each unique prob, see how many Po are classified as Li6
    misclassified_po_count = 0
    threshold = None
    for i, prob in enumerate(sorted_probs):
        lbl = sorted_labels[i]
        if lbl == 1:
            misclassified_po_count += 1
        
        fraction_po_mis = misclassified_po_count / total_po
        if fraction_po_mis > max_po_contamination:
            idx_threshold = i - 1 if i > 0 else 0
            threshold = sorted_probs[idx_threshold]
            break
    
    # If no threshold found that satisfies the contamination level
    if threshold is None:
        threshold = sorted_probs[-1]
        print(f"No threshold found that limits Po contamination to {max_po_contamination*100}%. Using the lowest probability as threshold.")
    
    # 3) Apply the threshold to get predicted labels
    predicted_labels = (probabilities_li6 <= threshold).astype(int)  # 0=Li6, 1=Po
    
    if plot:
        # Compute Confusion Matrix
        cm = confusion_matrix(labels, predicted_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Li6', 'Po'])
        
        plt.figure(figsize=(6,5))
        disp.plot(cmap='Blues', values_format='d')
        plt.title(f'Confusion Matrix (Threshold={threshold:.4f})')
        plt.show()
        
        # Plot Probability Distributions
        plt.figure(figsize=(10,6))
        sns.histplot(probabilities_li6[labels == 0], color='blue', label='Li6', kde=True, stat="density", bins=30, alpha=0.6)
        sns.histplot(probabilities_li6[labels == 1], color='red', label='Po', kde=True, stat="density", bins=30, alpha=0.6)
        plt.axvline(threshold, color='green', linestyle='--', label=f'Threshold = {threshold:.4f}')
        plt.title('Probability Distributions for Li6 and Po')
        plt.xlabel('P(Li6)')
        plt.ylabel('Density')
        plt.legend()
        plt.show()
        
        # Optional: Plot Contamination vs Threshold
        # This plot shows how the contamination changes as the threshold varies.
        contamination_rates = []
        thresholds = sorted_probs
        for i, prob in enumerate(sorted_probs):
            lbl = sorted_labels[i]
            if lbl == 1:
                misclassified_po_count += 1
            contamination = misclassified_po_count / total_po
            contamination_rates.append(contamination)
        
        plt.figure(figsize=(10,6))
        plt.plot(sorted_probs, contamination_rates, label='Po Contamination')
        plt.axhline(y=max_po_contamination, color='r', linestyle='--', label=f'Max Contamination = {max_po_contamination*100}%')
        plt.axvline(x=threshold, color='g', linestyle='--', label=f'Threshold = {threshold:.4f}')
        plt.xlabel('Threshold')
        plt.ylabel('Po Contamination Rate')
        plt.title('Po Contamination vs Threshold')
        plt.legend()
        plt.show()
    
    return threshold

def plot_histograms_and_confusion_matrices(probabilities_li6, labels, threshold):
    """
    Plots histograms of probabilities and confusion matrices for given thresholds.

    Args:
        probabilities_li6 (np.ndarray): Predicted probabilities for Li6.
        labels (np.ndarray): True labels (0=Li6, 1=Po).
        threshold (float): Computed threshold for classification.
    """
    # Plot histograms
    plt.figure(figsize=(12, 6))
    plt.hist(probabilities_li6[labels == 0], bins=50, alpha=0.5, label="Li6", color="blue")
    plt.hist(probabilities_li6[labels == 1], bins=50, alpha=0.5, label="Po", color="red")
    plt.axvline(threshold, color="green", linestyle="--", label=f"Threshold (Found) = {threshold:.2f}")
    plt.axvline(0.5, color="orange", linestyle="--", label="Threshold = 0.5")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Frequency")
    plt.title("Histogram of Predicted Probabilities")
    plt.legend()
    plt.show()

    # Compute and display confusion matrices
    for thr, name in [(0.5, "Threshold = 0.5"), (threshold, f"Threshold = {threshold:.2f}")]:
        preds = (probabilities_li6 >= thr).astype(int)
        cm = confusion_matrix(labels, preds, labels=[0, 1])
        disp = ConfusionMatrixDisplay(cm, display_labels=["Li6", "Po"])
        disp.plot(cmap="Blues")
        plt.title(name)
        plt.show()

def classify_dataset(
    model, 
    waveforms, 
    device, 
    batch_size=64, 
    threshold=0.5
):
    """
    Classifies a new dataset using a specified probability threshold for Li6.

    Args:
        model (torch.nn.Module): Trained model (output shape [N, 2] for 2 classes).
        waveforms (np.ndarray): shape (N, 4, T) => waveforms to classify.
        device (torch.device): Device to use (CPU or GPU).
        batch_size (int): Batch size for the DataLoader.
        threshold (float): Probability threshold for labeling a sample as Li6.

    Returns:
        np.ndarray: Predicted labels (0=Li6, 1=Po) for each sample.
    """
    from torch.utils.data import TensorDataset, DataLoader
    
    # Convert waveforms to Tensor, no ground truth labels
    data_tensor = torch.tensor(waveforms, dtype=torch.float32)
    # Expand dims => shape (N,1,4,T)
    data_tensor = data_tensor.unsqueeze(1)
    
    dataset = TensorDataset(data_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    predicted_labels = []

    with torch.no_grad():
        for (batch_x,) in loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)         # shape => (B, 2)
            probs  = torch.softmax(logits, dim=1)  # shape => (B, 2)
            probs_li6 = probs[:, 0]         # Probability of Li6

            # Compare with threshold => if > threshold => label=0 (Li6), else=1 (Po)
            batch_preds = (probs_li6 <= threshold).long().cpu().numpy()
            predicted_labels.extend(batch_preds)

    return np.array(predicted_labels)

