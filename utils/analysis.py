"""
analysis.py

Utility functions for:
1) Finding an optimal threshold to cap Po contamination at 5%.
2) Classifying an unlabeled dataset (e.g., Phys) to estimate Li6 vs Po counts.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

def find_threshold_for_contamination(
    probabilities_li6, 
    labels, 
    max_po_contamination=0.05
):
    """
    Finds a probability threshold that caps the contamination of Po signals at a desired level.

    Args:
        probabilities_li6 (np.ndarray): shape (N,) => predicted probability that a sample is Li6
        labels (np.ndarray): shape (N,) => true labels (0=Li6, 1=Po)
        max_po_contamination (float): Maximum allowed fraction of Po mislabeled as Li6.

    Returns:
        float: The threshold T such that only 'max_po_contamination' fraction of Po gets classified as Li6.
               If no threshold can achieve that, returns None.
    """
    # 1) Sort all probabilities in descending order so we can sweep thresholds
    sort_indices = np.argsort(probabilities_li6)[::-1]
    sorted_probs = probabilities_li6[sort_indices]
    sorted_labels = labels[sort_indices]

    # Count total Po
    total_po = np.sum(sorted_labels == 1)
    if total_po == 0:
        print("Warning: no Po samples in dataset. Returning threshold=0.5 by default.")
        return 0.5
    
    # 2) Sweep down from highest to lowest probability
    #    For each unique prob, see how many Po are classified as Li6
    #    fraction of misclassified Po = (#Po with prob>threshold) / total_po
    misclassified_po_count = 0
    for i, prob in enumerate(sorted_probs):
        lbl = sorted_labels[i]
        # If it's Po, at or above the current prob => mislabeled as Li6
        if lbl == 1:
            misclassified_po_count += 1
        
        fraction_po_mis = misclassified_po_count / total_po
        # Once we exceed the contamination level, the threshold is the prob of the last point
        if fraction_po_mis > max_po_contamination:
            # The threshold is just below this prob
            # We can take threshold = sorted_probs[i] if we want strictly less contamination
            # or threshold = sorted_probs[i+1] if we want to accept this point
            idx_threshold = i - 1 if i > 0 else 0
            threshold = sorted_probs[idx_threshold]
            return threshold
    
    # If we never exceed max_po_contamination, means we can accept even the lowest prob
    # => contamination is never above 5%, so threshold can be the min prob
    return sorted_probs[-1]


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
            batch_preds = (probs_li6 > threshold).long().cpu().numpy()
            predicted_labels.extend(batch_preds)

    return np.array(predicted_labels)
