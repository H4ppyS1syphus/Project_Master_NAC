"""
uncertainty.py

Utilities for estimating epistemic uncertainty in PyTorch models using
Monte Carlo (MC) Dropout.
"""

import torch
import numpy as np


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
        predictions_mean (np.ndarray): Array of shape (N, C), where N is the number
                                       of samples and C is number of classes,
                                       representing the mean predicted probability.
        predictions_std  (np.ndarray): Array of shape (N, C), standard deviation
                                       of predicted probabilities across MC samples.
        true_labels      (np.ndarray): Ground-truth labels (if available).
    """

    model.eval()  # Put model in eval mode first (for BN, etc.)
    enable_dropout(model)  # Force dropout layers to remain active

    all_preds = []  # Will hold predicted probabilities for each forward pass
    all_labels = []

    with torch.no_grad():
        for _ in range(num_forward_passes):
            # For each pass, we re-iterate over the dataloader
            preds_for_pass = []
            labels_for_pass = []

            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                # Convert to probabilities (if it's raw logits)
                probs = torch.softmax(outputs, dim=1)
                preds_for_pass.append(probs.cpu().numpy())
                labels_for_pass.append(labels.cpu().numpy())

            # Concatenate for this pass
            preds_for_pass = np.concatenate(preds_for_pass, axis=0)
            all_preds.append(preds_for_pass)

            if len(all_labels) == 0:
                # Collect ground-truth only once
                all_labels = np.concatenate(labels_for_pass, axis=0)

    # Convert list of arrays => shape = (num_forward_passes, N, C)
    all_preds = np.stack(all_preds, axis=0)

    # Mean & Std over the MC dimension
    predictions_mean = np.mean(all_preds, axis=0)  # (N, C)
    predictions_std  = np.std(all_preds, axis=0)   # (N, C)

    return predictions_mean, predictions_std, all_labels
