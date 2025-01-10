# utils/analysis.py

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

def get_predictions(model, dataloader, device):
    """
    Runs the model on the provided dataloader and returns the predicted probabilities and true labels.
    
    Args:
        model (torch.nn.Module): Trained model to use for predictions.
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
        device (torch.device): Device to perform computations on.
    
    Returns:
        np.ndarray, np.ndarray: Array of predicted probabilities for Li6 and true labels.
    """
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            if outputs.shape[1] == 1:
                # Binary classification with a single output (sigmoid)
                probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
                probs_li6 = probs  # Assuming output is P(Li6)
            else:
                # Binary classification with two outputs (softmax)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                probs_li6 = probs[:, 0]  # Assuming class 0 is Li6
            
            all_probs.extend(probs_li6)
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_probs), np.array(all_labels)

def evaluate_model_with_fixed_threshold(model, model_name, dataloader, device, threshold=0.4):
    """
    Evaluates a model on a dataset using a fixed probability threshold for classification.
    Plots the confusion matrix and probability distributions.
    
    Args:
        model (torch.nn.Module): Trained model to evaluate.
        model_name (str): Name of the model (for plotting titles).
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
        device (torch.device): Device to perform computations on.
        threshold (float, optional): Fixed probability threshold for classification. Defaults to 0.4.
    
    Returns:
        None
    """
    # Get predictions and true labels
    probabilities_li6, true_labels = get_predictions(model, dataloader, device)
    
    # Classify based on the fixed threshold
    # Ensure consistency: 0=Li6, 1=Po
    predicted_labels = np.where(probabilities_li6 > threshold, 0, 1)  # 0=Li6, 1=Po based on threshold
    
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Li6', 'Po'])
    
    # Plot Confusion Matrix
    plt.figure(figsize=(6,5))
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f'Confusion Matrix: {model_name} (Threshold={threshold})')
    plt.show()
    
    # Plot Probability Distributions with threshold
    plt.figure(figsize=(10,6))
    sns.histplot(probabilities_li6[true_labels == 0], color='blue', label='Li6', kde=True, stat="density", bins=30, alpha=0.6)
    sns.histplot(probabilities_li6[true_labels == 1], color='red', label='Po', kde=True, stat="density", bins=30, alpha=0.6)
    plt.axvline(threshold, color='green', linestyle='--', label=f'Threshold = {threshold}')
    plt.title(f'Probability Distributions for {model_name}')
    plt.xlabel('P(Li6)')
    plt.ylabel('Density')
    plt.legend()
    plt.show()
    
    # Print threshold and contamination info
    po_mislabeled = np.sum((probabilities_li6 > threshold) & (true_labels == 1))
    total_po = np.sum(true_labels == 1)
    contamination = (po_mislabeled / total_po) * 100 if total_po > 0 else 0
    print(f"{model_name} Threshold: {threshold:.2f}")
    print(f"Po Mislabeled as Li6: {po_mislabeled}/{total_po} ({contamination:.2f}%)")
    print("-" * 60)

def evaluate_model_on_balanced_set_with_fixed_threshold(model, model_name, balanced_loader, device, threshold=0.4):
    """
    Evaluates a model on a balanced dataset using a fixed probability threshold for classification.
    Plots the confusion matrix and probability distributions, and prints class counts.
    
    Args:
        model (torch.nn.Module): Trained model to evaluate.
        model_name (str): Name of the model (for plotting titles).
        balanced_loader (torch.utils.data.DataLoader): DataLoader for the balanced dataset.
        device (torch.device): Device to perform computations on.
        threshold (float, optional): Fixed probability threshold for classification. Defaults to 0.4.
    
    Returns:
        None
    """
    # Get predictions and true labels
    probabilities_li6, true_labels = get_predictions(model, balanced_loader, device)
    
    # Classify based on the fixed threshold
    # Ensure consistency: 0=Li6, 1=Po
    predicted_labels = np.where(probabilities_li6 > threshold, 0, 1)  # 0=Li6, 1=Po based on threshold
    
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Li6', 'Po'])
    
    # Plot Confusion Matrix
    plt.figure(figsize=(6,5))
    disp.plot(cmap='Blues', values_format='d')
    plt.title(f'Confusion Matrix (Balanced Dataset): {model_name} (Threshold={threshold})')
    plt.show()
    
    # Plot Probability Distributions with threshold
    plt.figure(figsize=(10,6))
    sns.histplot(probabilities_li6[true_labels == 0], color='blue', label='Li6', kde=True, stat="density", bins=30, alpha=0.6)
    sns.histplot(probabilities_li6[true_labels == 1], color='red', label='Po', kde=True, stat="density", bins=30, alpha=0.6)
    plt.axvline(threshold, color='green', linestyle='--', label=f'Threshold = {threshold}')
    plt.title(f'Probability Distributions (Balanced Dataset): {model_name}')
    plt.xlabel('P(Li6)')
    plt.ylabel('Density')
    plt.legend()
    plt.show()
    
    # Print counts
    total_li6 = np.sum(true_labels == 0)
    total_po = np.sum(true_labels == 1)
    predicted_li6 = np.sum(predicted_labels == 0)
    predicted_po = np.sum(predicted_labels == 1)
    
    print(f"{model_name} on Balanced Dataset:")
    print(f"True Li6: {total_li6}, True Po: {total_po}")
    print(f"Predicted Li6: {predicted_li6}, Predicted Po: {predicted_po}")
    print("-" * 60)

def create_balanced_dataset(li6_data, po_data, num_samples_each=None):
    """
    Creates a balanced dataset with 50% Li6 and 50% Po.
    
    Args:
        li6_data (np.ndarray): Li6 waveform data.
        po_data (np.ndarray): Po waveform data.
        num_samples_each (int, optional): Number of samples for each class. 
                                          If None, uses the minimum available.
    
    Returns:
        np.ndarray, np.ndarray: Combined data and labels.
    """
    if num_samples_each is None:
        num_samples_each = min(len(li6_data), len(po_data))
    
    # Randomly select samples
    li6_indices = np.random.choice(len(li6_data), num_samples_each, replace=False)
    po_indices = np.random.choice(len(po_data), num_samples_each, replace=False)
    
    balanced_li6 = li6_data[li6_indices]
    balanced_po = po_data[po_indices]
    
    # Create labels
    li6_labels = np.zeros(num_samples_each, dtype=np.int64)  # 0=Li6
    po_labels = np.ones(num_samples_each, dtype=np.int64)    # 1=Po
    
    # Combine data and labels
    balanced_data = np.concatenate([balanced_li6, balanced_po], axis=0)
    balanced_labels = np.concatenate([li6_labels, po_labels], axis=0)
    
    return balanced_data, balanced_labels

def create_full_dataset(li6_data, po_data):
    """
    Creates a full dataset combining all Li6 and Po data.
    
    Args:
        li6_data (np.ndarray): Li6 waveform data.
        po_data (np.ndarray): Po waveform data.
    
    Returns:
        np.ndarray, np.ndarray: Combined data and labels.
    
    Explanation:
        We combine the Li6 & Po waveforms, but for deep learning below,
        we typically rely on the custom 'ScintillationDataset' rather than a single array.
    """
    # Create labels for the full dataset
    li6_labels = np.zeros(len(li6_data), dtype=np.int64)  # 0=Li6
    po_labels  = np.ones(len(po_data), dtype=np.int64)    # 1=Po
    
    # Combine data and labels
    full_data   = np.concatenate([li6_data, po_data], axis=0)
    full_labels = np.concatenate([li6_labels, po_labels], axis=0)
    
    return full_data, full_labels

class BalancedScintillationDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.int64)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

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