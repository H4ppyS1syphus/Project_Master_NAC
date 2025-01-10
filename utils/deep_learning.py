"""
deep_learning.py

Contains PyTorch neural network classes and training utilities for:
1) ScintillationDataset  (custom dataset for Li/Po data)
2) TwoDConvNet           (a simple 2D CNN, single branch)
3) MultiBranchCNN        (4 parallel 1D CNN branches)
4) TwoDConvNetWithFullAttention (2D CNN + Self-Attention)
5) train_one_epoch, evaluate    (training & evaluation loops)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

##############################################################################
# 1) Dataset: ScintillationDataset
##############################################################################
class ScintillationDataset(Dataset):
    """
    Merges Liw (label=0) and Pow (label=1) arrays of shape (N,4,timepoints),
    and stores them for classification.
    """
    def __init__(self, Liw, Pow):
        """
        Args:
            Liw (np.ndarray): shape (N_Li, 4, T)
            Pow (np.ndarray): shape (N_Po, 4, T)
        """
        data_full = np.concatenate([Liw, Pow], axis=0)  # (N_Li+N_Po, 4, T)
        labels_full = np.concatenate([
            np.zeros(len(Liw)), 
            np.ones(len(Pow))
        ], axis=0)

        self.data = torch.tensor(data_full, dtype=torch.float32)
        self.labels = torch.tensor(labels_full, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns:
            waveforms: shape (1, 4, T), as a FloatTensor
            label: shape (), LongTensor (0 or 1)
        """
        waveforms = self.data[idx]          # (4, T)
        waveforms = waveforms.unsqueeze(0)  # (1, 4, T)
        label     = self.labels[idx]
        return waveforms, label
##############################################################################
# 2) TwoDConvNet: single-branch 2D CNN
##############################################################################
class TwoDConvNet(nn.Module):
    """
    A smaller version of the TwoDConvNet model for more efficient computation.
    Reduced the number of convolutional layers and filters.
    """
    def __init__(self, num_classes=2, height=4, width=2000):
        super().__init__()
        # Input shape: (batch, 1, 4, width)
        
        # Single convolutional layer with fewer filters
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(3, 3), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(4)

        # Optional second convolutional layer (commented out to reduce size)
        # self.conv2 = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=(3, 3), padding=(1, 1))
        # self.bn2 = nn.BatchNorm2d(8)

        self.pool = nn.MaxPool2d(kernel_size=(2, 2))  # Reduce both dimensions by half
        self.dropout = nn.Dropout(p=0.3)  # Lower dropout rate for smaller model
        self.relu = nn.ReLU()

        # Calculate the number of input features for the first fully connected layer
        out_c = 4  # Reduced number of channels
        out_h = height // 2  # After pooling
        out_w = width // 2
        in_features = out_c * out_h * out_w

        # Fully connected layers with fewer units
        self.fc1 = nn.Linear(in_features, 128)  # Reduced number of hidden units
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Input: (batch, 1, 4, width)
        x = self.conv1(x)  # => (batch, 4, 4, width)
        x = self.bn1(x)
        x = self.relu(x)

        # Optional second conv layer
        # x = self.conv2(x)  # => (batch, 8, 4, width)
        # x = self.bn2(x)
        # x = self.relu(x)

        x = self.pool(x)  # => (batch, 4, 2, width//2)
        x = self.dropout(x)

        # Flatten
        x = x.view(x.size(0), -1)  # => (batch, 4 * 2 * (width//2))

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

##############################################################################
# 3) MultiBranchCNN: one 1D CNN per channel
##############################################################################
# utils/deep_learning.py
class MultiBranchCNN(nn.Module):
    """
    A 1D CNN for each channel (4 detectors), then concatenates the outputs.
    """
    def __init__(self, input_length=800, num_classes=2):
        super().__init__()
        
        # Define four separate branches, one for each channel
        self.branch1 = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.branch4 = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )

        # Calculate the number of input features for the first fully connected layer
        # Each branch outputs (batch_size, 8, input_length//2)
        # After flattening and concatenating 4 branches: 8 * (input_length//2) * 4
        fc_input_dim = 8 * (input_length // 2) * 4

        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        Forward pass for MultiBranchCNN.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 4, 800)
        
        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes)
        """
        # Check if input has shape [batch_size, 1, 4, 800]
        if x.dim() == 4 and x.size(1) == 1:
            x = x.squeeze(1)  # Remove the channel dimension: [batch_size, 4, 800]
        elif x.dim() != 3:
            raise ValueError(f"Expected input tensor to be 3D after squeezing, but got shape {x.shape}")

        # Split the input into four separate channels
        x1 = x[:, 0:1, :]  # [batch_size, 1, 800]
        x2 = x[:, 1:2, :]
        x3 = x[:, 2:3, :]
        x4 = x[:, 3:4, :]

        # Pass each channel through its respective branch
        out1 = self.branch1(x1)  # [batch_size, 8, 400]
        out2 = self.branch2(x2)
        out3 = self.branch3(x3)
        out4 = self.branch4(x4)

        # Flatten each branch's output
        out1_flat = out1.view(out1.size(0), -1)  # [batch_size, 3200]
        out2_flat = out2.view(out2.size(0), -1)
        out3_flat = out3.view(out3.size(0), -1)
        out4_flat = out4.view(out4.size(0), -1)

        # Concatenate all flattened outputs
        combined = torch.cat([out1_flat, out2_flat, out3_flat, out4_flat], dim=1)  # [batch_size, 12800]

        # Pass through fully connected layers
        logits = self.fc(combined)  # [batch_size, num_classes]
        return logits

##############################################################################
# 4) TwoDConvNetWithFullAttention: 2D CNN + Self-Attention
##############################################################################
class SelfAttentionLayer(nn.Module):
    """
    Standard Scaled Dot-Product Self-Attention.
    Input:  (batch, time, features)
    Output: (batch, time, features)
    """
    def __init__(self, feature_size):
        super().__init__()
        self.feature_size = feature_size
        self.key   = nn.Linear(feature_size, feature_size)
        self.query = nn.Linear(feature_size, feature_size)
        self.value = nn.Linear(feature_size, feature_size)

    def forward(self, x, mask=None):
        """
        x => shape (batch, time, feature_size)
        """
        B, T, feature_dim = x.shape
        keys    = self.key(x)
        queries = self.query(x)
        values  = self.value(x)

        # scaled dot-product
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / (feature_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)  # (B, T, T)
        output = torch.matmul(attention_weights, values)  # (B, T, feature_dim)
        return output, attention_weights


class TwoDConvNetWithFullAttention(nn.Module):
    """
    2D CNN that keeps the time dimension, then applies a self-attention layer.
    """
    def __init__(self, num_classes=2, feature_size=64):
        super().__init__()  # Correct Python 3 syntax for super()
        # 2D CNN stack
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(4, 4), padding=(0,1))
        self.bn1   = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=(1, 4), padding=(0,1))
        self.bn2   = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 32, kernel_size=(1, 4), padding=(0,1))
        self.bn3   = nn.BatchNorm2d(32)

        self.pool = nn.MaxPool2d(kernel_size=(1,2))
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

        # Self-attention
        self.attention = SelfAttentionLayer(feature_size=32)

        # Fully-connected layers
        self.fc1 = nn.Linear(32, feature_size)
        self.fc2 = nn.Linear(feature_size, num_classes)

    def forward(self, x):
        """
        Forward pass for TwoDConvNetWithFullAttention.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 4, 800)
        
        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes)
        """
        x = self.relu(self.bn1(self.conv1(x)))  # => (batch,8,1,797)
        x = self.relu(self.bn2(self.conv2(x)))  # => (batch,16,1,796)
        x = self.relu(self.bn3(self.conv3(x)))  # => (batch,32,1,795)
        x = self.pool(x)                        # => (batch,32,1,397)
        x = self.dropout(x)

        # Flatten height=1 => shape => (batch,32,397)
        x = x.squeeze(2)  # => (batch,32,397)
        x = x.transpose(1, 2)  # => (batch,397,32)

        # Self-attention => (batch,397,32)
        x_attn, _ = self.attention(x)

        # Average pooling over time => (batch,32)
        x_pooled = x_attn.mean(dim=1)

        # Classification
        x_pooled = self.relu(self.fc1(x_pooled))
        logits = self.fc2(x_pooled)
        return logits


##############################################################################
# 5) train_one_epoch & evaluate (lower-level utilities)
##############################################################################
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Single epoch training. Returns (train_loss, train_accuracy).
    """
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    epoch_loss = running_loss / len(dataloader)
    epoch_acc  = 100.0 * correct / total
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    """
    Evaluation loop. Returns (test_loss, test_accuracy, preds, labels).
    """
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss = running_loss / len(dataloader)
    test_acc  = 100.0 * correct / total
    return test_loss, test_acc, all_preds, all_labels


##############################################################################
# 6) Optional: wrapper that trains & evaluates across multiple epochs
##############################################################################
# deep_learning.py

def train_and_evaluate(model, train_loader, test_loader, 
                       num_epochs=10, learning_rate=1e-3, device=None):
    """
    Trains and evaluates the model over multiple epochs.
    Automatically plots the training/testing loss & accuracy,
    plus the confusion matrix at the end.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Lists to store metrics
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    all_preds, all_labels = None, None  # placeholders for final confusion matrix

    for epoch in range(1, num_epochs + 1):
        # 1) Training step
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        # 2) Testing step
        ts_loss, ts_acc, preds, labels = evaluate(
            model, test_loader, criterion, device
        )

        train_losses.append(tr_loss)
        test_losses.append(ts_loss)
        train_accs.append(tr_acc)
        test_accs.append(ts_acc)

        # Save these for final confusion matrix
        all_preds  = preds
        all_labels = labels

        print(f"[Epoch {epoch}/{num_epochs}] "
              f"Train Loss={tr_loss:.4f}, Acc={tr_acc:.2f}% | "
              f"Test Loss={ts_loss:.4f}, Acc={ts_acc:.2f}%")

    # -- Plot Loss and Accuracy Curves --
    plt.figure(figsize=(14, 5))

    # Subplot 1: Loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs+1), test_losses,  label='Test Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss over Epochs')
    plt.legend()
    plt.grid(True)

    # Subplot 2: Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs+1), train_accs, label='Train Accuracy', marker='o')
    plt.plot(range(1, num_epochs+1), test_accs,  label='Test Accuracy', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Testing Accuracy over Epochs')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # -- Final Confusion Matrix --
    if all_preds is not None and all_labels is not None:
        plot_confusion_matrix(
            true_labels=all_labels, 
            pred_labels=all_preds, 
            class_names=["Li6","Po"],
            title="Confusion Matrix (Final)"
        )

    # Return the model + any metrics you want to track
    metrics = {
        "train_losses": train_losses,
        "test_losses": test_losses,
        "train_accs": train_accs,
        "test_accs": test_accs
    }
    return model, metrics


def plot_confusion_matrix(true_labels, pred_labels, class_names=None, title="Confusion Matrix"):
    """
    Plots a confusion matrix using Seaborn heatmap.
    
    Args:
        true_labels (list or np.array): Ground-truth labels.
        pred_labels (list or np.array): Model-predicted labels.
        class_names (list): Optional list of class names for x/y axes.
        title (str): Plot title.
    """
    cm = confusion_matrix(true_labels, pred_labels)
    
    plt.figure(figsize=(5,4))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt="d", 
        cmap="Blues",
        xticklabels=class_names if class_names else ["Class 0", "Class 1"],
        yticklabels=class_names if class_names else ["Class 0", "Class 1"]
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()
