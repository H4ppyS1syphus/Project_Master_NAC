"""
deep_learning.py

Contains PyTorch neural network classes and training utilities for:
1) SingleBranchCNN (a simple 2D CNN)
2) MultiBranchCNN (4 parallel 1D CNN branches)
3) TwoDConvNetWithFullAttention (2D CNN + Self-Attention)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

##############################################################################
# 1) SingleBranchCNN: 2D CNN treating shape => (1,4,time) as an "image".
##############################################################################
class SingleBranchCNN(nn.Module):
    """
    A simple 2D CNN that interprets the input as (batch,1,4,time).
    Good for baseline deep learning classification.
    """
    def __init__(self, num_classes=2, time_length=2000):
        super(SingleBranchCNN, self).__init__()
        # Example architecture; tune as needed
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(3,3), padding=(1,1))
        self.bn1   = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3,3), padding=(1,1))
        self.bn2   = nn.BatchNorm2d(16)
        self.pool  = nn.MaxPool2d((2,2))  # reduce dimension
        self.dropout = nn.Dropout(p=0.5)
        self.relu  = nn.ReLU()

        # Suppose we do 2 conv layers, each halving the "image" dimension in time,
        # so final shape ~ (16,2, time_length//2) if height=4 => after one pool => height=2
        out_c = 16
        out_h = 2
        out_w = time_length // 2
        in_features = out_c * out_h * out_w

        self.fc1 = nn.Linear(in_features, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x => (batch,1,4,time)
        x = self.relu(self.bn1(self.conv1(x)))   # => (batch,8,4,time)
        x = self.pool(self.relu(self.bn2(self.conv2(x))))  # => (batch,16,2,time//2)
        x = self.dropout(x)
        # flatten
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits


##############################################################################
# 2) MultiBranchCNN: one 1D CNN per channel
##############################################################################
class MultiBranchCNN(nn.Module):
    """
    A 1D CNN for each channel (shape => (batch,4,time)),
    then concatenates the outputs for final classification.
    """
    def __init__(self, input_length=2000, num_classes=2):
        super(MultiBranchCNN, self).__init__()
        
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

        # After MaxPool1d(kernel_size=2), each branch => shape (batch,8, input_length//2)
        # We'll flatten => 8*(input_length//2) from each branch => times 4 branches => total
        self.fc = nn.Sequential(
            nn.Linear(8 * (input_length // 2) * 4, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x => shape (batch,4,input_length)
        x1 = x[:, 0:1, :]
        x2 = x[:, 1:2, :]
        x3 = x[:, 2:3, :]
        x4 = x[:, 3:4, :]

        out1 = self.branch1(x1)
        out2 = self.branch2(x2)
        out3 = self.branch3(x3)
        out4 = self.branch4(x4)

        combined = torch.cat([
            out1.view(out1.size(0), -1),
            out2.view(out2.size(0), -1),
            out3.view(out3.size(0), -1),
            out4.view(out4.size(0), -1)
        ], dim=1)

        logits = self.fc(combined)
        return logits


##############################################################################
# 3) TwoDConvNetWithFullAttention: 2D CNN + Self-Attention
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
        self.key    = nn.Linear(feature_size, feature_size)
        self.query  = nn.Linear(feature_size, feature_size)
        self.value  = nn.Linear(feature_size, feature_size)

    def forward(self, x, mask=None):
        """
        x => shape (batch, time, feature_size)
        """
        B, T, F = x.shape
        keys    = self.key(x)      # (B, T, F)
        queries = self.query(x)    # (B, T, F)
        values  = self.value(x)    # (B, T, F)

        # scaled dot-product
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / (F ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)  # (B, T, T)
        output = torch.matmul(attention_weights, values)  # (B, T, F)
        return output, attention_weights


class TwoDConvNetWithFullAttention(nn.Module):
    """
    2D CNN that keeps the time dimension, then applies a self-attention layer.
    """
    def __init__(self, num_classes=2, feature_size=64):
        super(TwoDConvNetWithFullAttention, self).__init__()
        # 2D Conv stack
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(4, 4), padding=(0, 1))
        self.bn1   = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(8, 16, kernel_size=(1, 4), padding=(0, 1))
        self.bn2   = nn.BatchNorm2d(16)

        self.conv3 = nn.Conv2d(16, 32, kernel_size=(1, 4), padding=(0, 1))
        self.bn3   = nn.BatchNorm2d(32)

        self.pool  = nn.MaxPool2d(kernel_size=(1,2))
        self.dropout = nn.Dropout(p=0.5)
        self.relu  = nn.ReLU()

        # Self-Attention
        self.attention = SelfAttentionLayer(feature_size=32)

        # Fully-connected
        self.fc1 = nn.Linear(32, feature_size)
        self.fc2 = nn.Linear(feature_size, num_classes)

    def forward(self, x):
        """
        x => shape (batch,1,4,width)
        """
        x = self.relu(self.bn1(self.conv1(x)))  # => (batch,8,1,width')
        x = self.relu(self.bn2(self.conv2(x)))  # => (batch,16,1,width'')
        x = self.relu(self.bn3(self.conv3(x)))  # => (batch,32,1,width''')
        x = self.pool(x)                        # => (batch,32,1,width_reduced)
        x = self.dropout(x)

        # flatten out height=1 => (batch,32,width_reduced)
        x = x.squeeze(2)  # => (batch,32,width_reduced)
        x = x.transpose(1, 2)  # => (batch,width_reduced,32)

        # Apply attention => (batch,width_reduced,32)
        x_attn, _ = self.attention(x)

        # average pool across time => (batch,32)
        x_pooled = x_attn.mean(dim=1)

        # classification
        x_pooled = self.relu(self.fc1(x_pooled))
        logits   = self.fc2(x_pooled)
        return logits


##############################################################################
# 4) Standard Training Loop
##############################################################################
def train_model(
    model, train_loader, test_loader,
    num_epochs=10, learning_rate=1e-3,
    device=torch.device("cpu")
):
    """
    Generic training loop for any of the CNN models above.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    for epoch in range(num_epochs):
        # Training
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc  = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Testing
        model.eval()
        test_loss, correct_test, total_test = 0.0, 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct_test += (preds == labels).sum().item()
                total_test += labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_loss = test_loss / len(test_loader)
        test_acc  = correct_test / total_test
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Test  Loss: {test_loss:.4f}, Test  Acc: {test_acc:.4f}")

    # Final confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix (Test Set)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()

    return model
