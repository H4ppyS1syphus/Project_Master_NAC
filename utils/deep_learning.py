"""
deep_learning.py

Contains the PyTorch dataset class, MultiBranchCNN definition, and training loop.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

class ScintillationDataset(Dataset):
    def __init__(self, Liw, Pow):
        # Liw, Pow => shape (N, 4, T)
        data_full = np.concatenate([Liw, Pow], axis=0)
        labels_full = np.concatenate([
            np.zeros(len(Liw)), np.ones(len(Pow))
        ], axis=0)

        self.data = torch.tensor(data_full, dtype=torch.float32)
        self.labels = torch.tensor(labels_full, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class MultiBranchCNN(nn.Module):
    def __init__(self, input_length):
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

        self.fc = nn.Sequential(
            nn.Linear(8 * (input_length // 2) * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        """
        x: shape (batch_size, 4, input_length)
        Each branch -> shape (batch_size, 1, input_length)
        """
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

        out = self.fc(combined)
        return out

def train_and_evaluate(
    model,
    train_loader,
    test_loader,
    num_epochs,
    learning_rate,
    device
):
    """
    Standard training loop for the MultiBranchCNN.
    """
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    for epoch in range(num_epochs):
        # 1) Training Phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        train_loss = running_loss / len(train_loader)
        train_acc  = correct_train / total_train
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # 2) Evaluation Phase
        model.eval()
        test_loss = 0.0
        correct_test = 0
        total_test = 0
        all_labels = []
        all_preds  = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                test_loss += loss.item()

                _, preds = torch.max(outputs, 1)
                correct_test += (preds == labels).sum().item()
                total_test += labels.size(0)

                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        test_loss = test_loss / len(test_loader)
        test_acc  = correct_test / total_test
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "models/multi_cnn.pth")

    # Plot train vs test loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs+1), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Test Loss Over Epochs')
    plt.legend()
    plt.grid()
    plt.show()

    # Confusion Matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix (Test Set)")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
