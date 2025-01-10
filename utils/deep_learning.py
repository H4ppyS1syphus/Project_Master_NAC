import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix


class ScintillationDataset(Dataset):
    def __init__(self, Liw, Pow):
        data_full = np.concatenate([Liw, Pow], axis=0)
        labels_full = np.concatenate([np.zeros(len(Liw)), np.ones(len(Pow))], axis=0)

        self.data = torch.tensor(data_full, dtype=torch.float32)
        self.labels = torch.tensor(labels_full, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        waveforms = self.data[idx].unsqueeze(0)
        label = self.labels[idx]
        return waveforms, label


class TwoDConvNet(nn.Module):
    def __init__(self, num_classes=2, height=4, width=2000):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=(3, 3), padding=(1, 1))
        self.bn1 = nn.BatchNorm2d(4)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout = nn.Dropout(p=0.3)
        self.relu = nn.ReLU()

        out_c = 4
        out_h = height // 2
        out_w = width // 2
        in_features = out_c * out_h * out_w

        self.fc1 = nn.Linear(in_features, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MultiBranchCNN(nn.Module):
    def __init__(self, input_length=800, num_classes=2):
        super().__init__()
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

        fc_input_dim = 8 * (input_length // 2) * 4
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.squeeze(1)
        x1, x2, x3, x4 = x[:, 0:1, :], x[:, 1:2, :], x[:, 2:3, :], x[:, 3:4, :]
        out1, out2, out3, out4 = self.branch1(x1), self.branch2(x2), self.branch3(x3), self.branch4(x4)
        combined = torch.cat([out1.view(out1.size(0), -1), out2.view(out2.size(0), -1), 
                              out3.view(out3.size(0), -1), out4.view(out4.size(0), -1)], dim=1)
        logits = self.fc(combined)
        return logits


class SelfAttentionLayer(nn.Module):
    def __init__(self, feature_size):
        super().__init__()
        self.key = nn.Linear(feature_size, feature_size)
        self.query = nn.Linear(feature_size, feature_size)
        self.value = nn.Linear(feature_size, feature_size)

    def forward(self, x, mask=None):
        B, T, feature_dim = x.shape
        keys = self.key(x)
        queries = self.query(x)
        values = self.value(x)

        scores = torch.matmul(queries, keys.transpose(-2, -1)) / (feature_dim ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, values)
        return output, attention_weights


class TwoDConvNetWithFullAttention(nn.Module):
    def __init__(self, num_classes=2, feature_size=64):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(4, 4), padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(1, 4), padding=(0, 1))
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(1, 4), padding=(0, 1))
        self.bn3 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=(1, 2))
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

        self.attention = SelfAttentionLayer(feature_size=32)
        self.fc1 = nn.Linear(32, feature_size)
        self.fc2 = nn.Linear(feature_size, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.dropout(x)
        x = x.squeeze(2).transpose(1, 2)
        x_attn, _ = self.attention(x)
        x_pooled = x_attn.mean(dim=1)
        x_pooled = self.relu(self.fc1(x_pooled))
        logits = self.fc2(x_pooled)
        return logits


def train_one_epoch(model, dataloader, optimizer, criterion, device):
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
        total += labels.size(0)

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
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
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss = running_loss / len(dataloader)
    test_acc = 100.0 * correct / total
    return test_loss, test_acc, all_preds, all_labels


def train_and_evaluate(model, train_loader, test_loader, num_epochs=10, learning_rate=1e-3, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_losses, test_losses = [], []
    train_accs, test_accs = [], []
    all_preds, all_labels = None, None

    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        ts_loss, ts_acc, preds, labels = evaluate(model, test_loader, criterion, device)

        train_losses.append(tr_loss)
        test_losses.append(ts_loss)
        train_accs.append(tr_acc)
        test_accs.append(ts_acc)

        all_preds = preds
        all_labels = labels

        print(f"[Epoch {epoch}/{num_epochs}] Train Loss={tr_loss:.4f}, Acc={tr_acc:.2f}% | Test Loss={ts_loss:.4f}, Acc={ts_acc:.2f}%")

    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), test_losses, label='Test Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss over Epochs')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accs, label='Train Accuracy', marker='o')
    plt.plot(range(1, num_epochs + 1), test_accs, label='Test Accuracy', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Testing Accuracy over Epochs')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    if all_preds is not None and all_labels is not None:
        plot_confusion_matrix(all_labels, all_preds, ["Li6", "Po"], "Confusion Matrix (Final)")

    metrics = {"train_losses": train_losses, "test_losses": test_losses, "train_accs": train_accs, "test_accs": test_accs}
    return model, metrics


def plot_confusion_matrix(true_labels, pred_labels, class_names=None, title="Confusion Matrix"):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()
