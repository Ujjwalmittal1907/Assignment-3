"""
Data Science Learning Journey Phase-II – Assignment 3
MNIST CNN Classification (28x28 grayscale)

Architecture (exactly as per requirement):
  Conv1: 32 filters, 3x3, ReLU -> MaxPool 2x2
  Conv2: 64 filters, 3x3, ReLU -> MaxPool 2x2
  Flatten -> FC 128 ReLU -> Output 10 (Softmax via CrossEntropyLoss)

Training: CrossEntropyLoss, Adam, Epochs=10, Batch=64
Evaluation: test accuracy, loss vs epochs, accuracy vs epochs,
            sample predictions vs actual
"""
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

BATCH_SIZE = 64
EPOCHS = 10
LR = 0.001

# ---------------- 1. Load MNIST (28x28 grayscale) ----------------
print("Loading MNIST via fetch_openml (mnist_784)...")
mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
X = mnist["data"].astype(np.float32) / 255.0  # scale 0-1
y = mnist["target"].astype(np.int64)
print(f"Full data: {X.shape}, labels: {y.shape}")

# Standard MNIST split: first 60000 train, last 10000 test
X_train, X_test = X[:60000].reshape(-1, 1, 28, 28), X[60000:].reshape(-1, 1, 28, 28)
y_train, y_test = y[:60000], y[60000:]
print(f"Train: {X_train.shape} Test: {X_test.shape}")

train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=1000, shuffle=False)

# ---------------- 2. Define CNN ----------------
class MNISTCNN(nn.Module):
    """CNN exactly as specified in Assignment 3."""
    def __init__(self):
        super().__init__()
        # Conv Layer 1: 32 filters, 3x3, ReLU (input 1x28x28 -> 32x26x26 with no padding)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.relu1 = nn.ReLU()
        # MaxPooling 2x2 : 32x26x26 -> 32x13x13
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Conv Layer 2: 64 filters, 3x3, ReLU : 32x13x13 -> 64x11x11
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.relu2 = nn.ReLU()
        # MaxPooling 2x2 : 64x11x11 -> 64x5x5
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # Flatten: 64*5*5 = 1600
        self.flatten = nn.Flatten()
        # Fully Connected: 128 neurons, ReLU
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.relu3 = nn.ReLU()
        # Output Layer: 10 classes (Softmax applied inside CrossEntropyLoss)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x

model = MNISTCNN().to(DEVICE)
print(model)

criterion = nn.CrossEntropyLoss()  # includes Softmax + NLL
optimizer = optim.Adam(model.parameters(), lr=LR)

# ---------------- 3. Training loop ----------------
train_losses, train_accs = [], []

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)
    print(f"Epoch [{epoch}/{EPOCHS}] Loss: {epoch_loss:.4f} Acc: {epoch_acc*100:.2f}%")

# ---------------- 4. Evaluation ----------------
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
test_acc = correct / total
print(f"Test Accuracy: {test_acc*100:.2f}% ({correct}/{total})")

# Save model
out_dir = os.path.dirname(os.path.abspath(__file__))
torch.save(model.state_dict(), os.path.join(out_dir, "mnist_cnn.pt"))

# ---------------- 5. Visualizations ----------------
plt.figure()
plt.plot(range(1, EPOCHS+1), train_losses, marker="o")
plt.title("Loss vs. Epochs")
plt.xlabel("Epoch")
plt.ylabel("CrossEntropy Loss")
plt.grid(True)
plt.savefig(os.path.join(out_dir, "loss_vs_epochs.png"), dpi=150, bbox_inches="tight")
plt.close()

plt.figure()
plt.plot(range(1, EPOCHS+1), [a*100 for a in train_accs], marker="o")
plt.title("Accuracy vs. Epochs")
plt.xlabel("Epoch")
plt.ylabel("Training Accuracy (%)")
plt.grid(True)
plt.savefig(os.path.join(out_dir, "accuracy_vs_epochs.png"), dpi=150, bbox_inches="tight")
plt.close()

# Sample predictions
model.eval()
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
axes = axes.ravel()
with torch.no_grad():
    sample_X = torch.from_numpy(X_test[:10]).to(DEVICE)
    preds = model(sample_X).argmax(dim=1).cpu().numpy()
for i in range(10):
    img = X_test[i].reshape(28, 28)
    axes[i].imshow(img, cmap="gray")
    axes[i].set_title(f"P:{preds[i]} A:{y_test[i]}", color=("green" if preds[i]==y_test[i] else "red"))
    axes[i].axis("off")
plt.suptitle("Test Images: Predicted vs Actual")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "sample_predictions.png"), dpi=150, bbox_inches="tight")
plt.close()

print("Saved: mnist_cnn.pt, loss_vs_epochs.png, accuracy_vs_epochs.png, sample_predictions.png")
print(f"HISTORY_LOSSES={train_losses}")
print(f"HISTORY_ACCS={train_accs}")
print(f"TEST_ACC={test_acc}")
