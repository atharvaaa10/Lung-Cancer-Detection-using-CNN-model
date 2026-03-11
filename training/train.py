import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import numpy as np
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix

from training.data_loader import LungDataset
from training.model import LungCancerCNN

# ── Paths & Hyperparameters ────────────────────────────────────────────────────
TRAIN_PATH       = 'dataset/Data/train'
VALID_PATH       = 'dataset/Data/valid'
BEST_MODEL_PATH  = 'backend/model/lung_model_6class_best.pth'
MODEL_SAVE_PATH  = 'backend/model/lung_model_6class.pth'
NUM_CLASSES      = 6
BATCH_SIZE       = 16
EPOCHS           = 30
LR               = 0.001
IMG_SIZE         = 256

CLASS_NAMES = [
    "Normal", "Benign", "Adenocarcinoma",
    "Large Cell Carcinoma", "Squamous Cell Carcinoma", "Malignant"
]

# ── Device ─────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on: {device}\n")

# ── Transforms (with augmentation for training) ────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.05, 0.05)),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# ── Datasets & Loaders ─────────────────────────────────────────────────────────
train_dataset = LungDataset(TRAIN_PATH, transform=train_transform)
val_dataset   = LungDataset(VALID_PATH, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ── Class Weights (handle imbalance) ──────────────────────────────────────────
label_counts = Counter([label for _, label in train_dataset])
total_samples = len(train_dataset)

class_weights = []
for cls in range(NUM_CLASSES):
    count = label_counts.get(cls, 1)
    weight = total_samples / (NUM_CLASSES * count)
    class_weights.append(weight)

class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
print("Class distribution in training set:")
for i, name in enumerate(CLASS_NAMES):
    print(f"  [{i}] {name:<28} → {label_counts.get(i, 0):>5} samples  (weight: {class_weights[i]:.4f})")
print()

# ── Model, Loss, Optimizer ────────────────────────────────────────────────────
model     = LungCancerCNN().to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=LR)

os.makedirs(os.path.dirname(BEST_MODEL_PATH), exist_ok=True)

# ── Training Loop ─────────────────────────────────────────────────────────────
best_val_acc   = 0.0
best_epoch     = 0
start_time     = time.time()

for epoch in range(1, EPOCHS + 1):
    # ── Train ──────────────────────────────────────────────────────────────────
    model.train()
    running_loss = 0.0

    loop = tqdm(train_loader, desc=f"Epoch {epoch:>2}/{EPOCHS}", unit="batch", leave=False)
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = running_loss / len(train_loader)

    # ── Validate ───────────────────────────────────────────────────────────────
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs        = model(images)
            predicted      = outputs.argmax(dim=1)
            correct       += (predicted == labels).sum().item()
            total         += labels.size(0)

    val_acc = correct / total

    # ── Print epoch summary ────────────────────────────────────────────────────
    marker = " ◄ best" if val_acc > best_val_acc else ""
    print(f"Epoch {epoch:>2}/{EPOCHS}  |  Train Loss: {avg_loss:.4f}  |  Val Accuracy: {val_acc:.4f}{marker}")

    # ── Save best model ────────────────────────────────────────────────────────
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch   = epoch
        torch.save(model.state_dict(), BEST_MODEL_PATH)

# ── Also save final model ──────────────────────────────────────────────────────
torch.save(model.state_dict(), MODEL_SAVE_PATH)

# ── Final Summary ─────────────────────────────────────────────────────────────
elapsed = time.time() - start_time
mins, secs = divmod(int(elapsed), 60)

print("\n" + "═" * 55)
print("  TRAINING COMPLETE")
print("═" * 55)
print(f"  Best Validation Accuracy : {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
print(f"  Best Epoch               : {best_epoch} / {EPOCHS}")
print(f"  Total Training Time      : {mins}m {secs}s")
print(f"  Best model saved to      : {BEST_MODEL_PATH}")
print(f"  Final model saved to     : {MODEL_SAVE_PATH}")
print("═" * 55)

# ── Post-Training Evaluation on Test Set ──────────────────────────────────────
print("\nRunning evaluation on test set using best model...\n")

TEST_PATH    = 'dataset/Data/test'
test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])
test_dataset = LungDataset(TEST_PATH, transform=test_transform)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# Load best saved model for evaluation
best_model = LungCancerCNN().to(device)
best_model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=device))
best_model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Evaluating", unit="batch"):
        images, labels = images.to(device), labels.to(device)
        outputs = best_model(images)
        preds   = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)

test_acc = (all_preds == all_labels).mean()

print("\n" + "═" * 60)
print("  TEST SET EVALUATION RESULTS")
print("═" * 60)
print(f"  Overall Test Accuracy : {test_acc:.4f} ({test_acc*100:.2f}%)")
print("═" * 60)

print("\n── Classification Report ──────────────────────────────────────")
print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=4))

cm = confusion_matrix(all_labels, all_preds)
short = ["Norm", "Benign", "Adeno", "LargeC", "Squam", "Malig"]
print("── Confusion Matrix ───────────────────────────────────────────")
print(f"{'':>26}" + "".join(f"{n:>8}" for n in short))
for i, row in enumerate(cm):
    print(f"  {CLASS_NAMES[i]:>24} |" + "".join(f"{v:>8}" for v in row))
print("─" * 60)
