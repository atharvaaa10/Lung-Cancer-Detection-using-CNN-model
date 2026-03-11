import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

from training.data_loader import LungDataset
from training.model import LungCancerCNN

# Paths
TEST_PATH = "dataset/Data/test"
MODEL_PATH = "backend/model/lung_model_6class_best.pth"

IMG_SIZE = 256
BATCH_SIZE = 16

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transforms
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# Dataset
test_dataset = LungDataset(TEST_PATH, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load model
model = LungCancerCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Convert to numpy
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Class names
class_names = [
    "Normal",
    "Benign",
    "Adenocarcinoma",
    "Large Cell Carcinoma",
    "Squamous Cell Carcinoma",
    "Malignant"
]

# Classification report
print("\nClassification Report\n")
print(classification_report(all_labels, all_preds, target_names=class_names))

# Confusion matrix
print("\nConfusion Matrix\n")
print(confusion_matrix(all_labels, all_preds))