import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

def get_6class_label(folder_name):
    """Map folder name to class index for 6-class classification"""
    label_map = {
        'normal': 0,
        'benign': 1,
        'adenocarcinoma': 2,
        'large.cell.carcinoma': 3,
        'squamous.cell.carcinoma': 4,
        'malignant': 5
    }
    return label_map.get(folder_name.lower(), -1)

class LungDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.data = []
        self.transform = transform
        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            if os.path.isdir(folder_path):
                label = get_6class_label(folder)
                if label == -1:
                    continue  # Skip unknown folders
                for file in os.listdir(folder_path):
                    if file.endswith(('.jpg', '.png')):
                        img_path = os.path.join(folder_path, file)
                        self.data.append((img_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert('L')  # grayscale
        if self.transform:
            image = self.transform(image)
        return image, label
