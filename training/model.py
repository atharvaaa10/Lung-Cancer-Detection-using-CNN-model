import torch.nn as nn
import torchvision.models as models


class LungCancerCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Load pretrained ResNet18
        self.model = models.resnet18(weights=None)  # weights loaded from .pth at runtime

        # Replace first conv layer: RGB (3ch) → Grayscale (1ch)
        self.model.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # Replace final FC layer: 1000 classes → 6 classes
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, 6)

    def forward(self, x):
        return self.model(x)
