import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNet18Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = resnet18(weights=weights)
        
        self.features = nn.Sequential(*list(model.children())[:-1])
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        return x
