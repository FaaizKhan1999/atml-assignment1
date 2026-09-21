import torch
import torch.nn as nn
from torchvision.models import resnet18

class CIFAR_ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Initialize standard ResNet-18 without pretrained weights
        self.backbone = resnet18(weights=None)
        
        # Replace 7x7 stride-2 conv with 3x3 stride-1 conv for 32x32 images
        self.backbone.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        
        # Remove the initial max-pooling layer
        self.backbone.maxpool = nn.Identity()
        
        # Adjust the classifier head for the required number of classes
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        
    def forward(self, x, return_features=False):
        """Standard forward pass, supporting feature extraction for post-hoc scoring."""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.backbone.avgpool(x)
        f = torch.flatten(x, 1)
        logits = self.backbone.fc(f)
        
        if return_features:
            return logits, f
        return logits
        
    def forward_up_to_layer2(self, x):
        """Helper for PROSER: processes input up to layer2 for manifold mixup."""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        return x
        
    def forward_from_layer3(self, x):
        """Helper for PROSER: processes mixed representations from layer3 onward."""
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        f = torch.flatten(x, 1)
        logits = self.backbone.fc(f)
        return logits, f
