import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class DomainAdaptationResNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pre-trained ResNet18
        self.backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # 512-dimensional feature right before the classifier head
        self.out_features = self.backbone.fc.in_features
        # Remove original ImageNet classifier
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        return self.backbone(x)

    def train(self, mode=True):
        """
        Override train method to freeze BatchNorm running statistics.
        Scale and bias parameters (gamma and beta) remain trainable.
        """
        super().train(mode)
        if mode:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                    # Setting eval() stops tracking running mean/variance
                    m.eval()
                    # Ensure weight/bias remain trainable
                    if m.weight is not None:
                        m.weight.requires_grad_(True)
                    if m.bias is not None:
                        m.bias.requires_grad_(True)
