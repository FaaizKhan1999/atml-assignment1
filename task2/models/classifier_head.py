import torch.nn as nn

class LinearClassifier(nn.Module):
    def __init__(self, in_features: int = 512, num_classes: int = 7):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.fc(x)
