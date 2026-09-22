import torch.nn as nn

class DomainDiscriminator(nn.Module):
    def __init__(self, in_features=512):
        super().__init__()
        # Outputs 1 logit for BCEWithLogitsLoss (Source=1, Target=0)
        self.net = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )
        
    def forward(self, x):
        return self.net(x)
