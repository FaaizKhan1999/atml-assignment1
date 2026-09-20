import random
import numpy as np
import torch
import torch.nn as nn

def set_seed(seed: int = 6304):
    """Locks all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_batchnorm_eval(module: nn.Module):
    """
    Recursively sets all BatchNorm modules to eval mode.
    Must be called AFTER model.train() in the training loop.
    """
    for m in module.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.eval()

class EarlyStopping:
    """Tracks validation macro-F1 to stop training if it doesn't improve."""
    def __init__(self, patience: int = 5, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.best_score = -float('inf') if mode == "max" else float('inf')
        self.epochs_without_improvement = 0
        self.best_weights = None

    def step(self, score: float, model: nn.Module) -> bool:
        """Returns True if training should stop."""
        improved = score > self.best_score if self.mode == "max" else score < self.best_score
        
        if improved:
            self.best_score = score
            self.epochs_without_improvement = 0
            self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.epochs_without_improvement += 1

        return self.epochs_without_improvement >= self.patience
