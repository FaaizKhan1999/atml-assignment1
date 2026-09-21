import torch
import numpy as np

def mixup_data(h, y, alpha=2.0):
    """
    Performs manifold mixup on a batch of hidden states.
    Ensures that paired examples come from different classes (y_i != y_j).
    """
    batch_size = h.size(0)
    device = h.device
    
    # Sample lambda from Beta(alpha, alpha)
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
        
    # Find a permutation where y_i != y_j to satisfy the mixup condition
    index = torch.randperm(batch_size, device=device)
    for _ in range(10):  # Retry loop to minimize label collisions
        collision_mask = y == y[index]
        if not collision_mask.any():
            break
        fix_idx = torch.randperm(batch_size, device=device)
        index[collision_mask] = index[fix_idx[collision_mask]]
        
    mixed_h = lam * h + (1 - lam) * h[index]
    return mixed_h, y, y[index], lam
