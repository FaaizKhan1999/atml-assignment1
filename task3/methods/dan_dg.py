import torch
import torch.nn as nn
import torch.nn.functional as F

def rbf_kernel_median_heuristic(x, y, multipliers):
    """
    Computes the multikernel RBF using the median pairwise squared distance
    of the combined (x, y) set.
    """
    # Combine representations to find the median squared distance for the pair
    combined = torch.cat([x, y], dim=0)
    
    # Compute pairwise squared euclidean distances
    dist_sq = torch.cdist(combined, combined, p=2.0) ** 2
    
    # Extract the median of the non-zero distances to avoid collapsing bandwidth
    dist_sq_flat = dist_sq.view(-1)
    median_sq = torch.median(dist_sq_flat[dist_sq_flat > 0])
    if median_sq.item() == 0:
        median_sq = torch.tensor(1.0, device=combined.device)
        
    # Calculate distances for the individual MMD components
    xx_dist = torch.cdist(x, x, p=2.0) ** 2
    yy_dist = torch.cdist(y, y, p=2.0) ** 2
    xy_dist = torch.cdist(x, y, p=2.0) ** 2
    
    k_xx, k_yy, k_xy = 0.0, 0.0, 0.0
    
    for m in multipliers:
        # gamma = 1 / (2 * sigma^2), where sigma^2 is m * median_sq
        gamma = 1.0 / (2.0 * m * median_sq)
        k_xx += torch.exp(-gamma * xx_dist).mean()
        k_yy += torch.exp(-gamma * yy_dist).mean()
        k_xy += torch.exp(-gamma * xy_dist).mean()
        
    return k_xx, k_yy, k_xy

def compute_pairwise_mmd(x, y, multipliers):
    """
    Calculates MMD^2(x, y) = E[k(x,x)] + E[k(y,y)] - 2E[k(x,y)]
    """
    k_xx, k_yy, k_xy = rbf_kernel_median_heuristic(x, y, multipliers)
    return k_xx + k_yy - 2 * k_xy

def dan_dg_loss(features, logits, labels, config):
    """
    Computes the ERM Cross Entropy + Lambda-scaled Pairwise Source MMD.
    Expects features in a strict domain-balanced order: [8 Photo, 8 Art, 8 Cartoon].
    """
    ce_loss = F.cross_entropy(logits, labels)
    
    # Extract configurations
    lambda_dg = config['dan_dg'].get('lambda_dg', 1.0)
    multipliers = config['dan_dg'].get('kernel_multipliers', [0.5, 1.0, 2.0])
    
    # Split batch into the three domains
    batch_size = features.size(0)
    chunk = batch_size // 3
    
    f_p = features[0:chunk]            # Photo
    f_a = features[chunk:2*chunk]      # Art Painting
    f_c = features[2*chunk:batch_size] # Cartoon
    
    # Compute pairwise discrepancies
    mmd_pa = compute_pairwise_mmd(f_p, f_a, multipliers)
    mmd_pc = compute_pairwise_mmd(f_p, f_c, multipliers)
    mmd_ac = compute_pairwise_mmd(f_a, f_c, multipliers)
    
    # Average MMD across the three pairs
    mmd_loss = (mmd_pa + mmd_pc + mmd_ac) / 3.0
    
    total_loss = ce_loss + (lambda_dg * mmd_loss)
    
    return total_loss, ce_loss, mmd_loss
