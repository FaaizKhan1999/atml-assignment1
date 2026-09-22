import torch

def pairwise_distance(x, y):
    """Computes the squared Euclidean distance between all pairs (x_i, y_j)."""
    x_norm = (x ** 2).sum(1).view(-1, 1)
    y_norm = (y ** 2).sum(1).view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y.t())
    return torch.clamp(dist, 0.0, float('inf'))

def mmd_rbf_loss(source_features, target_features):
    """Calculates MMD using 3 RBF kernels (0.5x, 1x, 2x median distance)."""
    batch_size = source_features.size(0)
    
    # Combine features to find the median distance across the whole batch
    combined = torch.cat([source_features, target_features], dim=0)
    dist = pairwise_distance(combined, combined)
    
    # Extract the upper triangular part to find the median
    upper_tri_indices = torch.triu_indices(row=dist.size(0), col=dist.size(1), offset=1)
    pairwise_dists = dist[upper_tri_indices[0], upper_tri_indices[1]]
    median_dist = torch.median(pairwise_dists).item()
    
    if median_dist == 0:
        median_dist = 1.0 # Prevent division by zero
        
    bandwidths = [0.5 * median_dist, 1.0 * median_dist, 2.0 * median_dist]
    
    # Compute kernel matrices
    xx = pairwise_distance(source_features, source_features)
    yy = pairwise_distance(target_features, target_features)
    xy = pairwise_distance(source_features, target_features)
    
    mmd_loss = 0.0
    for bw in bandwidths:
        gamma = 1.0 / (2.0 * bw)
        kernel_xx = torch.exp(-gamma * xx).mean()
        kernel_yy = torch.exp(-gamma * yy).mean()
        kernel_xy = torch.exp(-gamma * xy).mean()
        mmd_loss += kernel_xx + kernel_yy - 2.0 * kernel_xy
        
    return mmd_loss
