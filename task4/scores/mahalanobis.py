import torch

class MahalanobisScorer:
    def __init__(self):
        self.class_means = None
        self.inv_covariance = None

    def fit(self, train_features, train_targets, num_classes=10):
        """
        Estimates class means and shared diagonal covariance from training data.
        Adds 1e-6 to the diagonal entries as required.
        """
        feat_dim = train_features.shape[1]
        self.class_means = torch.zeros((num_classes, feat_dim), device=train_features.device)
        
        # Compute class means \mu_c
        for c in range(num_classes):
            class_mask = (train_targets == c)
            self.class_means[c] = train_features[class_mask].mean(dim=0)
            
        # Compute shared diagonal covariance \Sigma
        # Center the features using their respective class means
        centered_features = train_features.clone()
        for c in range(num_classes):
            class_mask = (train_targets == c)
            centered_features[class_mask] -= self.class_means[c]
            
        # Variance along each feature dimension across all centered samples
        variances = torch.var(centered_features, dim=0, unbiased=True)
        
        # Add 1e-6 to every diagonal entry to prevent division by zero
        variances = variances + 1e-6
        
        # Inverse of a diagonal covariance matrix is just 1 / variance
        self.inv_covariance = 1.0 / variances

    def score(self, features):
        """
        Computes unknownness score: min_c (f(x) - \mu_c)^T \Sigma^{-1} (f(x) - \mu_c)
        Larger value = further from all known class clusters = more unknown.
        """
        if self.class_means is None or self.inv_covariance is None:
            raise RuntimeError("MahalanobisScorer must be fitted with training data first.")
            
        num_classes = self.class_means.shape[0]
        num_samples = features.shape[0]
        distances = torch.zeros((num_samples, num_classes), device=features.device)
        
        for c in range(num_classes):
            diff = features - self.class_means[c]
            # (diff^T * \Sigma^-1 * diff) for a diagonal matrix simplifies to sum((diff^2) * inv_cov)
            dist_c = torch.sum((diff ** 2) * self.inv_covariance, dim=1)
            distances[:, c] = dist_c
            
        # The score is the minimum distance to any known class cluster
        min_distances, _ = torch.min(distances, dim=1)
        return min_distances
