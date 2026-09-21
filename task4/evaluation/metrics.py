import numpy as np
from sklearn.metrics import roc_auc_score

def get_threshold(val_scores, percentile=95):
    """Calculates the rejection threshold tau at the 95th percentile of validation scores."""
    return np.percentile(val_scores, percentile)

def compute_auroc(known_scores, unknown_scores):
    """Computes AUROC mapping knowns to 0 and unknowns to 1."""
    y_true = np.concatenate([np.zeros(len(known_scores)), np.ones(len(unknown_scores))])
    y_scores = np.concatenate([known_scores, unknown_scores])
    return roc_auc_score(y_true, y_scores)

def compute_rates(test_scores, unknown_scores, tau):
    """Returns TPR (knowns correctly accepted) and FPR (unknowns incorrectly accepted)."""
    tpr = np.mean(test_scores <= tau)
    fpr = np.mean(unknown_scores <= tau)
    return tpr, fpr
