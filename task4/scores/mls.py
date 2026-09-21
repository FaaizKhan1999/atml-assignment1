import torch

def compute_mls(logits):
    """
    Maximum Logit Score (MLS).
    Returns unknownness score: -max(logits). Larger value = more unknown.
    """
    max_logits, _ = torch.max(logits, dim=1)
    return -max_logits
