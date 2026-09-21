import torch

def compute_msp(logits):
    """
    Maximum Softmax Probability (MSP).
    Returns unknownness score: 1 - max(softmax). Larger value = more unknown.
    """
    probs = torch.softmax(logits, dim=1)
    max_probs, _ = torch.max(probs, dim=1)
    return 1.0 - max_probs
