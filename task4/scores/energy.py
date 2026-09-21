import torch

def compute_energy(logits):
    """
    Energy Score.
    Returns unknownness score: -log(sum(exp(logits))). Larger value = more unknown.
    """
    # Use logsumexp for numerical stability
    energy = torch.logsumexp(logits, dim=1)
    return -energy
