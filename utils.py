"""Utility helpers for FedMTFI."""

import torch


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute top-1 accuracy from raw logits and integer labels.

    Parameters
    ----------
    logits : torch.Tensor
        Model output logits of shape ``(B, C)``.
    labels : torch.Tensor
        Ground-truth class indices of shape ``(B,)``.

    Returns
    -------
    float
        Fraction of correct predictions in ``[0, 1]``.
    """
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()
