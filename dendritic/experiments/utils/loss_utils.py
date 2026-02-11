"""
Loss computation utilities for language modeling and doubt prediction.

This module provides standardized loss functions to separate loss computation
from model forward passes, following PyTorch conventions.
"""

import torch
import torch.nn.functional as F


def compute_language_modeling_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """
    Compute standard language modeling loss for next-token prediction.

    Args:
        logits: Tensor of shape [batch_size, seq_len, vocab_size]
        labels: Tensor of shape [batch_size, seq_len] with token indices
        ignore_index: Token index to ignore in loss computation

    Returns:
        Scalar loss tensor
    """
    # Shift logits and labels for next-token prediction
    # logits[..., :-1, :] predicts labels[..., 1:]
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )


def compute_sequence_language_modeling_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Compute language modeling loss for each position in the sequence.

    Args:
        logits: Tensor of shape [batch_size, seq_len, vocab_size]
        labels: Tensor of shape [batch_size, seq_len] with token indices
        ignore_index: Token index to ignore in loss computation
        reduction: Loss reduction method ('none', 'mean', or 'sum')

    Returns:
        Loss tensor of shape [batch_size, seq_len] if reduction='none',
        otherwise scalar tensor
    """
    # Make tensors contiguous before reshaping to avoid view errors on sliced inputs
    logits = logits.contiguous()
    labels = labels.contiguous()

    # Reshape for per-position loss computation
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.view(-1, vocab_size)
    labels_flat = labels.view(-1)

    # Compute loss for each position
    loss_flat = F.cross_entropy(
        logits_flat,
        labels_flat,
        ignore_index=ignore_index,
        reduction="none",
    )

    # Reshape back to [batch_size, seq_len]
    loss_per_position = loss_flat.view(batch_size, seq_len)

    if reduction == "none":
        return loss_per_position
    elif reduction == "mean":
        # Average over non-ignored positions
        mask = labels != ignore_index
        if mask.any():
            return loss_per_position[mask].mean()
        else:
            return torch.tensor(0.0, device=logits.device)
    elif reduction == "sum":
        # Sum over non-ignored positions
        mask = labels != ignore_index
        if mask.any():
            return loss_per_position[mask].sum()
        else:
            return torch.tensor(0.0, device=logits.device)
    else:
        raise ValueError(f"Invalid reduction: {reduction}")
