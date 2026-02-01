"""
Loss computation utilities for language modeling and confidence prediction.

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


def compute_confidence_loss(
    confidence_pred: torch.Tensor,
    future_losses: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Compute confidence prediction loss using MSE.

    Args:
        confidence_pred: Predicted confidence scores of shape [batch_size, seq_len]
        future_losses: Actual future losses of shape [batch_size, seq_len]
        reduction: Loss reduction method ('mean', 'sum', or 'none')

    Returns:
        Confidence loss tensor
    """
    return F.mse_loss(
        confidence_pred,
        future_losses.detach(),
        reduction=reduction,
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


def compute_total_confidence_aware_loss(
    logits: torch.Tensor,
    confidence_pred: torch.Tensor,
    labels: torch.Tensor,
    future_losses: torch.Tensor,
    alpha: float = 1.0,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute total loss for confidence-aware models.

    Args:
        logits: Token prediction logits of shape [batch_size, seq_len, vocab_size]
        confidence_pred: Confidence predictions of shape [batch_size, seq_len]
        labels: Target labels of shape [batch_size, seq_len]
        future_losses: Actual future losses of shape [batch_size, seq_len]
        alpha: Weight for confidence loss
        ignore_index: Token index to ignore in loss computation

    Returns:
        Tuple of (total_loss, lm_loss, confidence_loss)
    """
    # Language modeling loss (only up to seq_len-1 for valid next tokens)
    lm_loss = compute_language_modeling_loss(logits, labels, ignore_index)

    # Confidence loss (only up to seq_len-1 where we have future losses)
    # Note: confidence_pred and future_losses should already be aligned
    confidence_loss = compute_confidence_loss(
        confidence_pred[:, :-1],  # Exclude last position
        future_losses[:, :-1],  # Exclude last position
    )

    total_loss = lm_loss + alpha * confidence_loss

    return total_loss, lm_loss, confidence_loss
