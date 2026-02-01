"""
Data loading utilities for confidence-aware GPT experiments.

This module provides functions to prepare data for two-pass lookahead training,
where we need tokens at positions t, t+1, and t+2.
"""

from typing import Dict, Any, Tuple, Optional
import torch
from torch.utils.data import DataLoader, Dataset
from transformers.tokenization_utils import PreTrainedTokenizer

from dendritic.dataset_handlers.factory import get_handler
from dendritic.experiments.utils.PretrainingConfig import PretrainingConfig
from .config import ConfidenceExperimentConfig


class ConfidenceDataset(Dataset):
    """Dataset wrapper for confidence-aware training with lookahead.

    Takes a standard DataLoader and extracts sequences of length seq_len + 2,
    then provides triplets (tokens_t, tokens_t_plus_1, tokens_t_plus_2).
    """

    def __init__(self, dataloader: DataLoader, seq_len: int):
        self.dataloader = dataloader
        self.seq_len = seq_len
        # Pre-load all sequences (memory intensive but simple for now)
        # In production, this should be streaming
        self.sequences = []
        for batch in dataloader:
            input_ids = batch["input_ids"]
            # Ensure we have enough length
            if input_ids.shape[1] >= seq_len + 2:
                # Flatten batches into individual sequences
                for i in range(input_ids.shape[0]):
                    self.sequences.append(input_ids[i])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        input_ids = self.sequences[idx]  # Shape: (seq_len + 2,)
        # Extract the three sequences
        tokens_t = input_ids[: self.seq_len]
        # For lookahead training, we need single tokens
        # at positions seq_len and seq_len+1
        tokens_t_plus_1 = input_ids[self.seq_len]  # Single token
        tokens_t_plus_2 = input_ids[self.seq_len + 1]  # Single token

        return tokens_t, tokens_t_plus_1, tokens_t_plus_2


def prepare_confidence_data(
    config: ConfidenceExperimentConfig,
    tokenizer: PreTrainedTokenizer,
    dataset_kwargs: Optional[Dict[str, Any]] = None,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """
    Prepare data for confidence-aware experiments with two-pass lookahead training.

    This function loads standard dataset using existing dataset handlers,
    creates sequences of length `seq_len + 2` (where seq_len is the context window),
    and returns batches with structure: `(tokens_t, tokens_t_plus_1, tokens_t_plus_2)`.

    Parameters
    ----------
    config : ConfidenceExperimentConfig
        Configuration object containing dataset, max_seq_len, batch_size, etc.
    tokenizer : PreTrainedTokenizer
        Tokenizer for the dataset.
    dataset_kwargs : dict, optional
        Additional keyword arguments passed to the dataset handler.
    num_workers : int, optional
        Number of worker processes for data loading.

    Returns
    -------
    dict[str, DataLoader]
        Dictionary with 'train' and 'eval' DataLoaders. Each batch from the DataLoader
        is a tuple of three tensors: (tokens_t, tokens_t_plus_1, tokens_t_plus_2).

    Notes
    -----
    - The standard dataset handler returns sequences of length `max_seq_len` with
      labels shifted by 1 position for next-token prediction.
    - For confidence training, we need sequences of length `max_seq_len + 2` to
      have tokens at t, t+1, and t+2 positions.
    - This function adapts the existing data loading to produce the required triplets.
    """
    if dataset_kwargs is None:
        dataset_kwargs = {}

    # Get the appropriate dataset handler
    handler = get_handler(
        config.dataset,
        tokenizer,
        max_length=config.max_seq_len + 2,  # Need extra tokens for lookahead
    )

    # Prepare standard dataloaders using the handler
    # We'll use a modified config with increased sequence length
    modified_config = _create_modified_config(config)

    # Get standard dataloaders, passing dataset_kwargs
    # to prepare_pretraining_dataloaders
    standard_loaders = handler.prepare_pretraining_dataloaders(
        config=modified_config, num_workers=num_workers, kwargs=dataset_kwargs
    )

    # Convert standard batches to confidence format
    train_loader = _convert_to_confidence_loader(
        standard_loaders["train"], config.max_seq_len, config.batch_size, shuffle=True
    )

    eval_loader = _convert_to_confidence_loader(
        standard_loaders["eval"], config.max_seq_len, config.batch_size, shuffle=False
    )

    return {"train": train_loader, "eval": eval_loader}


def _create_modified_config(config: ConfidenceExperimentConfig) -> PretrainingConfig:
    """
    Create a modified config with increased sequence length for lookahead.

    The confidence experiment needs sequences of length max_seq_len + 2
    to have tokens at t, t+1, and t+2 positions.
    """
    # Create a copy of the config with modified max_seq_len
    # We need to be careful not to modify the original config
    modified = PretrainingConfig(
        # Copy all relevant fields from config
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        max_seq_len=config.max_seq_len + 2,  # Add 2 for lookahead
        dropout=config.dropout,
        layer_type=config.layer_type,
        poly_rank=config.poly_rank,
        poly_degree=config.poly_degree,
        dendritic_dropout=config.dendritic_dropout,
        training_steps=config.training_steps,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        scheduler_type=config.scheduler_type,
        eval_split_ratio=config.eval_split_ratio,
        plateau_patience=config.plateau_patience,
        plateau_factor=config.plateau_factor,
        plateau_threshold=config.plateau_threshold,
        plateau_cooldown=config.plateau_cooldown,
        plateau_min_lr=config.plateau_min_lr,
        early_stop_multiplier=config.early_stop_multiplier,
        eval_batches=config.eval_batches,
        seeds=config.seeds,
        cohort_scheduler=config.cohort_scheduler,
        output_dir=config.output_dir,
        dataset=config.dataset,
        grouped=config.grouped,
        group_separator=config.group_separator,
        dataset_kwargs=config.dataset_kwargs,
        baseline_hidden_dim=config.baseline_hidden_dim,
        dendritic_hidden_dim=config.dendritic_hidden_dim,
        dendritic_stack_hidden_dim=config.dendritic_stack_hidden_dim,
        param_grid=config.param_grid,
    )
    return modified


def confidence_collate_fn(batch):
    """Collate function that returns a tuple of three tensors."""
    # batch is a list of tuples: [(tokens_t1, tokens_t_plus_11, tokens_t_plus_21), ...]
    # We need to stack each component separately
    tokens_t_batch = torch.stack([item[0] for item in batch])
    tokens_t_plus_1_batch = torch.stack([item[1] for item in batch])
    tokens_t_plus_2_batch = torch.stack([item[2] for item in batch])
    return (tokens_t_batch, tokens_t_plus_1_batch, tokens_t_plus_2_batch)


def _convert_to_confidence_loader(
    dataloader: DataLoader,
    seq_len: int,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    """
    Convert a standard dataloader to produce confidence training triplets.

    Standard batches have shape (batch_size, seq_len + 2) for input_ids.
    We need to extract:
    - tokens_t: input_ids[:, :seq_len] (first seq_len tokens)
    - tokens_t_plus_1: input_ids[:, 1:seq_len+1] (shift by 1)
    - tokens_t_plus_2: input_ids[:, 2:seq_len+2] (shift by 2)

    Returns a new DataLoader that yields tuples of these three tensors.
    """

    # Create the dataset using the module-level class
    dataset = ConfidenceDataset(dataloader, seq_len)

    # Create a new DataLoader with custom collate function
    # Use num_workers=0 for Windows compatibility (multiprocessing pickling issues)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,  # Shuffle for training, but configurable
        num_workers=0,  # Force 0 for compatibility; can be overridden if needed
        pin_memory=dataloader.pin_memory,
        drop_last=dataloader.drop_last,
        collate_fn=confidence_collate_fn,
    )


def create_confidence_batch_from_sequences(
    input_ids: torch.Tensor,
    seq_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create confidence training triplets from a batch of sequences.

    Parameters
    ----------
    input_ids : torch.Tensor
        Tensor of shape (batch_size, seq_len + 2) containing token IDs.
    seq_len : int
        The context window size (original sequence length).

    Returns
    -------
    tuple of three torch.Tensor
        tokens_t: shape (batch_size, seq_len)
        tokens_t_plus_1: shape (batch_size,) - single token at position seq_len
        tokens_t_plus_2: shape (batch_size,) - single token at position seq_len + 1
    """
    if input_ids.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got shape {input_ids.shape}")

    if input_ids.shape[1] < seq_len + 2:
        raise ValueError(
            f"Sequence length {input_ids.shape[1]} is less than required {seq_len + 2}"
        )

    tokens_t = input_ids[:, :seq_len]
    tokens_t_plus_1 = input_ids[:, seq_len]  # Single token at position seq_len
    tokens_t_plus_2 = input_ids[:, seq_len + 1]  # Single token at position seq_len + 1

    return tokens_t, tokens_t_plus_1, tokens_t_plus_2
