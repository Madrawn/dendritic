"""
Data loading utilities for self-conditioned GPT experiments.

This module provides functions to prepare data for self-conditioned training,
which uses standard next-token prediction with sequences of length seq_len + 1,
producing pairs (tokens_t, tokens_t_plus_1).
"""

import logging
from typing import Dict, Any, Tuple, Optional
import torch
from torch.utils.data import DataLoader, Dataset
from transformers.tokenization_utils import PreTrainedTokenizer

from dendritic.dataset_handlers.factory import get_handler
from dendritic.experiments.utils.PretrainingConfig import PretrainingConfig
from .config import SelfConditionedExperimentConfig


class SelfConditionedDataset(Dataset):
    """Dataset wrapper for self-conditioned training.

    Takes a standard DataLoader and extracts sequences of length seq_len + 1,
    then provides pairs (tokens_t, tokens_t_plus_1).
    """

    def __init__(self, dataloader: DataLoader, seq_len: int):
        self.dataloader = dataloader
        self.seq_len = seq_len
        # Pre-load all sequences (memory intensive but simple for now)
        self.sequences = []
        for batch in dataloader:
            input_ids = batch["input_ids"]
            if input_ids.shape[1] >= seq_len + 1:
                for i in range(input_ids.shape[0]):
                    self.sequences.append(input_ids[i])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        input_ids = self.sequences[idx]  # Shape: (seq_len + 1,)
        tokens_t = input_ids[: self.seq_len]
        tokens_t_plus_1 = input_ids[self.seq_len]  # Single token
        return tokens_t, tokens_t_plus_1


def prepare_self_conditioned_data(
    config: SelfConditionedExperimentConfig,
    tokenizer: PreTrainedTokenizer,
    dataset_kwargs: Optional[Dict[str, Any]] = None,
    num_workers: int = 0,
) -> Dict[str, DataLoader]:
    """
    Prepare data for self-conditioned experiments.

    This function loads standard dataset using existing dataset handlers,
    creates sequences of length `seq_len + 1`, and returns batches with
    structure: (tokens_t, tokens_t_plus_1).
    """
    if dataset_kwargs is None:
        dataset_kwargs = {}

    handler = get_handler(
        config.dataset,
        tokenizer,
        max_length=config.max_seq_len + 1,
    )
    logging.info(f"Preparing data using handler: {handler.__class__.__name__}")
    modified_config = _create_modified_config(config)

    standard_loaders = handler.prepare_pretraining_dataloaders(
        config=modified_config, num_workers=num_workers, kwargs=dataset_kwargs
    )

    train_loader = _convert_to_self_conditioned_loader(
        standard_loaders["train"], config.max_seq_len, config.batch_size, shuffle=True
    )
    eval_loader = _convert_to_self_conditioned_loader(
        standard_loaders["eval"], config.max_seq_len, config.batch_size, shuffle=False
    )

    return {"train": train_loader, "eval": eval_loader}


def _create_modified_config(config: SelfConditionedExperimentConfig) -> PretrainingConfig:
    """Create a modified config with increased sequence length."""
    modified = PretrainingConfig(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        max_seq_len=config.max_seq_len + 1,
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


def self_conditioned_collate_fn(batch):
    """Collate function that returns a tuple of two tensors."""
    tokens_t_batch = torch.stack([item[0] for item in batch])
    tokens_t_plus_1_batch = torch.stack([item[1] for item in batch])
    return (tokens_t_batch, tokens_t_plus_1_batch)


def _convert_to_self_conditioned_loader(
    dataloader: DataLoader,
    seq_len: int,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    """Convert a standard dataloader to produce self-conditioned training pairs."""
    dataset = SelfConditionedDataset(dataloader, seq_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=dataloader.pin_memory,
        drop_last=dataloader.drop_last,
        collate_fn=self_conditioned_collate_fn,
    )


def create_self_conditioned_batch_from_sequences(
    input_ids: torch.Tensor,
    seq_len: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create self-conditioned training pairs from a batch of sequences."""
    if input_ids.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got shape {input_ids.shape}")
    if input_ids.shape[1] < seq_len + 1:
        raise ValueError(f"Sequence length {input_ids.shape[1]} is less than required {seq_len + 1}")
    tokens_t = input_ids[:, :seq_len]
    tokens_t_plus_1 = input_ids[:, seq_len]
    return tokens_t, tokens_t_plus_1
