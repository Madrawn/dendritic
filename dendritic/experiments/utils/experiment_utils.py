# dendritic/experiments/utils/experiment_utils.py
"""
Utility functions for experiment orchestration.
"""

import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers.models.gpt2 import GPT2Tokenizer


def set_random_seed(seed: int) -> None:
    """Set seeds for reproducibility across random, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(log_level: int | None = None) -> logging.Logger:
    """Configure logging."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = logging.getLogger(__name__)
    # Remove any existing handlers to avoid duplicate logs
    logger.handlers.clear()
    logger.propagate = False
    level = log_level if log_level is not None else logging.INFO
    logger.setLevel(level)

    # Stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)

    # File handler
    file_handler = logging.FileHandler(Path(f"experiment_{timestamp}.log"))
    file_handler.setLevel(level)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    # Ensure root logger also has a stream handler for modules that log via logging.info
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    # Remove existing stream handlers to avoid duplicates
    root_stream_handlers = [
        h for h in root_logger.handlers if isinstance(h, logging.StreamHandler)
    ]
    for h in root_stream_handlers:
        root_logger.removeHandler(h)
    # Add a single stream handler if none exist
    if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
        root_stream_handler = logging.StreamHandler()
        root_stream_handler.setLevel(level)
        root_stream_handler.setFormatter(formatter)
        root_logger.addHandler(root_stream_handler)

    return logger


def debug_dataset_integrity(
    dataset, tokenizer: GPT2Tokenizer, logger: logging.Logger
) -> None:
    """
    Inspect a single batch from the dataset and log debugging information.
    This replicates the previous print‑based debugging block but uses the
    provided logger at DEBUG level.
    """
    # Create a temporary loader to grab one batch
    debug_loader = DataLoader(dataset, batch_size=4)  # type: ignore
    batch = next(iter(debug_loader))

    inp = batch["input_ids"]
    lbl = batch["labels"]
    pad_id = tokenizer.pad_token_id

    # 1. Check Ratio of Padding
    total_tokens = inp.numel()
    pad_tokens = (inp == pad_id).sum().item()
    logger.debug(f"Batch Shape: {inp.shape}")
    logger.debug(f"Pad Token ID: {pad_id}")
    logger.debug(
        f"Padding Ratio: {pad_tokens / total_tokens:.2%} ({pad_tokens}/{total_tokens} tokens)"
    )

    # 2. Check Masking (Crucial for correct PPL)
    masked_tokens = (lbl == -100).sum().item()
    logger.debug(f"Masked Label Ratio (-100): {masked_tokens / total_tokens:.2%}")

    # 3. Visual Inspection
    logger.debug("-" * 20 + " Sample 0 (Decoded) " + "-" * 20)
    logger.debug("".join(tokenizer.decode(inp[0])))
    logger.debug("-" * 20 + " Sample 0 (Raw IDs) " + "-" * 20)
    logger.debug(inp[0].tolist())
    logger.debug("-" * 20 + " Sample 0 (Labels) " + "-" * 20)
    logger.debug(lbl[0].tolist())
    logger.debug("!" * 40 + "\n")
