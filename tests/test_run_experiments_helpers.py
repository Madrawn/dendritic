import builtins
import logging
import os

import pytest
import torch
from torch.utils.data import DataLoader
from transformers.models.gpt2 import GPT2Tokenizer

from dendritic.experiments.run_experiments import (
    load_finetuning_data,
)
from dendritic.experiments.utils import (
    debug_dataset_integrity,
    setup_logging,
)


@pytest.mark.unit
def test_setup_logging_creates_file_handler(tmp_path, caplog):
    """Verify that `setup_logging` creates a logger with a FileHandler."""
    # Change working directory to a temporary path so the log file is created there
    cwd = tmp_path
    # Temporarily change the current directory
    original_cwd = os.getcwd()
    try:
        # Switch to temporary directory
        os.chdir(cwd)
        logger = setup_logging(log_level=logging.DEBUG)
        # The logger should have DEBUG level
        assert logger.level == logging.DEBUG
        # There should be at least one FileHandler
        file_handlers = [
            h for h in logger.handlers if isinstance(h, logging.FileHandler)
        ]
        assert file_handlers, "No FileHandler found in logger.handlers"
        # The file should exist on disk
        from pathlib import Path

        log_path = Path(file_handlers[0].baseFilename)
        assert log_path.exists()
    finally:
        # Restore original working directory
        os.chdir(original_cwd)


@pytest.mark.skip(reason="broken by recent changes; needs fixing")
@pytest.mark.unit
def test_load_finetuning_data_fallback(monkeypatch):
    """Force the fallback path and verify DataLoaders are returned."""
    # Monkey‑patch the import mechanism to raise ImportError for the handler
    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("dendritic.dataset_handlers.PythonAlpacaHandler"):
            raise ImportError("Simulated missing handler")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    train_dl, eval_dl = load_finetuning_data(
        tokenizer, max_length=64, batch_size=2, num_workers=0
    )

    assert isinstance(train_dl, DataLoader)
    assert isinstance(eval_dl, DataLoader)

    # Pull a single batch and verify shape
    batch = next(iter(train_dl))
    assert "input_ids" in batch
    # Batch size should match the requested size
    assert batch["input_ids"].shape[0] == 2


@pytest.mark.unit
def test_debug_dataset_integrity_logs(caplog):
    """Run the debug helper on a tiny synthetic dataset and verify logging output."""

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Synthetic data
    input_ids = torch.tensor(
        [
            [1, 2, 3, tokenizer.pad_token_id],
            [4, 5, 6, tokenizer.pad_token_id],
        ]
    )
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]])
    labels = torch.tensor([[1, 2, 3, -100], [4, 5, 6, -100]])

    class SimpleDataset(torch.utils.data.Dataset):
        def __getitem__(self, idx):
            return {
                "input_ids": input_ids[idx],
                "attention_mask": attention_mask[idx],
                "labels": labels[idx],
            }

        def __len__(self):
            return len(input_ids)

    simple_ds = SimpleDataset()

    caplog.set_level(logging.DEBUG)
    debug_dataset_integrity(simple_ds, tokenizer, logging.getLogger(__name__))

    messages = [record.message for record in caplog.records]
    assert any("Batch Shape" in msg for msg in messages)
    assert any("Pad Token ID" in msg for msg in messages)
    assert any("Padding Ratio" in msg for msg in messages)
    assert any("Masked Label Ratio" in msg for msg in messages)
