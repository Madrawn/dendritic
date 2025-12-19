import random
import numpy as np
import torch
import logging
from pathlib import Path

from dendritic.experiments.run_experiments import (
    set_random_seed,
    debug_dataset_integrity,  # type: ignore
)

def test_set_random_seed_reproducibility():
    """Ensure that setting the same seed yields identical random numbers across libraries."""
    seed = 12345
    set_random_seed(seed)

    # Generate random numbers from each library
    py_rand = random.random()
    np_rand = np.random.rand()
    torch_rand = torch.rand(1).item()

    # Reset seed and generate again
    set_random_seed(seed)
    assert random.random() == py_rand
    assert np.random.rand() == np_rand
    assert torch.rand(1).item() == torch_rand

def test_debug_dataset_integrity_logs(caplog):
    """Run the debug helper on a tiny synthetic dataset and verify logging output."""
    from torch.utils.data import DataLoader, TensorDataset
    from transformers.models.gpt2 import GPT2Tokenizer

    # Create a minimal tokenizer (use pretrained to avoid heavy download in CI)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Synthetic dataset: two samples with input_ids and labels
    input_ids = torch.tensor([[1, 2, 3, tokenizer.pad_token_id], [4, 5, 6, tokenizer.pad_token_id]])
    attention_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0]])
    labels = torch.tensor([[1, 2, 3, -100], [4, 5, 6, -100]])

    dataset = TensorDataset(input_ids, attention_mask, labels)
    # Convert to dict format expected by debug_dataset_integrity
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

    # Capture debug logs
    caplog.set_level(logging.DEBUG)
    debug_dataset_integrity(simple_ds, tokenizer, logging.getLogger(__name__))

    # Verify that debug logs contain expected keys
    messages = [record.message for record in caplog.records]
    assert any("Batch Shape" in msg for msg in messages)
    assert any("Pad Token ID" in msg for msg in messages)
    assert any("Padding Ratio" in msg for msg in messages)
    assert any("Masked Label Ratio" in msg for msg in messages)