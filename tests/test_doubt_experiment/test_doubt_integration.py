# ruff: noqa: PLR6301, PLR2004
"""
Integration tests for doubt-aware GPT experiment.

These tests verify that the doubt experiment runs end-to-end with dummy data
and produces valid results.
"""

import pytest
import tempfile
import shutil
from transformers.models.gpt2 import GPT2Tokenizer

from dendritic.experiments.doubt.config import DoubtExperimentConfig
from dendritic.experiments.doubt.data_loader import prepare_doubt_data


class TestDoubtExperimentIntegration:
    """Integration tests for doubt experiment."""

    @pytest.fixture
    def temp_results_dir(self):
        """Create a temporary directory for test results."""
        temp_dir = tempfile.mkdtemp(prefix="doubt_test_")
        yield temp_dir
        # Cleanup after test
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def small_config(self, temp_results_dir):
        """Create a minimal configuration for fast testing."""
        # GPT2 tokenizer has vocab_size=50257, so we need to match that
        return DoubtExperimentConfig(
            vocab_size=50257,  # Match GPT2 tokenizer vocab size
            embed_dim=64,
            num_heads=4,
            num_layers=2,
            max_seq_len=32,
            batch_size=2,
            training_steps=10,  # Very small for testing
            seeds=[42],  # Single seed for speed
            results_dir=temp_results_dir,
            dataset="openwebmath",  # Use openwebmath handler for longer sequences
            dataset_kwargs={"max_samples": 16},  # Very small dataset
        )

    @pytest.fixture
    def tokenizer(self):
        """Create a tokenizer for testing."""
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    @pytest.mark.integration
    def test_data_loading(self, small_config, tokenizer):
        """Test that data can be loaded with the doubt data loader."""
        dataloaders = prepare_doubt_data(
            config=small_config,
            tokenizer=tokenizer,
            dataset_kwargs=small_config.dataset_kwargs,
        )

        # Verify dataloaders are created
        assert "train" in dataloaders
        assert "eval" in dataloaders

        train_loader = dataloaders["train"]
        eval_loader = dataloaders["eval"]

        # Verify they are DataLoader instances
        from torch.utils.data import DataLoader

        assert isinstance(train_loader, DataLoader)
        assert isinstance(eval_loader, DataLoader)

        # Try to get a batch
        batch = next(iter(train_loader))

        # Doubt data loader returns tuple of two tensors
        # (tokens_t, tokens_t_plus_1)
        assert isinstance(batch, tuple)
        assert len(batch) == 2

        tokens_t, tokens_t_plus_1 = batch

        # Verify shapes
        assert tokens_t.shape == (small_config.batch_size, small_config.max_seq_len)
        # tokens_t_plus_1 is a single token (not a sequence)
        assert tokens_t_plus_1.shape == (small_config.batch_size,)

        # Verify they are tensors
        import torch

        assert isinstance(tokens_t, torch.Tensor)
        assert isinstance(tokens_t_plus_1, torch.Tensor)
