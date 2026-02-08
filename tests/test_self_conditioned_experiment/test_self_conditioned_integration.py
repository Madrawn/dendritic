# ruff: noqa: PLR6301, PLR2004
"""
Integration tests for self-conditioned GPT experiment.

These tests verify that the self-conditioned experiment runs end-to-end with dummy data
and produces valid results.
"""

import pytest
import torch
import tempfile
import shutil
from pathlib import Path
from transformers.models.gpt2 import GPT2Tokenizer

from dendritic.experiments.self_conditioned.config import SelfConditionedExperimentConfig
from dendritic.experiments.self_conditioned.experiment import SelfConditionedExperiment
from dendritic.experiments.self_conditioned.data_loader import prepare_self_conditioned_data


@pytest.mark.integration
class TestSelfConditionedExperimentIntegration:
    """Integration tests for self-conditioned experiment."""

    @pytest.fixture
    def temp_results_dir(self):
        """Create a temporary directory for test results."""
        temp_dir = tempfile.mkdtemp(prefix="self_cond_test_")
        yield temp_dir
        # Cleanup after test
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def small_config(self, temp_results_dir):
        """Create a minimal configuration for fast testing."""
        # GPT2 tokenizer has vocab_size=50257, so we need to match that
        return SelfConditionedExperimentConfig(
            vocab_size=50257,  # Match GPT2 tokenizer vocab size
            embed_dim=64,
            num_heads=4,
            num_layers=2,
            max_seq_len=32,
            batch_size=2,
            training_steps=10,  # Very small for testing
            seeds=[42],  # Single seed for speed
            results_dir=temp_results_dir,
            do_compile=False,  # Disable compilation for testing
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
    def test_experiment_initialization(self, small_config, tokenizer):
        """Test that the experiment can be initialized and models created."""
        experiment = SelfConditionedExperiment(small_config)

        # Verify experiment attributes
        assert experiment.config == small_config
        assert experiment.device in {"cuda", "cpu"}

        # Create models
        standard_model, self_conditioned_model = experiment.create_models()

        # Verify models are created
        assert standard_model is not None
        assert self_conditioned_model is not None

        # Verify they're on a valid device (could be CPU or CUDA)
        # Models might not be moved to device yet in create_models()
        std_device = next(standard_model.parameters()).device.type
        sc_device = next(self_conditioned_model.parameters()).device.type
        assert std_device in {"cpu", "cuda"}
        assert sc_device in {"cpu", "cuda"}
        assert std_device == sc_device  # Both should be on same device

        # Verify parameter counts are reasonable
        std_params = sum(p.numel() for p in standard_model.parameters())
        sc_params = sum(p.numel() for p in self_conditioned_model.parameters())

        # Self-conditioned model should have more parameters due to loss predictor and adaptive layers
        assert sc_params > std_params

        # But not orders of magnitude more (should be similar scale)
        assert sc_params < std_params * 1.5  # Within 50% more parameters

    @pytest.mark.integration
    def test_data_loading(self, small_config, tokenizer):
        """Test that data can be loaded with the self-conditioned data loader."""
        dataloaders = prepare_self_conditioned_data(
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

        # Self-conditioned data loader returns tuple of two tensors
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

    @pytest.mark.integration
    def test_minimal_experiment_run(self, small_config, tokenizer):
        """Test that the experiment can run end-to-end with minimal steps."""
        experiment = SelfConditionedExperiment(small_config)

        # Run the experiment
        results = experiment.run(tokenizer)

        # Verify results structure
        assert results is not None
        assert hasattr(results, "standard_model_results")
        assert hasattr(results, "self_conditioned_model_results")
        assert hasattr(results, "config")
        assert hasattr(results, "timestamp")
        assert hasattr(results, "training_time")
        assert hasattr(results, "parameter_counts")

        # Verify results contain data for the seed
        assert "42" in results.standard_model_results
        assert "42" in results.self_conditioned_model_results

        # Verify training times are recorded
        assert "standard" in results.training_time
        assert "self_conditioned" in results.training_time

        # Verify parameter counts are recorded
        assert "standard" in results.parameter_counts
        assert "self_conditioned" in results.parameter_counts

        # Verify results were saved to disk
        results_dir = Path(small_config.results_dir)
        assert results_dir.exists()

        # Check for expected files
        assert (results_dir / "experiment.log").exists()
        # Results JSON should exist (saved by save_results)
        json_files = list(results_dir.glob("*.json"))
        assert len(json_files) > 0

    @pytest.mark.integration
    def test_parameter_count_validation(self, small_config):
        """Test parameter count validation between models."""
        experiment = SelfConditionedExperiment(small_config)
        standard_model, self_conditioned_model = experiment.create_models()

        # Get parameter counts
        std_params = sum(p.numel() for p in standard_model.parameters())
        sc_params = sum(p.numel() for p in self_conditioned_model.parameters())

        # Calculate parameter difference
        param_diff = sc_params - std_params

        # The difference should be positive (self-conditioned model has extra components)
        assert param_diff > 0

        # Calculate relative difference
        relative_diff = param_diff / std_params

        # The relative difference should be reasonable (not too large)
        # Self-conditioned adds loss predictor and adaptive layers
        assert relative_diff < 0.3  # Less than 30% more parameters

        # Log for debugging
        print(f"Standard model params: {std_params}")
        print(f"Self-conditioned model params: {sc_params}")
        print(f"Relative difference: {relative_diff:.2%}")
