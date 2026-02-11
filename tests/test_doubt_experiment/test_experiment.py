# ruff: noqa: PLR6301, PLR2004
"""
Tests for the DoubtAwareExperiment class.
"""

import pytest
import torch
from unittest.mock import Mock, patch
from dendritic.experiments.doubt import (
    DoubtExperimentConfig,
    DoubtTrainingResult,
    DoubtExperimentResults,
)
from dendritic.experiments.models.doubt_conditioning.DoubtAwareGPT import DoubtAwareGPT
from dendritic.experiments.models.MiniGPT import MiniGPT


class TestDoubtAwareExperiment:
    """Test suite for DoubtAwareExperiment."""

    @pytest.mark.unit
    def test_config_initialization(self):
        """Test that DoubtExperimentConfig can be initialized."""
        config = DoubtExperimentConfig(
            vocab_size=1000,
            embed_dim=64,
            num_heads=4,
            num_layers=2,
            max_seq_len=32,
            batch_size=4,
            training_steps=10,
            seeds=[42],
        )

        assert config.vocab_size == 1000
        assert config.embed_dim == 64
        assert config.doubt_alpha == 1.0  # Default value
        assert config.doubt_vector_dim == 1  # Default value
        assert config.results_dir == "results/doubt_experiments"

    @pytest.mark.unit
    def test_dataclasses(self):
        """Test that dataclasses can be instantiated."""
        # Test DoubtTrainingResult
        doubt_result = DoubtTrainingResult(
            model_type="doubt",
            seed=42,
            final_train_loss=2.5,
            final_eval_loss=2.8,
            final_perplexity=16.44,
            best_eval_loss=2.7,
            best_perplexity=14.88,
            loss_history=[],
            training_time=100.0,
            config={},
            doubt_loss_history=[0.5, 0.4, 0.3],
            token_loss_history=[2.5, 2.4, 2.3],
            loss_predictions=[0.1, 0.2, 0.3],
            actual_future_losses=[2.6, 2.5, 2.4],
        )

        assert doubt_result.model_type == "doubt"
        assert doubt_result.seed == 42
        assert len(doubt_result.doubt_loss_history) == 3

        # Test DoubtExperimentResults
        from dendritic.experiments.utils.TrainingResult import TrainingResult

        standard_result = TrainingResult(
            model_type="standard",
            seed=42,
            final_train_loss=2.6,
            final_eval_loss=2.9,
            final_perplexity=18.17,
            best_eval_loss=2.8,
            best_perplexity=16.44,
            loss_history=[],
            training_time=95.0,
            config={},
        )

        results = DoubtExperimentResults(
            standard_model_results={"42": [standard_result]},
            doubt_model_results={"42": [doubt_result]},
            config=DoubtExperimentConfig(),
            timestamp="2024-01-01T00:00:00",
            training_time={"standard": 95.0, "doubt": 110.0},
            parameter_counts={"standard": 1000, "doubt": 1200},
        )

        assert len(results.standard_model_results) == 1
        assert len(results.doubt_model_results) == 1
        assert results.parameter_counts["doubt"] > results.parameter_counts["standard"]

    @pytest.mark.unit
    def test_dataclass_serialization(self):
        """Test that dataclasses can be serialized to JSON."""
        import json

        # Create DoubtTrainingResult
        doubt_result = DoubtTrainingResult(
            model_type="doubt",
            seed=42,
            final_train_loss=2.5,
            final_eval_loss=2.8,
            final_perplexity=16.44,
            best_eval_loss=2.7,
            best_perplexity=14.88,
            loss_history=[],
            training_time=100.0,
            config={"test": "value"},
            doubt_loss_history=[0.5, 0.4, 0.3],
            token_loss_history=[2.5, 2.4, 2.3],
            loss_predictions=[0.1, 0.2, 0.3],
            actual_future_losses=[2.6, 2.5, 2.4],
        )

        # Convert to dict (simulating what _save_results does)
        result_dict = doubt_result.__dict__

        # Should be JSON serializable
        json_str = json.dumps(result_dict)
        assert isinstance(json_str, str)

        # Deserialize and verify
        loaded_dict = json.loads(json_str)
        assert loaded_dict["model_type"] == "doubt"
        assert loaded_dict["seed"] == 42
        assert len(loaded_dict["doubt_loss_history"]) == 3

        # Test DoubtExperimentResults serialization
        from dendritic.experiments.utils.TrainingResult import TrainingResult

        standard_result = TrainingResult(
            model_type="standard",
            seed=42,
            final_train_loss=2.6,
            final_eval_loss=2.9,
            final_perplexity=18.17,
            best_eval_loss=2.8,
            best_perplexity=16.44,
            loss_history=[],
            training_time=95.0,
            config={"test": "value"},
        )

        results = DoubtExperimentResults(
            standard_model_results={"42": [standard_result]},
            doubt_model_results={"42": [doubt_result]},
            config=DoubtExperimentConfig(),
            timestamp="2024-01-01T00:00:00",
            training_time={"standard": 95.0, "doubt": 110.0},
            parameter_counts={"standard": 1000, "doubt": 1200},
        )

        # Use the same conversion logic as in experiment.py
        def convert_to_serializable(obj):
            if isinstance(obj, (TrainingResult, DoubtTrainingResult)):
                return obj.__dict__
            elif hasattr(obj, "__dict__"):  # Handle dataclasses and other objects
                return convert_to_serializable(obj.__dict__)
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            else:
                return obj

        # Convert to serializable dict
        serializable_results = convert_to_serializable(results.__dict__)

        # Should be JSON serializable
        json_str = json.dumps(serializable_results)
        assert isinstance(json_str, str)

        # Deserialize and verify
        loaded_dict = json.loads(json_str)
        assert "42" in loaded_dict["standard_model_results"]
        assert "42" in loaded_dict["doubt_model_results"]
        assert loaded_dict["parameter_counts"]["doubt"] == 1200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
