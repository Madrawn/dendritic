"""
Tests for the ConfidenceAwareExperiment class.
"""

import pytest
import torch
from unittest.mock import Mock, patch
from dendritic.experiments.confidence import (
    ConfidenceExperimentConfig,
    ConfidenceAwareExperiment,
    ConfidenceTrainingResult,
    ConfidenceExperimentResults,
)
from dendritic.experiments.models.MiniGPT import MiniGPT, ConfidenceAwareGPT


class TestConfidenceAwareExperiment:
    """Test suite for ConfidenceAwareExperiment."""

    @pytest.mark.unit
    def test_config_initialization(self):
        """Test that ConfidenceExperimentConfig can be initialized."""
        config = ConfidenceExperimentConfig(
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
        assert config.confidence_alpha == 1.0  # Default value
        assert config.lookahead_steps == 2  # Default value
        assert config.results_dir == "results/confidence_experiments"

    @pytest.mark.unit
    def test_experiment_initialization(self):
        """Test that ConfidenceAwareExperiment can be initialized."""
        config = ConfidenceExperimentConfig(
            vocab_size=1000,
            embed_dim=64,
            num_heads=4,
            num_layers=2,
            max_seq_len=32,
            batch_size=4,
            training_steps=10,
            seeds=[42],
        )

        experiment = ConfidenceAwareExperiment(config)

        assert experiment.config == config
        assert experiment.device in ["cuda", "cpu"]
        assert experiment.results_dir.exists()

    @pytest.mark.unit
    def test_create_models(self):
        """Test that create_models returns both model variants."""
        config = ConfidenceExperimentConfig(
            vocab_size=1000,
            embed_dim=64,
            num_heads=4,
            num_layers=2,
            max_seq_len=32,
            batch_size=4,
            training_steps=10,
            seeds=[42],
        )

        experiment = ConfidenceAwareExperiment(config)
        standard_model, confidence_model = experiment.create_models()

        assert isinstance(standard_model, MiniGPT)
        assert isinstance(confidence_model, ConfidenceAwareGPT)

        # Check that models have different architectures
        assert hasattr(confidence_model, "confidence_predictor")
        assert not hasattr(standard_model, "confidence_predictor")

        # Log parameter counts (should be printed)
        std_params = sum(p.numel() for p in standard_model.parameters())
        conf_params = sum(p.numel() for p in confidence_model.parameters())

        # Confidence model should have more parameters due to confidence predictor
        assert conf_params > std_params

    @pytest.mark.unit
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_placement(self):
        """Test that models can be moved to CUDA if available."""
        config = ConfidenceExperimentConfig(
            vocab_size=1000,
            embed_dim=64,
            num_heads=4,
            num_layers=2,
            max_seq_len=32,
            batch_size=4,
            training_steps=10,
            seeds=[42],
        )

        experiment = ConfidenceAwareExperiment(config)
        standard_model, confidence_model = experiment.create_models()

        # Move to CUDA
        standard_model = standard_model.to("cuda")
        confidence_model = confidence_model.to("cuda")

        # Check device placement
        assert next(standard_model.parameters()).is_cuda
        assert next(confidence_model.parameters()).is_cuda

    @pytest.mark.unit
    def test_dataclasses(self):
        """Test that dataclasses can be instantiated."""
        # Test ConfidenceTrainingResult
        confidence_result = ConfidenceTrainingResult(
            model_type="confidence",
            seed=42,
            final_train_loss=2.5,
            final_eval_loss=2.8,
            final_perplexity=16.44,
            best_eval_loss=2.7,
            best_perplexity=14.88,
            loss_history=[],
            training_time=100.0,
            config={},
            confidence_loss_history=[0.5, 0.4, 0.3],
            token_loss_history=[2.5, 2.4, 2.3],
            confidence_predictions=[0.1, 0.2, 0.3],
            actual_future_losses=[2.6, 2.5, 2.4],
        )

        assert confidence_result.model_type == "confidence"
        assert confidence_result.seed == 42
        assert len(confidence_result.confidence_loss_history) == 3

        # Test ConfidenceExperimentResults
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

        results = ConfidenceExperimentResults(
            standard_model_results={"42": [standard_result]},
            confidence_model_results={"42": [confidence_result]},
            config=ConfidenceExperimentConfig(),
            timestamp="2024-01-01T00:00:00",
            training_time={"standard": 95.0, "confidence": 110.0},
            parameter_counts={"standard": 1000, "confidence": 1200},
        )

        assert len(results.standard_model_results) == 1
        assert len(results.confidence_model_results) == 1
        assert results.parameter_counts["confidence"] > results.parameter_counts["standard"]

    @pytest.mark.unit
    @patch("dendritic.experiments.confidence.experiment.prepare_confidence_data")
    @patch("dendritic.experiments.confidence.experiment.ConfidenceAwareExperiment.train_standard_model")
    @patch("dendritic.experiments.confidence.experiment.ConfidenceAwareExperiment.train_confidence_model")
    @patch("dendritic.experiments.confidence.experiment.save_results")
    def test_run_method_mocked(self, mock_save_results, mock_train_conf, mock_train_std, mock_prepare_data):
        """Test the run method with mocked dependencies."""
        # Setup mocks
        mock_tokenizer = Mock()
        mock_dataloaders = {"train": Mock(), "eval": Mock()}
        mock_prepare_data.return_value = mock_dataloaders

        # Use actual TrainingResult objects to avoid numpy recursion issues
        from dendritic.experiments.utils.TrainingResult import TrainingResult

        mock_std_result = TrainingResult(
            model_type="standard",
            seed=42,
            final_train_loss=2.0,
            final_eval_loss=2.5,
            final_perplexity=12.0,
            best_eval_loss=2.3,
            best_perplexity=10.0,
            loss_history=[],
            training_time=120.5,
            config={"test": "config"},
        )
        mock_train_std.return_value = mock_std_result

        mock_conf_result = ConfidenceTrainingResult(
            model_type="confidence",
            seed=42,
            final_train_loss=2.1,
            final_eval_loss=2.6,
            final_perplexity=12.5,
            best_eval_loss=2.4,
            best_perplexity=10.5,
            loss_history=[],
            training_time=130.5,
            config={"test": "config"},
            confidence_loss_history=[0.5, 0.4, 0.3],
            token_loss_history=[2.0, 1.8, 1.6],
            confidence_predictions=[0.8, 0.7, 0.6],
            actual_future_losses=[1.2, 1.1, 1.0],
        )
        mock_train_conf.return_value = mock_conf_result

        # Create config and experiment
        config = ConfidenceExperimentConfig(
            vocab_size=1000,
            embed_dim=64,
            num_heads=4,
            num_layers=2,
            max_seq_len=32,
            batch_size=4,
            training_steps=10,
            seeds=[42],
        )

        experiment = ConfidenceAwareExperiment(config)

        # Mock create_models to avoid actual model creation
        with patch.object(experiment, "create_models") as mock_create:
            mock_std_model = Mock()
            mock_conf_model = Mock()
            # Mock parameters() to return an empty list
            mock_std_model.parameters.return_value = []
            mock_conf_model.parameters.return_value = []
            mock_create.return_value = (mock_std_model, mock_conf_model)

            # Run the experiment
            results = experiment.run(mock_tokenizer)

        # Verify calls
        mock_prepare_data.assert_called_once_with(
            config=config,
            tokenizer=mock_tokenizer,
            dataset_kwargs=config.dataset_kwargs,
        )

        mock_train_std.assert_called_once_with(
            mock_std_model,
            mock_dataloaders["train"],
            mock_dataloaders["eval"],
            experiment.device,
            42,
            mock_tokenizer,
        )

        mock_train_conf.assert_called_once_with(
            mock_conf_model,
            mock_dataloaders["train"],
            mock_dataloaders["eval"],
            experiment.device,
            42,
            mock_tokenizer,
        )

        # Verify results structure
        assert isinstance(results, ConfidenceExperimentResults)
        assert "42" in results.standard_model_results
        assert "42" in results.confidence_model_results

        # Verify save_results was called
        mock_save_results.assert_called_once()

    @pytest.mark.unit
    def test_two_pass_training_step_signature(self):
        """Test that ConfidenceAwareGPT.two_pass_training_step has correct signature."""
        import inspect

        sig = inspect.signature(ConfidenceAwareGPT.two_pass_training_step)

        # Check required parameters (updated to match refactored signature)
        expected_params = [
            "model",
            "tokens_t",
            "tokens_t_plus_1",
            "alpha",
        ]
        for param in expected_params:
            assert param in sig.parameters

        # Check alpha has default value
        assert sig.parameters["alpha"].default == 1.0

    @pytest.mark.unit
    def test_dataclass_serialization(self):
        """Test that dataclasses can be serialized to JSON."""
        import json

        # Create ConfidenceTrainingResult
        confidence_result = ConfidenceTrainingResult(
            model_type="confidence",
            seed=42,
            final_train_loss=2.5,
            final_eval_loss=2.8,
            final_perplexity=16.44,
            best_eval_loss=2.7,
            best_perplexity=14.88,
            loss_history=[],
            training_time=100.0,
            config={"test": "value"},
            confidence_loss_history=[0.5, 0.4, 0.3],
            token_loss_history=[2.5, 2.4, 2.3],
            confidence_predictions=[0.1, 0.2, 0.3],
            actual_future_losses=[2.6, 2.5, 2.4],
        )

        # Convert to dict (simulating what _save_results does)
        result_dict = confidence_result.__dict__

        # Should be JSON serializable
        json_str = json.dumps(result_dict)
        assert isinstance(json_str, str)

        # Deserialize and verify
        loaded_dict = json.loads(json_str)
        assert loaded_dict["model_type"] == "confidence"
        assert loaded_dict["seed"] == 42
        assert len(loaded_dict["confidence_loss_history"]) == 3

        # Test ConfidenceExperimentResults serialization
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

        results = ConfidenceExperimentResults(
            standard_model_results={"42": [standard_result]},
            confidence_model_results={"42": [confidence_result]},
            config=ConfidenceExperimentConfig(),
            timestamp="2024-01-01T00:00:00",
            training_time={"standard": 95.0, "confidence": 110.0},
            parameter_counts={"standard": 1000, "confidence": 1200},
        )

        # Use the same conversion logic as in experiment.py
        def convert_to_serializable(obj):
            if isinstance(obj, (TrainingResult, ConfidenceTrainingResult)):
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
        assert "42" in loaded_dict["confidence_model_results"]
        assert loaded_dict["parameter_counts"]["confidence"] == 1200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
