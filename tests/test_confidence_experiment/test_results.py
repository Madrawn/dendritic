# ruff: noqa: PLR6301, PLR2004

"""
Unit tests for confidence experiment results module.
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import patch

from dendritic.experiments.confidence.results import (
    ConfidenceTrainingResult,
    ConfidenceExperimentResults,
    save_results,
    load_results,
    create_results_filename,
    _convert_to_serializable,
)
from dendritic.experiments.confidence.config import ConfidenceExperimentConfig
from dendritic.experiments.utils.TrainingResult import TrainingResult


class TestConfidenceTrainingResult:
    """Tests for ConfidenceTrainingResult dataclass."""

    @pytest.mark.unit
    def test_initialization(self):
        """Test that ConfidenceTrainingResult can be initialized with all fields."""
        # Create a base TrainingResult
        _ = TrainingResult(
            model_type="confidence",
            seed=42,
            final_train_loss=1.5,
            final_eval_loss=2.0,
            final_perplexity=7.5,
            best_eval_loss=1.8,
            best_perplexity=6.0,
            loss_history=[{"step": 0, "train_loss": 3.0, "eval_loss": 3.5}],
            training_time=120.5,
            config={"vocab_size": 1000},
        )

        # Create ConfidenceTrainingResult
        result = ConfidenceTrainingResult(
            model_type="confidence",
            seed=42,
            final_train_loss=1.5,
            final_eval_loss=2.0,
            final_perplexity=7.5,
            best_eval_loss=1.8,
            best_perplexity=6.0,
            loss_history=[{"step": 0, "train_loss": 3.0, "eval_loss": 3.5}],
            training_time=120.5,
            config={"vocab_size": 1000},
            confidence_loss_history=[0.5, 0.4, 0.3],
            token_loss_history=[2.0, 1.8, 1.6],
            loss_predictions=[0.8, 0.7, 0.6],
            actual_future_losses=[1.2, 1.1, 1.0],
        )

        # Verify base fields
        assert result.model_type == "confidence"
        assert result.seed == 42
        assert result.final_train_loss == 1.5
        assert result.final_eval_loss == 2.0
        assert result.final_perplexity == 7.5

        # Verify confidence-specific fields
        assert result.confidence_loss_history == [0.5, 0.4, 0.3]
        assert result.token_loss_history == [2.0, 1.8, 1.6]
        assert result.loss_predictions == [0.8, 0.7, 0.6]
        assert result.actual_future_losses == [1.2, 1.1, 1.0]

    @pytest.mark.unit
    def test_default_fields(self):
        """Test that ConfidenceTrainingResult uses default_factory for lists."""
        result = ConfidenceTrainingResult(
            model_type="confidence",
            seed=42,
            final_train_loss=1.5,
            final_eval_loss=2.0,
            final_perplexity=7.5,
            best_eval_loss=1.8,
            best_perplexity=6.0,
            loss_history=[],
            training_time=120.5,
            config={},
        )

        # Default fields should be empty lists
        assert result.confidence_loss_history == []
        assert result.token_loss_history == []
        assert result.loss_predictions == []
        assert result.actual_future_losses == []


# Module-level fixtures for use across test classes
@pytest.fixture
def sample_config():
    """Create a sample ConfidenceExperimentConfig."""
    return ConfidenceExperimentConfig(
        vocab_size=1000,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        max_seq_len=32,
        batch_size=4,
        training_steps=10,
        seeds=[42, 43],
    )


class TestConfidenceExperimentResults:
    """Tests for ConfidenceExperimentResults dataclass."""

    @pytest.fixture
    def sample_training_result(self):
        """Create a sample TrainingResult."""
        return TrainingResult(
            model_type="standard",
            seed=42,
            final_train_loss=1.5,
            final_eval_loss=2.0,
            final_perplexity=7.5,
            best_eval_loss=1.8,
            best_perplexity=6.0,
            loss_history=[{"step": 0, "train_loss": 3.0, "eval_loss": 3.5}],
            training_time=120.5,
            config={"vocab_size": 1000},
        )

    @pytest.fixture
    def sample_confidence_result(self):
        """Create a sample ConfidenceTrainingResult."""
        return ConfidenceTrainingResult(
            model_type="confidence",
            seed=42,
            final_train_loss=1.6,
            final_eval_loss=2.1,
            final_perplexity=7.8,
            best_eval_loss=1.9,
            best_perplexity=6.2,
            loss_history=[{"step": 0, "train_loss": 3.1, "eval_loss": 3.6}],
            training_time=130.5,
            config={"vocab_size": 1000},
            confidence_loss_history=[0.5, 0.4, 0.3],
            token_loss_history=[2.0, 1.8, 1.6],
            loss_predictions=[0.8, 0.7, 0.6],
            actual_future_losses=[1.2, 1.1, 1.0],
        )

    @pytest.mark.unit
    def test_initialization(self, sample_config, sample_training_result, sample_confidence_result):
        """Test that ConfidenceExperimentResults can be initialized."""
        results = ConfidenceExperimentResults(
            standard_model_results={"42": [sample_training_result]},
            confidence_model_results={"42": [sample_confidence_result]},
            config=sample_config,
            timestamp="2024-01-01_12:00:00",
            training_time={"standard": 120.5, "confidence": 130.5},
            parameter_counts={"standard": 1000000, "confidence": 1200000},
        )

        assert len(results.standard_model_results["42"]) == 1
        assert len(results.confidence_model_results["42"]) == 1
        assert results.config == sample_config
        assert results.timestamp == "2024-01-01_12:00:00"
        assert results.training_time["standard"] == 120.5
        assert results.training_time["confidence"] == 130.5
        assert results.parameter_counts["standard"] == 1000000

    @pytest.mark.unit
    def test_to_dict(self, sample_config, sample_training_result, sample_confidence_result):
        """Test to_dict method converts results to serializable dictionary."""
        results = ConfidenceExperimentResults(
            standard_model_results={"42": [sample_training_result]},
            confidence_model_results={"42": [sample_confidence_result]},
            config=sample_config,
            timestamp="2024-01-01_12:00:00",
            training_time={"standard": 120.5, "confidence": 130.5},
        )

        result_dict = results.to_dict()

        # Check top-level fields
        assert "standard_model_results" in result_dict
        assert "confidence_model_results" in result_dict
        assert "config" in result_dict
        assert "timestamp" in result_dict
        assert "training_time" in result_dict

        # Config should be converted to dict
        assert isinstance(result_dict["config"], dict)
        assert result_dict["config"]["vocab_size"] == 1000

        # Results should be serializable
        assert isinstance(result_dict["standard_model_results"], dict)
        assert isinstance(result_dict["confidence_model_results"], dict)

    @pytest.mark.unit
    def test_from_dict(self, sample_config):
        """Test from_dict method reconstructs results from dictionary."""
        # Create a dict representation
        data = {
            "standard_model_results": {
                "42": [
                    {
                        "model_type": "standard",
                        "seed": 42,
                        "final_train_loss": 1.5,
                        "final_eval_loss": 2.0,
                        "final_perplexity": 7.5,
                        "best_eval_loss": 1.8,
                        "best_perplexity": 6.0,
                        "loss_history": [{"step": 0, "train_loss": 3.0, "eval_loss": 3.5}],
                        "training_time": 120.5,
                        "config": {"vocab_size": 1000},
                    }
                ]
            },
            "confidence_model_results": {
                "42": [
                    {
                        "model_type": "confidence",
                        "seed": 42,
                        "final_train_loss": 1.6,
                        "final_eval_loss": 2.1,
                        "final_perplexity": 7.8,
                        "best_eval_loss": 1.9,
                        "best_perplexity": 6.2,
                        "loss_history": [{"step": 0, "train_loss": 3.1, "eval_loss": 3.6}],
                        "training_time": 130.5,
                        "config": {"vocab_size": 1000},
                        "confidence_loss_history": [0.5, 0.4, 0.3],
                        "token_loss_history": [2.0, 1.8, 1.6],
                        "loss_predictions": [0.8, 0.7, 0.6],
                        "actual_future_losses": [1.2, 1.1, 1.0],
                    }
                ]
            },
            "config": {
                "vocab_size": 1000,
                "embed_dim": 64,
                "num_heads": 4,
                "num_layers": 2,
                "max_seq_len": 32,
                "batch_size": 4,
                "training_steps": 10,
                "seeds": [42, 43],
                "confidence_alpha": 1.0,
                "lookahead_steps": 2,
                "results_dir": "results/confidence_experiments",
            },
            "timestamp": "2024-01-01_12:00:00",
            "training_time": {"standard": 120.5, "confidence": 130.5},
            "parameter_counts": {},
        }

        results = ConfidenceExperimentResults.from_dict(data)

        assert results.timestamp == "2024-01-01_12:00:00"
        assert results.config.vocab_size == 1000
        assert len(results.standard_model_results["42"]) == 1
        assert len(results.confidence_model_results["42"]) == 1
        assert results.training_time["standard"] == 120.5


class TestSerializationFunctions:
    """Tests for serialization helper functions."""

    @pytest.mark.unit
    def test_convert_to_serializable_numpy(self):
        """Test _convert_to_serializable handles numpy types."""
        # Test numpy integers
        np_int = np.int32(42)
        result = _convert_to_serializable(np_int)
        assert isinstance(result, int)
        assert result == 42

        # Test numpy floats
        np_float = np.float64(3.14)
        result = _convert_to_serializable(np_float)
        assert isinstance(result, float)
        assert result == 3.14

        # Test numpy bool
        np_bool = np.bool_(True)
        result = _convert_to_serializable(np_bool)
        assert isinstance(result, bool)
        assert result is True

        # Test numpy array
        np_array = np.array([1, 2, 3])
        result = _convert_to_serializable(np_array)
        assert isinstance(result, list)
        assert result == [1, 2, 3]

    @pytest.mark.unit
    def test_convert_to_serializable_dict(self):
        """Test _convert_to_serializable handles nested structures."""
        data = {
            "int": np.int32(42),
            "float": np.float64(3.14),
            "list": [np.int32(1), np.int32(2)],
            "nested": {"array": np.array([1, 2, 3])},
        }

        result = _convert_to_serializable(data)

        # All values should be native Python types
        assert isinstance(result["int"], int)
        assert isinstance(result["float"], float)
        assert isinstance(result["list"], list)
        assert isinstance(result["list"][0], int)
        assert isinstance(result["nested"]["array"], list)

    @pytest.mark.unit
    def test_create_results_filename(self):
        """Test create_results_filename generates correct format."""
        with patch("dendritic.experiments.confidence.results.datetime") as mock_datetime:
            mock_datetime.now.return_value.strftime.return_value = "20240101_120000"

            filename = create_results_filename()
            assert filename == "confidence_20240101_120000.json"

            filename = create_results_filename("test")
            assert filename == "test_20240101_120000.json"

    @pytest.mark.unit
    def test_save_and_load_results(self, sample_config, tmp_path):
        """Test save_results and load_results round-trip."""
        # Create sample results
        training_result = TrainingResult(
            model_type="standard",
            seed=42,
            final_train_loss=1.5,
            final_eval_loss=2.0,
            final_perplexity=7.5,
            best_eval_loss=1.8,
            best_perplexity=6.0,
            loss_history=[{"step": 0, "train_loss": 3.0, "eval_loss": 3.5}],
            training_time=120.5,
            config={"vocab_size": 1000},
        )

        confidence_result = ConfidenceTrainingResult(
            model_type="confidence",
            seed=42,
            final_train_loss=1.6,
            final_eval_loss=2.1,
            final_perplexity=7.8,
            best_eval_loss=1.9,
            best_perplexity=6.2,
            loss_history=[{"step": 0, "train_loss": 3.1, "eval_loss": 3.6}],
            training_time=130.5,
            config={"vocab_size": 1000},
            confidence_loss_history=[0.5, 0.4, 0.3],
            token_loss_history=[2.0, 1.8, 1.6],
            loss_predictions=[0.8, 0.7, 0.6],
            actual_future_losses=[1.2, 1.1, 1.0],
        )

        results = ConfidenceExperimentResults(
            standard_model_results={"42": [training_result]},
            confidence_model_results={"42": [confidence_result]},
            config=sample_config,
            timestamp="2024-01-01_12:00:00",
            training_time={"standard": 120.5, "confidence": 130.5},
        )

        # Save results
        save_path = save_results(results, tmp_path, "test_results.json")

        # Verify file was created
        assert save_path.exists()
        assert save_path.name == "test_results.json"

        # Load results back
        loaded_results = load_results(save_path)

        # Verify loaded results match original
        assert loaded_results.timestamp == results.timestamp
        assert loaded_results.config.vocab_size == results.config.vocab_size
        assert loaded_results.training_time["standard"] == results.training_time["standard"]

        # Verify results structure
        assert "42" in loaded_results.standard_model_results
        assert "42" in loaded_results.confidence_model_results

    @pytest.mark.unit
    def test_save_results_creates_directory(self, sample_config, tmp_path):
        """Test save_results creates directory if it doesn't exist."""
        results_dir = tmp_path / "nonexistent" / "subdir"

        # Create minimal results
        results = ConfidenceExperimentResults(
            standard_model_results={},
            confidence_model_results={},
            config=sample_config,
            timestamp="2024-01-01_12:00:00",
            training_time={},
        )

        # Should create directory automatically
        save_path = save_results(results, results_dir, "test.json")
        assert results_dir.exists()
        assert save_path.exists()

    @pytest.mark.unit
    def test_load_results_file_not_found(self):
        """Test load_results raises appropriate error for missing file."""
        with pytest.raises(FileNotFoundError):
            load_results(Path("/nonexistent/path/results.json"))


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.mark.unit
    def test_empty_results(self, sample_config):
        """Test handling of empty results."""
        results = ConfidenceExperimentResults(
            standard_model_results={},
            confidence_model_results={},
            config=sample_config,
            timestamp="2024-01-01_12:00:00",
            training_time={},
        )

        # Should serialize without error
        result_dict = results.to_dict()
        assert result_dict["standard_model_results"] == {}
        assert result_dict["confidence_model_results"] == {}

    @pytest.mark.unit
    def test_results_with_numpy_values(self, sample_config):
        """Test serialization with numpy values in results."""
        import numpy as np

        # Test _convert_to_serializable directly with numpy values
        test_data = {
            "int": np.int32(42),
            "float": np.float64(3.14),
            "array": np.array([1, 2, 3]),
            "nested": {
                "bool": np.bool_(True),
                "list": [np.float32(1.5), np.float64(2.5)],
            },
        }

        converted = _convert_to_serializable(test_data)

        # Check that all values are native Python types
        assert isinstance(converted["int"], int)
        assert isinstance(converted["float"], float)
        assert isinstance(converted["array"], list)
        assert isinstance(converted["nested"]["bool"], bool)
        assert isinstance(converted["nested"]["list"][0], float)
        assert isinstance(converted["nested"]["list"][1], float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
