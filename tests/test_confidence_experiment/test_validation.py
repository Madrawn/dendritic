# ruff: noqa: PLR6301, PLR2004,E712

"""
Tests for confidence experiment validation functions.
"""

import pytest
import torch
from unittest.mock import Mock, patch
from dendritic.experiments.confidence.validation import (
    compare_parameter_counts,
    validate_loss_predictions,
    validate_experiment_results,
    generate_validation_report,
)
from dendritic.experiments.models.MiniGPT import MiniGPT, ConfidenceAwareGPT
from dendritic.experiments.confidence.config import ConfidenceExperimentConfig
from dendritic.experiments.confidence.results import (
    ConfidenceTrainingResult,
    ConfidenceExperimentResults,
)
from dendritic.experiments.utils.TrainingResult import TrainingResult


class TestCompareParameterCounts:
    """Test compare_parameter_counts function."""

    @pytest.fixture
    def mock_standard_model(self):
        """Create a mock standard MiniGPT model."""
        model = Mock(spec=MiniGPT)
        # Mock parameters
        param1 = Mock()
        param1.numel.return_value = 1000
        param2 = Mock()
        param2.numel.return_value = 2000

        model.parameters.return_value = [param1, param2]

        # Mock named_parameters for layer-wise breakdown
        model.named_parameters.return_value = [
            ("tok_emb.weight", param1),
            ("head.weight", param2),
        ]
        return model

    @pytest.fixture
    def mock_confidence_model(self):
        """Create a mock ConfidenceAwareGPT model."""
        model = Mock(spec=ConfidenceAwareGPT)
        # Mock parameters - confidence model should have more parameters
        param1 = Mock()
        param1.numel.return_value = 1000
        param2 = Mock()
        param2.numel.return_value = 2000
        param3 = Mock()  # Extra parameter for confidence predictor
        param3.numel.return_value = 500

        model.parameters.return_value = [param1, param2, param3]

        # Mock named_parameters for layer-wise breakdown
        model.named_parameters.return_value = [
            ("tok_emb.weight", param1),
            ("head.weight", param2),
            ("loss_predictor.weight", param3),
        ]
        return model

    @pytest.mark.unit
    def test_compare_parameter_counts_basic(self, mock_standard_model, mock_confidence_model):
        """Test basic parameter count comparison."""
        result = compare_parameter_counts(mock_standard_model, mock_confidence_model)

        # Check total counts
        assert result["standard_total"] == 3000  # 1000 + 2000
        assert result["confidence_total"] == 3500  # 1000 + 2000 + 500
        assert result["difference"] == 500
        assert result["relative_difference"] == 500 / 3000

        # Check layer-wise breakdown
        assert "standard_by_layer" in result
        assert "confidence_by_layer" in result
        assert result["standard_by_layer"]["tok_emb.weight"] == 1000
        assert result["confidence_by_layer"]["loss_predictor.weight"] == 500

        # Check fair comparison flag (less than 30% more params)
        assert result["is_fair_comparison"] == True  # 500/3000 = 16.7% < 30%

    @pytest.mark.unit
    def test_compare_parameter_counts_unfair(self):
        """Test parameter comparison when confidence model has too many parameters."""
        # Create mocks with large difference
        std_model = Mock(spec=MiniGPT)
        std_param = Mock()
        std_param.numel.return_value = 1000
        std_model.parameters.return_value = [std_param]
        std_model.named_parameters.return_value = [("layer.weight", std_param)]

        conf_model = Mock(spec=ConfidenceAwareGPT)
        conf_param1 = Mock()
        conf_param1.numel.return_value = 1000
        conf_param2 = Mock()  # Large extra parameter
        conf_param2.numel.return_value = 1000  # 100% increase
        conf_model.parameters.return_value = [conf_param1, conf_param2]
        conf_model.named_parameters.return_value = [
            ("layer.weight", conf_param1),
            ("loss_predictor.weight", conf_param2),
        ]

        result = compare_parameter_counts(std_model, conf_model)

        # Relative difference should be 1.0 (100% increase)
        assert result["relative_difference"] == 1.0
        assert result["is_fair_comparison"] == False  # 100% > 30%

    @pytest.mark.unit
    def test_compare_parameter_counts_zero_standard(self):
        """Test edge case where standard model has zero parameters."""
        std_model = Mock(spec=MiniGPT)
        std_model.parameters.return_value = []
        std_model.named_parameters.return_value = []

        conf_model = Mock(spec=ConfidenceAwareGPT)
        conf_param = Mock()
        conf_param.numel.return_value = 1000
        conf_model.parameters.return_value = [conf_param]
        conf_model.named_parameters.return_value = [("layer.weight", conf_param)]

        result = compare_parameter_counts(std_model, conf_model)

        # With zero standard params, relative difference should be inf
        assert result["standard_total"] == 0
        assert result["confidence_total"] == 1000
        assert result["difference"] == 1000
        assert result["relative_difference"] == float("inf")
        assert result["is_fair_comparison"] == False  # inf > 30%


class TestValidateLossPredictions:
    """Test validate_loss_predictions function."""

    @pytest.fixture
    def mock_confidence_model(self):
        """Create a mock ConfidenceAwareGPT model."""
        model = Mock(spec=ConfidenceAwareGPT)

        # Mock forward method to return loss prediction values
        # Updated to match new API: labels=None, confidence_scalars=None
        def mock_forward(input_ids, labels=None, confidence_scalars=None):
            batch_size, seq_len = input_ids.shape
            # Create loss prediction values (raw linear outputs)
            # Range ~2-4 due to loss_init_bias=2.0
            loss_pred = 2.0 + torch.randn(batch_size, seq_len) * 0.5

            return {"loss_prediction": loss_pred}

        model.eval = Mock()
        model.return_value = Mock()
        model.side_effect = mock_forward

        return model

    @pytest.mark.unit
    def test_validate_loss_predictions_valid(self, mock_confidence_model):
        """Test validation with valid loss predictions."""
        # Create input tensors
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones_like(input_ids)

        # Mock the model to return valid loss predictions (around 2.0)
        loss_pred = 2.0 + torch.randn(batch_size, seq_len) * 0.5  # Values around 2.0
        mock_output = {"loss_prediction": loss_pred}

        with patch.object(mock_confidence_model, "__call__", return_value=mock_output):
            result = validate_loss_predictions(mock_confidence_model, input_ids, attention_mask, threshold=0.95)

        # Check result structure
        assert result["shape"] == (batch_size, seq_len)
        assert result["has_nan"] == False
        assert result["has_inf"] == False
        # percent_in_range no longer exists in new validation function
        assert result["passes_range_check"] == True  # Mean should be around 2.0
        assert result["passes_sanity_check"] == True
        assert result["is_valid"] == True

        # Check statistics
        stats = result["statistics"]
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "median" in stats
        # Values are raw predictions, not bounded to [0, 1]
        # Just check they're reasonable (not checking bounds)

    @pytest.mark.unit
    def test_validate_loss_predictions_nan(self):
        """Test validation with NaN values."""
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones_like(input_ids)

        # Create loss prediction with NaN
        loss_pred = torch.ones(batch_size, seq_len)
        loss_pred[0, 0] = torch.tensor(float("nan"))

        # Create a mock model that returns our loss prediction
        mock_model = Mock(spec=ConfidenceAwareGPT)
        mock_model.eval = Mock()

        # Mock the __call__ method to return dict with loss_prediction
        mock_output = {"loss_prediction": loss_pred}
        mock_model.return_value = mock_output

        result = validate_loss_predictions(mock_model, input_ids, attention_mask)

        assert result["has_nan"] == True
        assert result["passes_sanity_check"] == False
        assert result["is_valid"] == False

    @pytest.mark.unit
    def test_validate_loss_predictions_out_of_range(self, mock_confidence_model):
        """Test validation with unreasonable mean/std values."""
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones_like(input_ids)

        # Create loss prediction with unreasonable mean (far from 2.0)
        # Mean = 50.0, which is > 10.0 away from 2.0
        loss_pred = torch.tensor([
            [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0],
            [50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0],
        ])

        # Create a mock function that returns our loss prediction dict
        def mock_call(input_ids, labels=None, confidence_scalars=None):
            return {"loss_prediction": loss_pred}

        # Clear side_effect and set our mock
        mock_confidence_model.side_effect = None
        mock_confidence_model.side_effect = mock_call

        result = validate_loss_predictions(mock_confidence_model, input_ids, attention_mask, threshold=0.95)

        # Mean is 50.0, which is 48.0 away from 2.0 > 10.0 threshold
        assert result["passes_range_check"] == False
        assert result["is_valid"] == False
        assert abs(result["mean_value"] - 50.0) < 0.1  # Should be close to 50.0

    @pytest.mark.unit
    def test_validate_loss_predictions_shape_mismatch(self, mock_confidence_model):
        """Test validation with shape mismatch."""
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones_like(input_ids)

        # Create loss prediction with wrong shape
        loss_pred = torch.ones(batch_size, seq_len + 1)  # Wrong shape

        # Create a mock function that returns our loss prediction dict with wrong shape
        def mock_call(input_ids, labels=None, confidence_scalars=None):
            return {"loss_prediction": loss_pred}

        # Clear side_effect and set our mock
        mock_confidence_model.side_effect = None
        mock_confidence_model.side_effect = mock_call

        # Should raise assertion error
        with pytest.raises(AssertionError, match="doesn't match input shape"):
            validate_loss_predictions(mock_confidence_model, input_ids, attention_mask)


class TestValidateExperimentResults:
    """Test validate_experiment_results function."""

    @pytest.fixture
    def sample_config(self):
        """Create a sample configuration."""
        return ConfidenceExperimentConfig(
            vocab_size=1000,
            embed_dim=64,
            num_heads=4,
            num_layers=2,
            max_seq_len=32,
            batch_size=4,
            training_steps=10,
            seeds=[42, 123],
        )

    @pytest.fixture
    def sample_results(self, sample_config):
        """Create sample experiment results."""
        # Create standard training result
        standard_result = TrainingResult(
            model_type="standard",
            seed=42,
            final_train_loss=2.5,
            final_eval_loss=2.8,
            final_perplexity=16.44,
            best_eval_loss=2.7,
            best_perplexity=14.88,
            loss_history=[],
            training_time=100.0,
            config={},
        )

        # Create confidence training result
        confidence_result = ConfidenceTrainingResult(
            model_type="confidence",
            seed=42,
            final_train_loss=2.6,
            final_eval_loss=2.9,
            final_perplexity=18.17,
            best_eval_loss=2.8,
            best_perplexity=16.44,
            loss_history=[],
            training_time=110.0,
            config={},
            confidence_loss_history=[0.5, 0.4, 0.3],
            token_loss_history=[2.5, 2.4, 2.3],
            loss_predictions=[0.1, 0.2, 0.3],
            actual_future_losses=[2.6, 2.5, 2.4],
        )

        # Create experiment results
        results = Mock(spec=ConfidenceExperimentResults)
        results.standard_model_results = {
            "42": [standard_result],
            "123": [standard_result],
        }
        results.confidence_model_results = {
            "42": [confidence_result],
            "123": [confidence_result],
        }
        results.config = sample_config
        results.timestamp = "2024-01-01T00:00:00"
        results.training_time = {"standard": 200.0, "confidence": 220.0}
        results.parameter_counts = {"standard": 1000, "confidence": 1200}

        return results

    @pytest.mark.unit
    def test_validate_experiment_results_valid(self, sample_results, sample_config):
        """Test validation with valid results."""
        validation = validate_experiment_results(sample_results, sample_config)

        # Should be valid
        assert validation["is_valid"] == True
        assert len(validation["errors"]) == 0
        assert len(validation["warnings"]) == 0

        # Should have passed checks
        assert "has_standard_model_results" in validation["checks_passed"]
        assert "has_confidence_model_results" in validation["checks_passed"]
        assert "seed_consistency" in validation["checks_passed"]
        assert "seed_completeness" in validation["checks_passed"]
        assert "standard_training_time_positive" in validation["checks_passed"]
        assert "confidence_training_time_positive" in validation["checks_passed"]
        assert "confidence_has_more_params" in validation["checks_passed"]
        assert "reasonable_param_increase" in validation["checks_passed"]

    @pytest.mark.unit
    def test_validate_experiment_results_missing_attribute(self, sample_config):
        """Test validation with missing required attribute."""
        results = Mock(spec=ConfidenceExperimentResults)
        # Missing standard_model_results
        results.confidence_model_results = {"42": []}
        results.config = sample_config
        results.timestamp = "2024-01-01T00:00:00"
        results.training_time = {}
        results.parameter_counts = {}

        validation = validate_experiment_results(results, sample_config)

        assert validation["is_valid"] == False
        assert any("Missing attribute: standard_model_results" in error for error in validation["errors"])

    @pytest.mark.unit
    def test_validate_experiment_results_seed_mismatch(self, sample_config):
        """Test validation with seed mismatch."""
        results = Mock(spec=ConfidenceExperimentResults)
        results.standard_model_results = {"42": [], "123": []}
        results.confidence_model_results = {"42": []}  # Missing seed 123
        results.config = sample_config
        results.timestamp = "2024-01-01T00:00:00"
        results.training_time = {}
        results.parameter_counts = {}

        validation = validate_experiment_results(results, sample_config)

        assert validation["is_valid"] == False
        assert any("Seed mismatch" in error for error in validation["errors"])

    @pytest.mark.unit
    def test_validate_experiment_results_missing_seeds(self, sample_config):
        """Test validation with missing config seeds."""
        results = Mock(spec=ConfidenceExperimentResults)
        results.standard_model_results = {"42": []}  # Missing seed 123
        results.confidence_model_results = {"42": []}
        results.config = sample_config
        results.timestamp = "2024-01-01T00:00:00"
        results.training_time = {}
        results.parameter_counts = {}

        validation = validate_experiment_results(results, sample_config)

        # Missing seeds should be a warning, not an error
        assert validation["is_valid"] == True  # Still valid
        assert any("Missing seeds" in warning for warning in validation["warnings"])

    @pytest.mark.unit
    def test_validate_experiment_results_negative_training_time(self, sample_config):
        """Test validation with negative training time."""
        results = Mock(spec=ConfidenceExperimentResults)
        results.standard_model_results = {"42": []}
        results.confidence_model_results = {"42": []}
        results.config = sample_config
        results.timestamp = "2024-01-01T00:00:00"
        results.training_time = {"standard": -1.0, "confidence": 100.0}  # Negative time
        results.parameter_counts = {}

        validation = validate_experiment_results(results, sample_config)

        # Negative training time should be a warning
        assert validation["is_valid"] == True  # Still valid
        assert any("training time is -1.0" in warning for warning in validation["warnings"])

    @pytest.mark.unit
    def test_validate_experiment_results_parameter_counts(self, sample_config):
        """Test validation of parameter counts."""
        results = Mock(spec=ConfidenceExperimentResults)
        results.standard_model_results = {"42": []}
        results.confidence_model_results = {"42": []}
        results.config = sample_config
        results.timestamp = "2024-01-01T00:00:00"
        results.training_time = {}

        # Test 1: Confidence model has fewer parameters (error)
        results.parameter_counts = {"standard": 1000, "confidence": 900}
        validation = validate_experiment_results(results, sample_config)

        assert validation["is_valid"] == False
        assert any("should have more parameters" in error for error in validation["errors"])

        # Test 2: Confidence model has reasonable more parameters (pass)
        results.parameter_counts = {
            "standard": 1000,
            "confidence": 1200,
        }  # 20% increase
        validation = validate_experiment_results(results, sample_config)

        assert validation["is_valid"] == True
        assert "reasonable_param_increase" in validation["checks_passed"]

        # Test 3: Confidence model has too many parameters (warning)
        results.parameter_counts = {
            "standard": 1000,
            "confidence": 2000,
        }  # 100% increase
        validation = validate_experiment_results(results, sample_config)

        assert validation["is_valid"] == True  # Still valid, just warning
        assert any("Large parameter increase" in warning for warning in validation["warnings"])


class TestGenerateValidationReport:
    """Test generate_validation_report function."""

    @pytest.mark.unit
    def test_generate_validation_report_valid(self):
        """Test report generation for valid results."""
        validation_results = {
            "is_valid": True,
            "checks_passed": ["has_standard_model_results", "seed_consistency"],
            "errors": [],
            "warnings": ["Minor issue"],
        }

        report = generate_validation_report(validation_results, verbose=True)

        # Check report structure
        assert "EXPERIMENT VALIDATION REPORT" in report
        assert "Overall Status: PASS" in report
        assert "Checks Passed: 2" in report
        assert "Errors: 0" in report
        assert "Warnings: 1" in report
        assert "PASSED CHECKS:" in report  # Because verbose=True
        assert "has_standard_model_results" in report

    @pytest.mark.unit
    def test_generate_validation_report_invalid(self):
        """Test report generation for invalid results."""
        validation_results = {
            "is_valid": False,
            "checks_passed": ["has_timestamp"],
            "errors": ["Missing attribute: standard_model_results", "Seed mismatch"],
            "warnings": [],
        }

        report = generate_validation_report(validation_results, verbose=False)

        assert "Overall Status: FAIL" in report
        assert "Errors: 2" in report
        assert "ERRORS:" in report
        assert "Missing attribute" in report
        assert "Seed mismatch" in report
        # Should not include passed checks when verbose=False
        assert "PASSED CHECKS:" not in report

    @pytest.mark.unit
    def test_generate_validation_report_empty(self):
        """Test report generation with empty results."""
        validation_results = {
            "is_valid": True,
            "checks_passed": [],
            "errors": [],
            "warnings": [],
        }

        report = generate_validation_report(validation_results)

        assert "Overall Status: PASS" in report
        assert "Checks Passed: 0" in report
        assert "Errors: 0" in report
        assert "Warnings: 0" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
