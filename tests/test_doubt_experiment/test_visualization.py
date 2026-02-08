# ruff: noqa: PLR6301, PLR2004

"""
Unit tests for doubt experiment visualization module.
"""

import pytest
import json
import matplotlib
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from pathlib import Path
from unittest.mock import MagicMock
import tempfile

# Use non-interactive backend for testing
matplotlib.use("Agg")

from dendritic.experiments.doubt.visualization import (
    plot_loss_curves,
    plot_calibration_curve,
    plot_training_time_comparison,
    generate_summary_statistics,
    _plot_token_loss_comparison,
    _plot_doubt_loss,
    _plot_eval_loss_comparison,
    _plot_calibration_scatter,
    _plot_binned_calibration,
)
from dendritic.experiments.doubt.results import (
    DoubtExperimentResults,
    DoubtTrainingResult,
)
from dendritic.experiments.doubt.config import DoubtExperimentConfig
from dendritic.experiments.utils.TrainingResult import TrainingResult


# Module-level fixtures for use across test classes
@pytest.fixture
def sample_config():
    """Create a sample DoubtExperimentConfig."""
    return DoubtExperimentConfig(
        vocab_size=1000,
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        max_seq_len=32,
        batch_size=4,
        training_steps=10,
        seeds=[42, 43],
    )


@pytest.fixture
def sample_results(sample_config):
    """Create sample DoubtExperimentResults for testing."""
    # Create standard model results
    std_result = TrainingResult(
        model_type="standard",
        seed=42,
        final_train_loss=1.5,
        final_eval_loss=2.0,
        final_perplexity=7.5,
        best_eval_loss=1.8,
        best_perplexity=6.0,
        loss_history=[
            {"step": 0, "train_loss": 3.0, "eval_loss": 3.5},
            {"step": 5, "train_loss": 2.0, "eval_loss": 2.5},
            {"step": 10, "train_loss": 1.5, "eval_loss": 2.0},
        ],
        training_time=120.5,
        config={"vocab_size": 1000},
    )

    # Create doubt model results
    doubt_result = DoubtTrainingResult(
        model_type="doubt",
        seed=42,
        final_train_loss=1.6,
        final_eval_loss=2.1,
        final_perplexity=7.8,
        best_eval_loss=1.9,
        best_perplexity=6.2,
        loss_history=[
            {"step": 0, "train_loss": 3.1, "eval_loss": 3.6},
            {"step": 5, "train_loss": 2.1, "eval_loss": 2.6},
            {"step": 10, "train_loss": 1.6, "eval_loss": 2.1},
        ],
        training_time=130.5,
        config={"vocab_size": 1000},
        doubt_loss_history=[0.5, 0.4, 0.3, 0.25, 0.2],
        token_loss_history=[2.0, 1.8, 1.6, 1.5, 1.4],
        loss_predictions=[0.8, 0.7, 0.6, 0.5, 0.4],
        actual_future_losses=[1.2, 1.1, 1.0, 0.9, 0.8],
    )

    # Create results with multiple seeds
    results = DoubtExperimentResults(
        standard_model_results={
            "42": [std_result],
            "43": [
                TrainingResult(
                    model_type="standard",
                    seed=43,
                    final_train_loss=1.4,
                    final_eval_loss=1.9,
                    final_perplexity=7.0,
                    best_eval_loss=1.7,
                    best_perplexity=5.8,
                    loss_history=[
                        {"step": 0, "train_loss": 2.9, "eval_loss": 3.4},
                        {"step": 5, "train_loss": 1.9, "eval_loss": 2.4},
                        {"step": 10, "train_loss": 1.4, "eval_loss": 1.9},
                    ],
                    training_time=118.5,
                    config={"vocab_size": 1000},
                )
            ],
        },
        doubt_model_results={
            "42": [doubt_result],
            "43": [
                DoubtTrainingResult(
                    model_type="doubt",
                    seed=43,
                    final_train_loss=1.5,
                    final_eval_loss=2.0,
                    final_perplexity=7.2,
                    best_eval_loss=1.8,
                    best_perplexity=6.0,
                    loss_history=[
                        {"step": 0, "train_loss": 3.0, "eval_loss": 3.5},
                        {"step": 5, "train_loss": 2.0, "eval_loss": 2.5},
                        {"step": 10, "train_loss": 1.5, "eval_loss": 2.0},
                    ],
                    training_time=128.5,
                    config={"vocab_size": 1000},
                    doubt_loss_history=[0.6, 0.5, 0.4, 0.35, 0.3],
                    token_loss_history=[2.1, 1.9, 1.7, 1.6, 1.5],
                    loss_predictions=[0.9, 0.8, 0.7, 0.6, 0.5],
                    actual_future_losses=[1.3, 1.2, 1.1, 1.0, 0.9],
                )
            ],
        },
        config=sample_config,
        timestamp="2024-01-01_12:00:00",
        training_time={"standard": 239.0, "doubt": 259.0},
        parameter_counts={"standard": 1000000, "doubt": 1200000},
    )

    return results


@pytest.mark.unit
def test_plot_loss_curves_with_results_object(sample_results, tmp_path):
    """Test plot_loss_curves with DoubtExperimentResults object."""
    # Call function with results object
    fig = plot_loss_curves(
        results=sample_results,
        output_path=None,
        show=False,  # Don't show during tests
        figsize=(10, 8),
    )

    # Verify figure was created
    assert fig is not None
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 4  # Should have 4 subplots

    # Test saving to file
    output_path = tmp_path / "loss_curves.png"
    fig = plot_loss_curves(results=sample_results, output_path=output_path, show=False)

    assert output_path.exists()


@pytest.mark.unit
def test_plot_loss_curves_with_file_path(sample_results, tmp_path):
    """Test plot_loss_curves with file path string."""
    # Save results to JSON file
    results_file = tmp_path / "results.json"
    with open(results_file, "w") as f:
        json.dump(sample_results.to_dict(), f)

    # Call function with file path
    fig = plot_loss_curves(results=str(results_file), output_path=None, show=False)

    assert fig is not None
    assert isinstance(fig, Figure)


@pytest.mark.unit
def test_plot_loss_curves_with_dict(sample_results):
    """Test plot_loss_curves with results dictionary."""
    # Convert results to dict
    results_dict = sample_results.to_dict()

    # Call function with dict
    fig = plot_loss_curves(results=results_dict, output_path=None, show=False)

    assert fig is not None
    assert isinstance(fig, Figure)


@pytest.mark.unit
def test_plot_calibration_curve(sample_results, tmp_path):
    """Test plot_calibration_curve function."""
    # Test with results object
    fig = plot_calibration_curve(results=sample_results, output_path=None, show=False)

    assert fig is not None
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2  # Should have 2 subplots

    # Test saving to file
    output_path = tmp_path / "calibration.png"
    fig = plot_calibration_curve(results=sample_results, output_path=output_path, show=False)

    assert output_path.exists()


@pytest.mark.unit
def test_plot_training_time_comparison(sample_results, tmp_path):
    """Test plot_training_time_comparison function."""
    # Test with results object
    fig = plot_training_time_comparison(results=sample_results, output_path=None, show=False)

    assert fig is not None
    assert isinstance(fig, Figure)

    # Verify axes has correct data
    ax = fig.axes[0]
    bars = ax.patches
    assert len(bars) == 2  # Standard and doubt bars

    # Test saving to file
    output_path = tmp_path / "training_time.png"
    fig = plot_training_time_comparison(results=sample_results, output_path=output_path, show=False)

    assert output_path.exists()


@pytest.mark.unit
def test_generate_summary_statistics(sample_results):
    """Test generate_summary_statistics function."""
    # Generate statistics
    stats = generate_summary_statistics(sample_results)

    # Verify structure
    assert "standard" in stats
    assert "doubt" in stats
    assert "comparison" in stats

    # Verify standard model stats
    std_stats = stats["standard"]
    assert "final_eval_loss_mean" in std_stats
    assert "final_eval_loss_std" in std_stats
    assert "final_perplexity_mean" in std_stats
    assert "final_perplexity_std" in std_stats
    assert "training_time_mean" in std_stats
    assert "training_time_std" in std_stats

    # Verify doubt model stats
    doubt_stats = stats["doubt"]
    assert "final_eval_loss_mean" in doubt_stats
    assert "final_eval_loss_std" in doubt_stats

    # Verify comparison stats
    comp_stats = stats["comparison"]
    assert "relative_improvement_percent" in comp_stats
    assert "absolute_improvement" in comp_stats

    # Test with file path
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sample_results.to_dict(), f)
        temp_path = f.name

    try:
        stats_from_file = generate_summary_statistics(temp_path)
        assert "standard" in stats_from_file
        assert "doubt" in stats_from_file
    finally:
        Path(temp_path).unlink()


@pytest.mark.unit
def test_generate_summary_statistics_empty_results(sample_config):
    """Test generate_summary_statistics with empty results."""
    # Create empty results
    empty_results = DoubtExperimentResults(
        standard_model_results={},
        doubt_model_results={},
        config=sample_config,
        timestamp="2024-01-01_12:00:00",
        training_time={},
    )

    stats = generate_summary_statistics(empty_results)

    # Should still have the structure but empty stats
    assert "standard" in stats
    assert "doubt" in stats
    assert stats["standard"] == {}
    assert stats["doubt"] == {}
    assert "comparison" not in stats  # No comparison with empty data


class TestInternalPlottingFunctions:
    """Tests for internal helper plotting functions."""

    @pytest.fixture
    def mock_axes(self):
        """Create a mock matplotlib Axes object."""
        ax = MagicMock(spec=Axes)
        ax.text = MagicMock()
        ax.hist = MagicMock()
        ax.plot = MagicMock()
        ax.scatter = MagicMock()
        ax.errorbar = MagicMock()
        ax.set_xlabel = MagicMock()
        ax.set_ylabel = MagicMock()
        ax.set_title = MagicMock()
        ax.legend = MagicMock()
        ax.grid = MagicMock()
        ax.transAxes = MagicMock()  # Add transAxes attribute
        return ax

    @pytest.mark.unit
    def test_plot_token_loss_comparison(self, sample_results, mock_axes):
        """Test _plot_token_loss_comparison function."""
        _plot_token_loss_comparison(sample_results, mock_axes)

        # Should call hist for both standard and doubt
        assert mock_axes.hist.call_count >= 1
        mock_axes.set_title.assert_called_with("Token Loss Distribution")

    @pytest.mark.unit
    def test_plot_token_loss_comparison_empty(self, sample_config, mock_axes):
        """Test _plot_token_loss_comparison with empty data."""
        # Create results with no token loss data
        empty_results = DoubtExperimentResults(
            standard_model_results={},
            doubt_model_results={},
            config=sample_config,
            timestamp="2024-01-01_12:00:00",
            training_time={},
        )

        _plot_token_loss_comparison(empty_results, mock_axes)

        # Should display "No token loss data available" text
        mock_axes.text.assert_called()
        mock_axes.set_title.assert_called_with("Token Loss Distribution")

    @pytest.mark.unit
    def test_plot_doubt_loss(self, sample_results, mock_axes):
        """Test _plot_doubt_loss function."""
        _plot_doubt_loss(sample_results, mock_axes)

        # Should plot doubt loss history
        mock_axes.plot.assert_called()
        mock_axes.set_title.assert_called_with("Doubt Loss Over Time")

    @pytest.mark.unit
    def test_plot_doubt_loss_empty(self, sample_config, mock_axes):
        """Test _plot_doubt_loss with empty data."""
        # Create results with no doubt loss data
        empty_results = DoubtExperimentResults(
            standard_model_results={},
            doubt_model_results={},
            config=sample_config,
            timestamp="2024-01-01_12:00:00",
            training_time={},
        )

        _plot_doubt_loss(empty_results, mock_axes)

        # Should display "No doubt loss data available" text
        mock_axes.text.assert_called()
        mock_axes.set_title.assert_called_with("Doubt Loss Over Time")

    @pytest.mark.unit
    def test_plot_eval_loss_comparison(self, sample_results, mock_axes):
        """Test _plot_eval_loss_comparison function."""
        _plot_eval_loss_comparison(sample_results, mock_axes)

        # Should plot evaluation loss for both models
        mock_axes.plot.assert_called()
        mock_axes.set_title.assert_called_with("Evaluation Loss Comparison")

    @pytest.mark.unit
    def test_plot_eval_loss_comparison_empty(self, sample_config, mock_axes):
        """Test _plot_eval_loss_comparison with empty data."""
        # Create results with no evaluation loss data
        empty_results = DoubtExperimentResults(
            standard_model_results={},
            doubt_model_results={},
            config=sample_config,
            timestamp="2024-01-01_12:00:00",
            training_time={},
        )

        _plot_eval_loss_comparison(empty_results, mock_axes)

        # Should display "No evaluation loss data available" text
        mock_axes.text.assert_called()
        mock_axes.set_title.assert_called_with("Evaluation Loss Comparison")

    @pytest.mark.unit
    def test_plot_calibration_scatter(self, sample_results, mock_axes):
        """Test _plot_calibration_scatter function."""
        _plot_calibration_scatter(sample_results, mock_axes)

        # Should create scatter plot
        mock_axes.scatter.assert_called()
        mock_axes.plot.assert_called()  # For ideal line
        mock_axes.set_title.assert_called_with("Loss Prediction Calibration Scatter")

    @pytest.mark.unit
    def test_plot_calibration_scatter_empty(self, sample_config, mock_axes):
        """Test _plot_calibration_scatter with empty data."""
        # Create results with no calibration data
        empty_results = DoubtExperimentResults(
            standard_model_results={},
            doubt_model_results={},
            config=sample_config,
            timestamp="2024-01-01_12:00:00",
            training_time={},
        )

        _plot_calibration_scatter(empty_results, mock_axes)

        # Should display "No calibration data available" text
        mock_axes.text.assert_called()
        mock_axes.set_title.assert_called_with("Loss Prediction Calibration Scatter")

    @pytest.mark.unit
    def test_plot_binned_calibration(self, sample_results, mock_axes):
        """Test _plot_binned_calibration function."""
        _plot_binned_calibration(sample_results, mock_axes)

        # Should create errorbar plot
        mock_axes.errorbar.assert_called()
        mock_axes.plot.assert_called()  # For ideal line
        mock_axes.set_title.assert_called_with("Binned Loss Prediction Curve")

    @pytest.mark.unit
    def test_plot_binned_calibration_empty(self, sample_config, mock_axes):
        """Test _plot_binned_calibration with empty data."""
        # Create results with no calibration data
        empty_results = DoubtExperimentResults(
            standard_model_results={},
            doubt_model_results={},
            config=sample_config,
            timestamp="2024-01-01_12:00:00",
            training_time={},
        )

        _plot_binned_calibration(empty_results, mock_axes)

        # Should display "No calibration data available" text
        mock_axes.text.assert_called()
        mock_axes.set_title.assert_called_with("Binned Loss Prediction Curve")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
