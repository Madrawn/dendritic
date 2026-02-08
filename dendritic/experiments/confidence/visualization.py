"""
Visualization utilities for confidence-aware experiment results.

This module provides plotting functions for analyzing and visualizing
results from confidence-aware GPT experiments.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from pathlib import Path
from typing import Dict, List, Optional, Union

from .results import ConfidenceExperimentResults, load_results


def plot_loss_curves(
    results: Union[ConfidenceExperimentResults, str, Path, Dict],
    output_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    figsize: tuple = (14, 10),
) -> Figure:
    """
    Plot loss curves for both standard and confidence models.

    Args:
        results: ConfidenceExperimentResults object, path to JSON file, or results dict
        output_path: Optional path to save figure
        show: Whether to display the plot
        figsize: Figure size (width, height)

    Returns:
        matplotlib Figure object
    """
    # Load results if needed
    if isinstance(results, (str, Path)):
        results = load_results(Path(results))
    elif isinstance(results, dict):
        results = ConfidenceExperimentResults.from_dict(results)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    ax1, ax2, ax3, ax4 = axes.flat

    # Plot 1: Token loss comparison
    _plot_token_loss_comparison(results, ax1)

    # Plot 2: Confidence loss (confidence model only)
    _plot_confidence_loss(results, ax2)

    # Plot 3: Evaluation loss over time
    _plot_eval_loss_comparison(results, ax3)

    # Plot 4: Confidence predictions vs actual future losses
    _plot_calibration_scatter(results, ax4)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig


def plot_calibration_curve(
    results: Union[ConfidenceExperimentResults, str, Path, Dict],
    output_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    figsize: tuple = (10, 8),
) -> Figure:
    """
    Plot calibration curve showing predicted confidence vs actual future loss.

    Args:
        results: ConfidenceExperimentResults object, path to JSON file, or results dict
        output_path: Optional path to save figure
        show: Whether to display the plot
        figsize: Figure size (width, height)

    Returns:
        matplotlib Figure object
    """
    # Load results if needed
    if isinstance(results, (str, Path)):
        results = load_results(Path(results))
    elif isinstance(results, dict):
        results = ConfidenceExperimentResults.from_dict(results)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    ax1, ax2 = axes

    # Plot 1: Calibration scatter plot
    _plot_calibration_scatter(results, ax1)

    # Plot 2: Binned calibration curve
    _plot_binned_calibration(results, ax2)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig


def plot_training_time_comparison(
    results: Union[ConfidenceExperimentResults, str, Path, Dict],
    output_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    figsize: tuple = (8, 6),
) -> Figure:
    """
    Plot training time comparison between standard and confidence models.

    Args:
        results: ConfidenceExperimentResults object, path to JSON file, or results dict
        output_path: Optional path to save figure
        show: Whether to display the plot
        figsize: Figure size (width, height)

    Returns:
        matplotlib Figure object
    """
    # Load results if needed
    if isinstance(results, (str, Path)):
        results = load_results(Path(results))
    elif isinstance(results, dict):
        results = ConfidenceExperimentResults.from_dict(results)

    fig, ax = plt.subplots(figsize=figsize)

    # Extract training times
    model_types = ["standard", "confidence"]
    training_times = [results.training_time.get(model_type, 0.0) for model_type in model_types]

    # Create bar chart
    bars = ax.bar(model_types, training_times, color=["#1f77b4", "#ff7f0e"])

    # Add value labels on bars
    for bar, time_val in zip(bars, training_times):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.1,
            f"{time_val:.1f}s",
            ha="center",
            va="bottom",
        )

    # Calculate speedup/slowdown
    if training_times[0] > 0:
        relative_change = (training_times[1] - training_times[0]) / training_times[0] * 100
        change_text = (
            f"Slower by {abs(relative_change):.1f}%"
            if relative_change > 0
            else f"Faster by {abs(relative_change):.1f}%"
        )
        ax.text(
            0.5,
            0.95,
            change_text,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=12,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    ax.set_ylabel("Training Time (seconds)")
    ax.set_title("Training Time Comparison")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig


def generate_summary_statistics(
    results: Union[ConfidenceExperimentResults, str, Path, Dict],
) -> Dict[str, Dict[str, float]]:
    """
    Generate summary statistics for confidence experiment results.

    Args:
        results: ConfidenceExperimentResults object, path to JSON file, or results dict

    Returns:
        Dictionary containing summary statistics for both model types
    """
    # Load results if needed
    if isinstance(results, (str, Path)):
        results = load_results(Path(results))
    elif isinstance(results, dict):
        results = ConfidenceExperimentResults.from_dict(results)

    stats = {"standard": {}, "confidence": {}}

    # Helper function to extract metrics from results
    def extract_metrics(model_results_dict, model_type="standard"):
        metrics = {}
        all_final_eval_losses = []
        all_final_perplexities = []
        all_training_times = []

        for seed, result_list in model_results_dict.items():
            for result in result_list:
                # Handle both TrainingResult objects and dicts (from JSON)
                if isinstance(result, dict):
                    all_final_eval_losses.append(result.get("final_eval_loss", 0))
                    all_final_perplexities.append(result.get("final_perplexity", 0))
                    all_training_times.append(result.get("training_time", 0))
                else:
                    # Assume it's a TrainingResult object
                    all_final_eval_losses.append(result.final_eval_loss)
                    all_final_perplexities.append(result.final_perplexity)
                    all_training_times.append(result.training_time)

        if all_final_eval_losses:
            metrics["final_eval_loss_mean"] = float(np.mean(all_final_eval_losses))
            metrics["final_eval_loss_std"] = float(np.std(all_final_eval_losses))
            metrics["final_perplexity_mean"] = float(np.mean(all_final_perplexities))
            metrics["final_perplexity_std"] = float(np.std(all_final_perplexities))
            metrics["training_time_mean"] = float(np.mean(all_training_times))
            metrics["training_time_std"] = float(np.std(all_training_times))

        return metrics

    # Extract metrics for both model types
    stats["standard"] = extract_metrics(results.standard_model_results, "standard")
    stats["confidence"] = extract_metrics(results.confidence_model_results, "confidence")

    # Calculate relative improvements
    if stats["standard"] and stats["confidence"]:
        std_loss = stats["standard"].get("final_eval_loss_mean", 0)
        conf_loss = stats["confidence"].get("final_eval_loss_mean", 0)
        if std_loss > 0:
            relative_improvement = (std_loss - conf_loss) / std_loss * 100
            stats["comparison"] = {
                "relative_improvement_percent": float(relative_improvement),
                "absolute_improvement": float(std_loss - conf_loss),
            }

    return stats


# Internal helper functions
def _plot_token_loss_comparison(results: ConfidenceExperimentResults, ax: Axes):
    """Plot token loss comparison between standard and confidence models."""
    # Extract token loss histories
    std_token_losses = []
    conf_token_losses = []

    for seed, result_list in results.confidence_model_results.items():
        for result in result_list:
            conf_token_losses.extend(result.token_loss_history)

    # Standard models don't have token_loss_history, use loss_history instead
    for seed, result_list in results.standard_model_results.items():
        for result in result_list:
            # Extract training losses from loss_history
            if result.loss_history:
                train_losses = [h.get("train_loss", h.get("train_loss_lm", 0)) for h in result.loss_history]
                std_token_losses.extend(train_losses)

    # Plot histograms
    if std_token_losses and conf_token_losses:
        ax.hist(std_token_losses, bins=50, alpha=0.5, label="Standard", color="blue")
        ax.hist(conf_token_losses, bins=50, alpha=0.5, label="Confidence", color="orange")
        ax.set_xlabel("Token Loss")
        ax.set_ylabel("Frequency")
        ax.set_title("Token Loss Distribution")
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(
            0.5,
            0.5,
            "No token loss data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Token Loss Distribution")


def _plot_confidence_loss(results: ConfidenceExperimentResults, ax: Axes):
    """Plot confidence loss history."""
    confidence_losses = []

    for seed, result_list in results.confidence_model_results.items():
        for result in result_list:
            confidence_losses.extend(result.confidence_loss_history)

    if confidence_losses:
        steps = range(len(confidence_losses))
        ax.plot(steps, confidence_losses, color="red", alpha=0.7)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Confidence Loss")
        ax.set_title("Confidence Loss Over Time")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 15)  # Cap y-axis at 15 for loss plots

        # Add rolling average
        if len(confidence_losses) > 100:
            window = min(100, len(confidence_losses) // 10)
            rolling_avg = np.convolve(confidence_losses, np.ones(window) / window, mode="valid")
            ax.plot(
                steps[window - 1 :],
                rolling_avg,
                color="darkred",
                linewidth=2,
                label=f"{window}-step avg",
            )
            ax.legend()
    else:
        ax.text(
            0.5,
            0.5,
            "No confidence loss data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Confidence Loss Over Time")


def _plot_eval_loss_comparison(results: ConfidenceExperimentResults, ax: Axes):
    """Plot evaluation loss comparison over training."""
    # Extract evaluation losses from loss_history
    std_eval_losses = []
    conf_eval_losses = []
    std_steps = []
    conf_steps = []

    for seed, result_list in results.standard_model_results.items():
        for result in result_list:
            if result.loss_history:
                steps = [h.get("step", i) for i, h in enumerate(result.loss_history)]
                eval_losses = [h.get("perplexity", 0) for h in result.loss_history]
                std_steps.extend(steps)
                std_eval_losses.extend(eval_losses)

    for seed, result_list in results.confidence_model_results.items():
        for result in result_list:
            if result.loss_history:
                steps = [h.get("step", i) for i, h in enumerate(result.loss_history)]
                eval_losses = [h.get("perplexity", 0) for h in result.loss_history]
                conf_steps.extend(steps)
                conf_eval_losses.extend(eval_losses)

    if std_eval_losses and conf_eval_losses:
        # Sort by steps for cleaner plotting
        std_data = sorted(zip(std_steps, std_eval_losses))
        conf_data = sorted(zip(conf_steps, conf_eval_losses))

        if std_data:
            std_steps_sorted, std_eval_sorted = zip(*std_data)
            ax.plot(std_steps_sorted, std_eval_sorted, "b-", alpha=0.7, label="Standard")

        if conf_data:
            conf_steps_sorted, conf_eval_sorted = zip(*conf_data)
            ax.plot(conf_steps_sorted, conf_eval_sorted, "r-", alpha=0.7, label="Confidence")

        ax.set_xlabel("Training Step")
        ax.set_ylabel("Perplexity")
        ax.set_title("Evaluation Loss Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 150)  # Cap y-axis at 150 for perplexity plots
    else:
        ax.text(
            0.5,
            0.5,
            "No evaluation loss data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Evaluation Loss Comparison")


def _plot_calibration_scatter(results: ConfidenceExperimentResults, ax: Axes):
    """Plot scatter plot of loss predictions vs actual future losses."""
    loss_predictions = []
    actual_future_losses = []

    for seed, result_list in results.confidence_model_results.items():
        for result in result_list:
            if result.loss_predictions and result.actual_future_losses:
                # Take matching lengths
                min_len = min(len(result.loss_predictions), len(result.actual_future_losses))
                loss_predictions.extend(result.loss_predictions[:min_len])
                actual_future_losses.extend(result.actual_future_losses[:min_len])

    if loss_predictions and actual_future_losses:
        ax.scatter(loss_predictions, actual_future_losses, alpha=0.5, s=10)
        ax.set_xlabel("Predicted Loss")
        ax.set_ylabel("Actual Future Loss")
        ax.set_title("Loss Prediction Calibration Scatter")

        # Add ideal calibration line (loss prediction = actual loss)
        x_ideal = np.linspace(min(loss_predictions), max(loss_predictions), 100)
        y_ideal = x_ideal  # Ideal: predicted = actual
        ax.plot(x_ideal, y_ideal, "r--", alpha=0.7, label="Ideal: loss_pred = actual_loss")

        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 15)  # Cap y-axis at 15 for loss plots
    else:
        ax.text(
            0.5,
            0.5,
            "No calibration data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Loss Prediction Calibration Scatter")


def _plot_binned_calibration(results: ConfidenceExperimentResults, ax: Axes):
    """Plot binned calibration curve."""
    loss_predictions = []
    actual_future_losses = []

    for seed, result_list in results.confidence_model_results.items():
        for result in result_list:
            if result.loss_predictions and result.actual_future_losses:
                min_len = min(len(result.loss_predictions), len(result.actual_future_losses))
                loss_predictions.extend(result.loss_predictions[:min_len])
                actual_future_losses.extend(result.actual_future_losses[:min_len])

    if loss_predictions and actual_future_losses:
        # Bin the data
        n_bins = 10
        bins = np.linspace(min(loss_predictions), max(loss_predictions), n_bins + 1)

        bin_centers = []
        bin_avg_losses = []
        bin_std_losses = []

        for i in range(n_bins):
            mask = (np.array(loss_predictions) >= bins[i]) & (np.array(loss_predictions) < bins[i + 1])
            if np.any(mask):
                bin_data = np.array(actual_future_losses)[mask]
                bin_centers.append((bins[i] + bins[i + 1]) / 2)
                bin_avg_losses.append(np.mean(bin_data))
                bin_std_losses.append(np.std(bin_data))

        if bin_centers:
            ax.errorbar(
                bin_centers,
                bin_avg_losses,
                yerr=bin_std_losses,
                fmt="o-",
                capsize=5,
                label="Binned Calibration",
            )
            ax.set_xlabel("Predicted Loss (binned)")
            ax.set_ylabel("Average Actual Future Loss")
            ax.set_title("Binned Loss Prediction Curve")
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 15)  # Cap y-axis at 15 for loss plots

            # Add ideal line
            x_ideal = np.linspace(min(bin_centers), max(bin_centers), 100)
            y_ideal = x_ideal  # Ideal: predicted = actual
            ax.plot(x_ideal, y_ideal, "r--", alpha=0.7, label="Ideal")
            ax.legend()
    else:
        ax.text(
            0.5,
            0.5,
            "No calibration data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Binned Loss Prediction Curve")
