"""
Visualization utilities for self-conditioned experiment results.

This module provides plotting functions for analyzing and visualizing
results from self-conditioned GPT experiments.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from pathlib import Path
from typing import Dict, List, Optional, Union

from .results import SelfConditionedExperimentResults, load_results


def plot_loss_curves(
    results: Union[SelfConditionedExperimentResults, str, Path, Dict],
    output_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    figsize: tuple = (14, 10),
) -> Figure:
    """
    Plot loss curves for both standard and self-conditioned models.

    Args:
        results: SelfConditionedExperimentResults object, path to JSON file, or results dict
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
        results = SelfConditionedExperimentResults.from_dict(results)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    ax1, ax2, ax3, ax4 = axes.flat

    # Plot 1: Token loss distribution comparison
    _plot_token_loss_comparison(results, ax1)

    # Plot 2: Self-conditioned signal analysis (if available)
    _plot_self_conditioned_signal(results, ax2)

    # Plot 3: Evaluation loss over time
    _plot_eval_loss_comparison(results, ax3)

    # Plot 4: Perplexity comparison over time
    _plot_perplexity_comparison(results, ax4)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig


def plot_training_time_comparison(
    results: Union[SelfConditionedExperimentResults, str, Path, Dict],
    output_path: Optional[Union[str, Path]] = None,
    show: bool = True,
    figsize: tuple = (8, 6),
) -> Figure:
    """
    Plot training time comparison between standard and self-conditioned models.

    Args:
        results: SelfConditionedExperimentResults object, path to JSON file, or results dict
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
        results = SelfConditionedExperimentResults.from_dict(results)

    fig, ax = plt.subplots(figsize=figsize)

    # Extract training times
    model_types = ["standard", "self_conditioned"]
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
    results: Union[SelfConditionedExperimentResults, str, Path, Dict],
) -> Dict[str, Dict[str, float]]:
    """
    Generate summary statistics for self-conditioned experiment results.

    Args:
        results: SelfConditionedExperimentResults object, path to JSON file, or results dict

    Returns:
        Dictionary containing summary statistics for both model types
    """
    # Load results if needed
    if isinstance(results, (str, Path)):
        results = load_results(Path(results))
    elif isinstance(results, dict):
        results = SelfConditionedExperimentResults.from_dict(results)

    stats = {"standard": {}, "self_conditioned": {}}

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
    stats["self_conditioned"] = extract_metrics(results.self_conditioned_model_results, "self_conditioned")

    # Calculate relative improvements
    if stats["standard"] and stats["self_conditioned"]:
        std_loss = stats["standard"].get("final_eval_loss_mean", 0)
        sc_loss = stats["self_conditioned"].get("final_eval_loss_mean", 0)
        if std_loss > 0:
            relative_improvement = (std_loss - sc_loss) / std_loss * 100
            stats["comparison"] = {
                "relative_improvement_percent": float(relative_improvement),
                "absolute_improvement": float(std_loss - sc_loss),
            }

    return stats


# Internal helper functions
def _plot_token_loss_comparison(results: SelfConditionedExperimentResults, ax: Axes):
    """Plot token loss comparison between standard and self-conditioned models."""
    # Extract token loss histories
    std_token_losses = []
    sc_token_losses = []

    for seed, result_list in results.standard_model_results.items():
        for result in result_list:
            # Extract training losses from loss_history
            if result.loss_history:
                train_losses = [h.get("train_loss", h.get("train_loss_lm", 0)) for h in result.loss_history]
                std_token_losses.extend(train_losses)

    for seed, result_list in results.self_conditioned_model_results.items():
        for result in result_list:
            if result.loss_history:
                train_losses = [h.get("train_loss", h.get("train_loss_lm", 0)) for h in result.loss_history]
                sc_token_losses.extend(train_losses)

    # Plot histograms
    if std_token_losses and sc_token_losses:
        ax.hist(std_token_losses, bins=50, alpha=0.5, label="Standard", color="blue")
        ax.hist(sc_token_losses, bins=50, alpha=0.5, label="Self-Conditioned", color="green")
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


def _plot_self_conditioned_signal(results: SelfConditionedExperimentResults, ax: Axes):
    """Plot self-conditioned signal distribution if available."""
    # Note: SelfConditionedGPT doesn't store the conditioning signal in results by default.
    # This is a placeholder for future instrumentation if needed.
    ax.text(
        0.5,
        0.5,
        "Self-conditioned signal analysis\nrequires additional instrumentation",
        ha="center",
        va="center",
        transform=ax.transAxes,
        fontsize=10,
    )
    ax.set_title("Self-Conditioned Signal Analysis")


def _plot_eval_loss_comparison(results: SelfConditionedExperimentResults, ax: Axes):
    """Plot evaluation loss comparison over training."""
    # Extract evaluation losses from loss_history
    std_eval_losses = []
    sc_eval_losses = []
    std_steps = []
    sc_steps = []

    for seed, result_list in results.standard_model_results.items():
        for result in result_list:
            if result.loss_history:
                steps = [h.get("step", i) for i, h in enumerate(result.loss_history)]
                eval_losses = [h.get("eval_loss", h.get("perplexity", 0)) for h in result.loss_history]
                std_steps.extend(steps)
                std_eval_losses.extend(eval_losses)

    for seed, result_list in results.self_conditioned_model_results.items():
        for result in result_list:
            if result.loss_history:
                steps = [h.get("step", i) for i, h in enumerate(result.loss_history)]
                eval_losses = [h.get("eval_loss", h.get("perplexity", 0)) for h in result.loss_history]
                sc_steps.extend(steps)
                sc_eval_losses.extend(eval_losses)

    if std_eval_losses and sc_eval_losses:
        # Sort by steps for cleaner plotting
        std_data = sorted(zip(std_steps, std_eval_losses))
        sc_data = sorted(zip(sc_steps, sc_eval_losses))

        if std_data:
            std_steps_sorted, std_eval_sorted = zip(*std_data)
            ax.plot(std_steps_sorted, std_eval_sorted, "b-", alpha=0.7, label="Standard")

        if sc_data:
            sc_steps_sorted, sc_eval_sorted = zip(*sc_data)
            ax.plot(sc_steps_sorted, sc_eval_sorted, "g-", alpha=0.7, label="Self-Conditioned")

        ax.set_xlabel("Training Step")
        ax.set_ylabel("Evaluation Loss")
        ax.set_title("Evaluation Loss Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, None)  # Auto-scale but start at 0
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


def _plot_perplexity_comparison(results: SelfConditionedExperimentResults, ax: Axes):
    """Plot perplexity comparison over training."""
    # Extract perplexities from loss_history
    std_perplexities = []
    sc_perplexities = []
    std_steps = []
    sc_steps = []

    for seed, result_list in results.standard_model_results.items():
        for result in result_list:
            if result.loss_history:
                steps = [h.get("step", i) for i, h in enumerate(result.loss_history)]
                perplexities = [h.get("perplexity", 0) for h in result.loss_history if h.get("perplexity", 0) > 0]
                steps_filtered = [s for s, p in zip(steps, perplexities) if p > 0]
                std_steps.extend(steps_filtered)
                std_perplexities.extend([p for p in perplexities if p > 0])

    for seed, result_list in results.self_conditioned_model_results.items():
        for result in result_list:
            if result.loss_history:
                steps = [h.get("step", i) for i, h in enumerate(result.loss_history)]
                perplexities = [h.get("perplexity", 0) for h in result.loss_history if h.get("perplexity", 0) > 0]
                steps_filtered = [s for s, p in zip(steps, perplexities) if p > 0]
                sc_steps.extend(steps_filtered)
                sc_perplexities.extend([p for p in perplexities if p > 0])

    if std_perplexities and sc_perplexities:
        # Sort by steps for cleaner plotting
        std_data = sorted(zip(std_steps, std_perplexities))
        sc_data = sorted(zip(sc_steps, sc_perplexities))

        if std_data:
            std_steps_sorted, std_perp_sorted = zip(*std_data)
            ax.plot(std_steps_sorted, std_perp_sorted, "b-", alpha=0.7, label="Standard")

        if sc_data:
            sc_steps_sorted, sc_perp_sorted = zip(*sc_data)
            ax.plot(sc_steps_sorted, sc_perp_sorted, "g-", alpha=0.7, label="Self-Conditioned")

        ax.set_xlabel("Training Step")
        ax.set_ylabel("Perplexity")
        ax.set_title("Perplexity Comparison")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 150)  # Cap y-axis at 150 for perplexity plots
    else:
        ax.text(
            0.5,
            0.5,
            "No perplexity data available",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Perplexity Comparison")
