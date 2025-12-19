# dendritic/experiments/visualization.py
"""Visualization utilities for experiment results."""
import json

import numpy as np
import matplotlib.pyplot as plt

from ExperimentResults import ExperimentResults, TrainingResult


# Model configuration for consistent styling and labeling
def get_model_config(model_key: str) -> dict:
    """Dynamically generate model configuration based on model key.

    Args:
        model_key: The key identifying the model variant

    Returns:
        Dictionary with color, label, and bar_color properties
    """
    # Use matplotlib colormap for consistent colors
    colormap = plt.get_cmap("tab10")

    # Special case for baseline model
    if model_key == "baseline":
        return {"color": "#1f77b4", "label": "Baseline", "bar_color": "#1f77b4"}

    # For other models, generate colors from colormap
    # Hash model key to consistent index in colormap
    color_idx = hash(model_key) % 10
    color = colormap(color_idx)
    return {
        "color": color,
        "label": model_key.replace("_", " ").title(),
        "bar_color": color,
    }


def plot_training_curves(
    results: str | ExperimentResults | dict,
    output_path: str | None = None,
    show: bool = True,
):
    """
    Plot training curves from experiment results.

    Args:
        results: Path to JSON results file, ExperimentResults object, or results dict
        output_path: Optional path to save figure
        show: Whether to display the plot
    """
    # Handle different input types
    if isinstance(results, str):
        # Load from JSON file
        with open(results) as f:
            results_data = json.load(f)
    elif isinstance(results, ExperimentResults):
        # Use ExperimentResults object directly
        results_data = {
            "model_results": results.model_results,
            "statistical_analysis": results.statistical_analysis,
        }
    else:
        # Assume it's already a dict
        results_data = results

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Determine experiment type and plot
    try:
        # Pretraining experiment (with or without stack) - new format
        plot_pretraining_curves(results_data, axes)
    except Exception as e:
        print("[DEBUG] WARNING: Unrecognized experiment type! No curves plotted.")
        print(f"[DEBUG] Results contained keys: {list(results_data.keys())}")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()

    return fig


def plot_pretraining_curves(results: dict, axes):
    """Plot pretraining experiment curves."""
    ax1, ax2 = axes

    # Normalize results structure
    model_runs = {}
    if "model_runs" in results:
        model_runs = results["model_runs"]
    else:
        # Backward compatibility with old format
        model_runs["baseline"] = results.get("baseline_runs", [])
        model_runs["dendritic"] = results.get("dendritic_runs", [])
        model_runs["stack"] = results.get("stack_runs", [])

    # Plot individual runs with transparency
    for model_key in model_runs:
        if not model_runs[model_key]:
            continue
        config = get_model_config(model_key)
        color = config["color"]
        for run in model_runs[model_key]:
            # Handle both TrainingResult objects and dicts
            if isinstance(run, TrainingResult):
                steps = [h["step"] for h in run.loss_history]
                ppls = [h["perplexity"] for h in run.loss_history]
            else:
                steps = [h["step"] for h in run["loss_history"]]
                ppls = [h["perplexity"] for h in run["loss_history"]]
            ax1.plot(steps, ppls, color=color, alpha=0.3, linewidth=1)

    # Plot mean curves
    for model_key, runs in model_runs.items():
        if not runs:
            continue
        config = get_model_config(model_key)
        color = config["color"]
        label = config["label"]

        mean_curve = compute_mean_curve(runs)
        ax1.plot(
            mean_curve["steps"],
            mean_curve["ppl"],
            color=color,
            linewidth=2.5,
            label=label,
        )

    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Perplexity")
    ax1.set_title("Training Curves (Perplexity)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Bar chart of final results
    analysis = results["statistical_analysis"]
    methods = []
    means = []
    stds = []
    colors = []

    # Use natural ordering of models from results
    model_order = sorted(analysis.keys())
    for model_key in model_order:
        if model_key.startswith("comparison_"):
            continue
        if model_key in analysis and model_key in model_runs and model_runs[model_key]:
            config = get_model_config(model_key)
            methods.append(config["label"])
            means.append(analysis[model_key]["final_ppl_mean"])
            stds.append(analysis[model_key]["final_ppl_std"])
            colors.append(config["bar_color"])

    bars = ax2.bar(methods, means, yerr=stds, capsize=5, color=colors, alpha=0.7)

    ax2.set_ylabel("Final Perplexity")
    ax2.set_title("Final Performance Comparison")

    # Significance annotations
    y_pos = 0.95
    y_step = 0.05

    # Comparison configuration
    comparisons = [
        ("baseline", "dendritic", "Baseline vs Dendritic", "#1f77b4"),
        ("baseline", "stack", "Baseline vs Stack", "#2ca02c"),
        ("dendritic", "stack", "Dendritic vs Stack", "#ff7f0e"),
    ]

    for model1, model2, label, color in comparisons:
        comp_key = f"comparison_{model1}_{model2}"
        if comp_key not in analysis:
            continue

        comp = analysis[comp_key]
        if comp["significant_001"]:
            sig_text = f"{label}: *** p < 0.01"
        elif comp["significant_005"]:
            sig_text = f"{label}: * p < 0.05"
        else:
            continue

        ax2.annotate(
            sig_text,
            xy=(0.5, y_pos),
            xycoords="axes fraction",
            ha="center",
            fontsize=10,
            color=color,
        )
        y_pos -= y_step


def compute_mean_curve(
    runs: list[dict | TrainingResult],
) -> dict[str, list[float]]:
    """Compute mean curve across runs."""
    # Align steps
    all_steps = set()
    for run in runs:
        if isinstance(run, TrainingResult):
            all_steps.update(h["step"] for h in run.loss_history)
        else:
            all_steps.update(h["step"] for h in run["loss_history"])
    steps = sorted(all_steps)

    # Interpolate values at each step
    ppl_at_step = {s: [] for s in steps}
    for run in runs:
        if isinstance(run, TrainingResult):
            run_steps = [h["step"] for h in run.loss_history]
            run_ppl = [h["perplexity"] for h in run.loss_history]
        else:
            run_steps = [h["step"] for h in run["loss_history"]]
            run_ppl = [h["perplexity"] for h in run["loss_history"]]

        for s in steps:
            if s in run_steps:
                idx = run_steps.index(s)
                ppl_at_step[s].append(run_ppl[idx])

    # Convert to native float and handle empty cases
    mean_ppl = [
        float(np.mean(ppl_at_step[s])) if ppl_at_step[s] else 0.0 for s in steps
    ]

    return {"steps": steps, "ppl": mean_ppl}


if __name__ == "__main__":
    results_json_path = (
        r"results\pretraining_comparison\pretraining_experiment_20251218_110920.json"
    )
    plot_training_curves(
        results_json_path,
        show=True,
        output_path=results_json_path.replace(".json", ".png"),
    )
