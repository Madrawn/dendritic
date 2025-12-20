# dendritic/experiments/visualization.py
"""Visualization utilities for experiment results."""
import json

import numpy as np
import matplotlib.pyplot as plt

from dendritic.experiments.utils.ExperimentResults import (
    ExperimentResults,
    TrainingResult,
)


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


def _normalize_results(results: dict) -> tuple[dict, dict]:
    """Normalize results dict to have model_runs and statistical_analysis.

    Handles both old flat format and new variant format.
    """
    if "variants" in results:
        model_runs = {}
        analysis = {}
        for variant in results["variants"]:
            variant_name = variant["name"]
            for model_key, runs in variant["model_runs"].items():
                new_key = f"{variant_name}-{model_key}"
                model_runs[new_key] = runs
            for stat_key, stat_val in variant["statistical_analysis"].items():
                if not stat_key.startswith("comparison_"):
                    new_key = f"{variant_name}-{stat_key}"
                    analysis[new_key] = stat_val
        return model_runs, analysis
    else:
        # Old format
        if "model_runs" in results:
            model_runs = results["model_runs"]
        else:
            model_runs = {}
            model_runs["baseline"] = results.get("baseline_runs", [])
            model_runs["dendritic"] = results.get("dendritic_runs", [])
            model_runs["stack"] = results.get("stack_runs", [])
        analysis = results.get("statistical_analysis", {})
        return model_runs, analysis


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
        import traceback

        traceback.print_exc()
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
    model_runs, analysis = _normalize_results(results)

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
    ax2.tick_params(axis="x", rotation=45, ha="right")

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


def save_markdown_table(result_path: str, md_path: str, points: int = 10):
    """
    Save a markdown table of perplexity values at evenly spaced training steps.

    The table columns are: step, and for each model <name>_ppl where <name> is the
    same label used in the curve legend. The table contains `points` rows, with
    the first and last rows being the actual first and last step values, and
    intermediate rows linearly interpolated between the two closest actual values.

    Args:
        result_path: Path to JSON results file
        md_path: Path where markdown file will be saved
        points: Number of rows in the table (default 10)
    """
    # Load results
    if isinstance(result_path, str):
        with open(result_path) as f:
            results = json.load(f)
    else:
        results = result_path

    # Normalize results structure
    model_runs, _ = _normalize_results(results)

    # Compute mean curves for each model
    mean_curves = {}
    model_keys = []
    for model_key, runs in model_runs.items():
        if not runs:
            continue
        model_keys.append(model_key)
        mean_curves[model_key] = compute_mean_curve(runs)

    if not model_keys:
        raise ValueError("No model runs found in results")

    # Determine global step range across all models
    all_steps = []
    for curve in mean_curves.values():
        all_steps.extend(curve["steps"])
    min_step = min(all_steps)
    max_step = max(all_steps)

    # Generate evenly spaced steps (including endpoints)
    sample_steps = np.linspace(min_step, max_step, num=points, dtype=float)
    # Ensure first and last are exactly min_step and max_step (avoid floating errors)
    sample_steps[0] = min_step
    sample_steps[-1] = max_step

    # Build header: step + each model's column name
    headers = ["step"]
    for model_key in model_keys:
        config = get_model_config(model_key)
        # Use label as shown in curves, but replace spaces with underscores for column name
        label = config["label"].replace(" ", "_")
        headers.append(f"{label}_ppl")

    # Build rows
    rows = []
    for step in sample_steps:
        row = [step]
        for model_key in model_keys:
            curve = mean_curves[model_key]
            # Interpolate perplexity at this step
            # Use linear interpolation, extrapolate with nearest value
            ppl = np.interp(
                step,
                curve["steps"],
                curve["ppl"],
                left=curve["ppl"][0],
                right=curve["ppl"][-1],
            )
            row.append(ppl)
        rows.append(row)

    # Format as markdown table
    # Determine column widths
    col_widths = [max(len(str(row[i])) for row in rows + [headers]) for i in range(len(headers))]

    # Create header separator
    sep = ["-" * w for w in col_widths]

    # Build lines
    lines = []
    lines.append("| " + " | ".join(h.ljust(w) for h, w in zip(headers, col_widths)) + " |")
    lines.append("| " + " | ".join(s.ljust(w) for s, w in zip(sep, col_widths)) + " |")
    for row in rows:
        cells = []
        for i, val in enumerate(row):
            if i == 0:
                # step: format as integer if close to integer, else keep float
                if abs(val - round(val)) < 1e-9:
                    cell = str(int(round(val)))
                else:
                    cell = f"{val:.1f}"
            else:
                # perplexity: format with 2 decimal places
                cell = f"{val:.2f}"
            cells.append(cell.ljust(col_widths[i]))
        lines.append("| " + " | ".join(cells) + " |")

    # Write to file
    with open(md_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Markdown table saved to {md_path}")


if __name__ == "__main__":
    pretraining_experiment_results_path = (
        r"results\pretraining_comparison\new_format_pretraining_experiment.json"
    )
    plot_training_curves(
        pretraining_experiment_results_path,
        show=True,
        output_path=pretraining_experiment_results_path.replace(".json", ".png"),
    )

    save_markdown_table(
        pretraining_experiment_results_path,
        pretraining_experiment_results_path.replace(".json", ".md"),
        points=10,
    )
