from scipy import stats
from typing import Any



import numpy as np


import dataclasses
import json
import logging
from datetime import datetime
from pathlib import Path

from dendritic.experiments.utils.ExperimentResults import ExperimentResults, TrainingResult



def _convert_numpy_types(obj):
    """Recursively convert numpy types and dataclasses to native Python types for JSON serialization."""
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(item) for item in obj]
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        # Convert dataclass instance to dict, then recursively convert its values
        return _convert_numpy_types(dataclasses.asdict(obj))
    # For any other object with __dict__, convert to dict (e.g., simple classes)
    elif hasattr(obj, '__dict__') and not isinstance(obj, type):
        return _convert_numpy_types(obj.__dict__)
    return obj


def _serialize_runs(runs):
    """Serialize a list of TrainingResult objects for JSON output."""
    return [
        {
            "seed": r.seed,
            "final_ppl": _convert_numpy_types(r.final_perplexity),
            "best_ppl": _convert_numpy_types(r.best_perplexity),
            "training_time": r.training_time,
            "loss_history": _convert_numpy_types(r.loss_history),
            "polynomial_stats": _convert_numpy_types(r.polynomial_stats)
        }
        for r in runs
    ]


def save_experiment_results(results: ExperimentResults, output_dir: Path) -> None:
    """Save experiment results to JSON after converting numpy types to native types."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Convert to serializable format
    output = {
        "timestamp": timestamp,
        "config": _convert_numpy_types(results.config.__dict__),
        "statistical_analysis": _convert_numpy_types(results.statistical_analysis),
        "model_runs": {
            model_name: _serialize_runs(runs)
            for model_name, runs in results.model_results.items()
        }
    }

    filepath = output_dir / f"pretraining_experiment_{timestamp}.json"
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)



def serialize_experiment_results(results: ExperimentResults) -> dict:
    """Convert ExperimentResults to a serializable dictionary."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return {
        "timestamp": timestamp,
        "config": _convert_numpy_types(results.config.__dict__),
        "statistical_analysis": _convert_numpy_types(results.statistical_analysis),
        "model_runs": {
            model_name: _serialize_runs(runs)
            for model_name, runs in results.model_results.items()
        }
    }


def save_consolidated_results(
    results_by_variant: dict[str, ExperimentResults],
    output_dir: Path,
) -> None:
    """Save consolidated experiment results from multiple variants to a single JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    variants = {}
    for variant_id, results in results_by_variant.items():
        variants[variant_id] = serialize_experiment_results(results)
    
    output = {
        "timestamp": timestamp,
        "variants": variants,
    }
    
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"pretraining_experiment_consolidated_{timestamp}.json"
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)
    
    logging.info(f"Consolidated results saved to {filepath}")


def print_consolidated_summary(results_by_variant: dict[str, ExperimentResults]) -> None:
    """Print a summary of consolidated experiment results across variants."""
    print("\n" + "=" * 70)
    print("CONSOLIDATED EXPERIMENT SUMMARY")
    print("=" * 70)
    
    for variant_id, results in results_by_variant.items():
        print(f"\n--- Variant: {variant_id} ---")
        print_experiment_summary(results)


def print_metric_summary(
    analysis: dict,
    model_keys: list,
    metric_prefix: str,
    label_map: dict[str, str] | None = None,
    mean_suffix: str = "_mean",
    std_suffix: str = "_std",
    value_format: str = "{:8.2f} ± {:5.2f}"
) -> None:
    """
    Print summary for each model's metric (mean ± std).

    Args:
        analysis: The statistical analysis dictionary.
        model_keys: List of keys for each model in analysis.
        metric_prefix: Prefix for the metric (e.g., "final_ppl", "best_ppl", "final_loss").
        label_map: Optional dict mapping model keys to display names.
        mean_suffix: Suffix for mean value in analysis dict.
        std_suffix: Suffix for std value in analysis dict.
        value_format: Format string for displaying mean and std.
    """
    if label_map is None:
        label_map = {k: k for k in model_keys}
    for k in model_keys:
        mean = analysis[k][f"{metric_prefix}{mean_suffix}"]
        std = analysis[k][f"{metric_prefix}{std_suffix}"]
        print(f"{label_map[k]:16} {value_format.format(mean, std)}")

def print_experiment_summary(results: ExperimentResults) -> None:
    """Print formatted experiment summary with all model types."""
    analysis = results.statistical_analysis

    print("\n" + "=" * 70)
    print("PRETRAINING EXPERIMENT SUMMARY")
    print("=" * 70)

    # Get model keys dynamically
    model_keys = [k for k in analysis.keys() if not k.startswith('comparison_')]
    
    # Default label mapping
    def default_label(key):
        return key.replace('_', ' ').title() + ":"
    
    # Special case labels
    label_map = {
        "stack": "Dendritic Stack:",
        "baseline_wave": "Baseline Wave:"
    }
    # Combine special cases with default
    for k in model_keys:
        if k not in label_map:
            label_map[k] = default_label(k)

    # Comparison pairs to display
    comparison_pairs = [
        ("Baseline vs Dendritic", "baseline", "dendritic"),
        ("Baseline vs Baseline Wave", "baseline", "baseline_wave"),
        ("Baseline vs Dendritic Stack", "baseline", "stack"),
        ("Dendritic vs Dendritic Stack", "dendritic", "stack")
    ]

    # Model configuration
    print("\n--- Model Configuration ---")
    print(f"Number of seeds: {len(results.config.seeds)}")
    print(f"Training steps: {results.config.training_steps:,}")
    for model in model_keys:
        dim_key = f"{model}_hidden_dim"
        if hasattr(results.config, dim_key):
            print(f"{label_map[model]} {getattr(results.config, dim_key)}")

    # Final perplexity results
    print("\n--- Final Perplexity (mean ± std) ---")
    print_metric_summary(analysis, model_keys, "final_ppl", label_map)

    # Best perplexity results
    print("\n--- Best Perplexity (mean ± std) ---")
    print_metric_summary(analysis, model_keys, "best_ppl", label_map)

    # Print comparisons
    def print_comparison(name, comp):
        print(f"\n--- {name} Comparison ---")
        print(f"Perplexity difference: {comp['ppl_difference']:7.2f}")
        print(f"Improvement:           {comp['ppl_improvement_pct']:6.1f}%")
        print(f"Paired t-test: t = {comp['paired_ttest']['t_statistic']:6.3f}, p = {comp['paired_ttest']['p_value']:6.4f}")
        print(f"Cohen's d:            {comp['cohens_d']:7.3f}")

        # Significance
        if comp['significant_001']:
            sig = "HIGHLY SIGNIFICANT (p < 0.01)"
        elif comp['significant_005']:
            sig = "SIGNIFICANT (p < 0.05)"
        else:
            sig = "NOT significant (p >= 0.05)"
        print(f"Significance:          {sig}")

        # Effect size
        d = abs(comp['cohens_d'])
        if d < 0.2:
            effect = "negligible"
        elif d < 0.5:
            effect = "small"
        elif d < 0.8:
            effect = "medium"
        else:
            effect = "large"
        print(f"Effect size:           {effect} (|d| = {d:.2f})")

    # Dynamic comparisons
    for name, model1, model2 in comparison_pairs:
        comp_key = f"comparison_{model1}_{model2}"
        if comp_key in analysis:
            print_comparison(name, analysis[comp_key])
        else:
            logging.warning(f"Comparison not found: {comp_key}")


def analyze_results(
    model_results: dict[str, list[TrainingResult]]
) -> dict[str, Any]:
    """Perform statistical analysis on experiment results.
    
    Args:
        model_results: Dictionary mapping model names to lists of TrainingResult objects.
    
    Returns:
        Dictionary containing statistical analysis results.
    """
    
    def extract_metrics(results: list[TrainingResult]) -> tuple:
        """Extract final and best perplexities from training results."""
        final_ppl = [r.final_perplexity for r in results]
        best_ppl = [r.best_perplexity for r in results]
        return final_ppl, best_ppl

    def calculate_comparison(ppl1: list[float], ppl2: list[float]) -> tuple:
        """Calculate statistical comparison between two sets of perplexities."""
        t_stat, p_value = stats.ttest_rel(ppl1, ppl2)
        diff = np.array(ppl1) - np.array(ppl2)
        cohens_d = np.mean(diff) / np.std(diff, ddof=1)
        return t_stat, p_value, cohens_d

    # Extract metrics for all models
    model_metrics = {}
    for model_name, results in model_results.items():
        final_ppl, best_ppl = extract_metrics(results)
        model_metrics[model_name] = {
            'final_ppl': final_ppl,
            'best_ppl': best_ppl
        }

    # Calculate model summaries
    analysis = {}
    for model_name, metrics in model_metrics.items():
        analysis[model_name] = {
            "final_ppl_mean": np.mean(metrics['final_ppl']),
            "final_ppl_std": np.std(metrics['final_ppl'], ddof=1),
            "best_ppl_mean": np.mean(metrics['best_ppl']),
            "best_ppl_std": np.std(metrics['best_ppl'], ddof=1),
            "individual_ppls": metrics['final_ppl']
        }

    # Define comparisons to perform


    # Perform statistical comparisons
    for base_model in [m for m in model_metrics.keys() if m.startswith("baseline")]:
        for model2 in model_metrics.keys():
            if "baseline" != model2:
                ppl1 = model_metrics[base_model]['final_ppl']
                ppl2 = model_metrics[model2]['final_ppl']
                t_stat, p_value, cohens_d = calculate_comparison(ppl1, ppl2)
            
                mean1 = np.mean(ppl1)
                mean2 = np.mean(ppl2)
                
                analysis_key = f"comparison_baseline_{model2}"
                analysis[analysis_key] = {
                    "ppl_difference": mean1 - mean2,
                    "ppl_improvement_pct": 100 * (mean1 - mean2) / mean1 if mean1 != 0 else 0.0,
                    "paired_ttest": {"t_statistic": t_stat, "p_value": p_value},
                    "cohens_d": cohens_d,
                    "significant_005": p_value < 0.05,
                    "significant_001": p_value < 0.01
                }

    return analysis