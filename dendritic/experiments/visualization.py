# dendritic/experiments/visualization.py
"""Visualization utilities for experiment results."""

from typing import Dict, List, Optional
import json

import numpy as np
import matplotlib.pyplot as plt


def plot_training_curves(
    results_path: str,
    output_path: Optional[str] = None,
    show: bool = True
):
    """
    Plot training curves from experiment results.
    
    Args:
        results_path: Path to JSON results file
        output_path: Optional path to save figure
        show: Whether to display the plot
    """
    with open(results_path) as f:
        results = json.load(f)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Determine experiment type
    if "baseline_runs" in results and "dendritic_runs" in results:
        # Pretraining experiment (with or without stack)
        plot_pretraining_curves(results, axes)
    elif "dendritic_runs" in results and "lora_runs" in results:
        # Finetuning experiment
        plot_finetuning_curves(results, axes)
    else:
        print("[DEBUG] WARNING: Unrecognized experiment type! No curves plotted.")
        print(f"[DEBUG] Results contained keys: {list(results.keys())}")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    
    return fig


def plot_pretraining_curves(results: Dict, axes):
    """Plot pretraining experiment curves."""
    ax1, ax2 = axes
    
    # Colors
    baseline_color = "#1f77b4"
    dendritic_color = "#ff7f0e"
    stack_color = "#2ca02c"  # Green for stack
    
    # Plot individual runs with transparency
    for run in results["baseline_runs"]:
        steps = [h["step"] for h in run["loss_history"]]
        ppls = [h["perplexity"] for h in run["loss_history"]]
        ax1.plot(steps, ppls, color=baseline_color, alpha=0.3, linewidth=1)
    
    for run in results["dendritic_runs"]:
        steps = [h["step"] for h in run["loss_history"]]
        ppls = [h["perplexity"] for h in run["loss_history"]]
        ax1.plot(steps, ppls, color=dendritic_color, alpha=0.3, linewidth=1)
    # Plot individual runs for stack if present
    if "stack_runs" in results:
        for run in results["stack_runs"]:
            steps = [h["step"] for h in run["loss_history"]]
            ppls = [h["perplexity"] for h in run["loss_history"]]
            ax1.plot(steps, ppls, color=stack_color, alpha=0.3, linewidth=1)
    
    
    # Plot mean curves
    baseline_mean = compute_mean_curve(results["baseline_runs"])
    dendritic_mean = compute_mean_curve(results["dendritic_runs"])
    # Plot stack mean curve if present
    if "stack_runs" in results:
        stack_mean = compute_mean_curve(results["stack_runs"])
        ax1.plot(stack_mean["steps"], stack_mean["ppl"],
                 color=stack_color, linewidth=2.5, label="Dendritic Stack")
    
    ax1.plot(baseline_mean["steps"], baseline_mean["ppl"],
             color=baseline_color, linewidth=2.5, label="Baseline")
    ax1.plot(dendritic_mean["steps"], dendritic_mean["ppl"],
             color=dendritic_color, linewidth=2.5, label="Dendritic")
    
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Perplexity")
    ax1.set_title("Training Curves (Perplexity)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bar chart of final results
    analysis = results["statistical_analysis"]
    # Build bar chart data conditionally
    methods = ["Baseline", "Dendritic"]
    means = [analysis["baseline"]["final_ppl_mean"],
             analysis["dendritic"]["final_ppl_mean"]]
    stds = [analysis["baseline"]["final_ppl_std"],
            analysis["dendritic"]["final_ppl_std"]]
    colors = [baseline_color, dendritic_color]
    
    if "stack" in analysis:
        methods.append("Dendritic Stack")
        means.append(analysis["stack"]["final_ppl_mean"])
        stds.append(analysis["stack"]["final_ppl_std"])
        colors.append(stack_color)
    
    bars = ax2.bar(methods, means, yerr=stds, capsize=5,
                   color=colors, alpha=0.7)
    
    ax2.set_ylabel("Final Perplexity")
    ax2.set_title("Final Performance Comparison")
    print("[DeBUG] Keys in analysis:", analysis.keys())
    # Add significance annotations for all three comparisons
    y_pos = 0.95  # Start near top of plot
    y_step = 0.05  # Vertical spacing between annotations
    
    # Baseline vs Dendritic
    comp_baseline_dendritic = analysis["comparison_baseline_dendritic"]
    if comp_baseline_dendritic["significant_001"]:
        sig_text = "Baseline vs Dendritic: *** p < 0.01"
    elif comp_baseline_dendritic["significant_005"]:
        sig_text = "Baseline vs Dendritic: * p < 0.05"
    else:
        sig_text = None
        
    if sig_text:
        ax2.annotate(sig_text, xy=(0.5, y_pos), xycoords="axes fraction",
                     ha="center", fontsize=10, color="#1f77b4")
        y_pos -= y_step
    
    # Baseline vs Stack
    if "comparison_baseline_stack" in analysis:
        comp_baseline_stack = analysis["comparison_baseline_stack"]
        if comp_baseline_stack["significant_001"]:
            sig_text = "Baseline vs Stack: *** p < 0.01"
        elif comp_baseline_stack["significant_005"]:
            sig_text = "Baseline vs Stack: * p < 0.05"
        else:
            sig_text = None
            
        if sig_text:
            ax2.annotate(sig_text, xy=(0.5, y_pos), xycoords="axes fraction",
                         ha="center", fontsize=10, color="#2ca02c")
            y_pos -= y_step
    
    # Dendritic vs Stack
    if "comparison_dendritic_stack" in analysis:
        comp_dendritic_stack = analysis["comparison_dendritic_stack"]
        if comp_dendritic_stack["significant_001"]:
            sig_text = "Dendritic vs Stack: *** p < 0.01"
        elif comp_dendritic_stack["significant_005"]:
            sig_text = "Dendritic vs Stack: * p < 0.05"
        else:
            sig_text = None
            
        if sig_text:
            ax2.annotate(sig_text, xy=(0.5, y_pos), xycoords="axes fraction",
                         ha="center", fontsize=10, color="#ff7f0e")


def plot_finetuning_curves(results: Dict, axes):
    """Plot finetuning experiment curves."""
    ax1, ax2 = axes
    
    # Define colors for all methods
    baseline_color = "gray"
    lora_color = "#2ca02c"
    dendritic_color = "#ff7f0e"
    stack_color = "#1f77b4"  # Blue for dendritic_stack
    
    # Plot individual runs with appropriate colors
    for run_type, color in [
        ("lora_runs", lora_color),
        ("dendritic_runs", dendritic_color),
        ("dendritic_stack_runs", stack_color)
    ]:
        if run_type in results:
            for run in results[run_type]:
                steps = [h["step"] for h in run["loss_history"]]
                ppls = [h["perplexity"] for h in run["loss_history"]]
                ax1.plot(steps, ppls, color=color, alpha=0.3, linewidth=1)
    
    # Plot mean curves with labels
    if "lora_runs" in results:
        lora_mean = compute_mean_curve(results["lora_runs"])
        ax1.plot(lora_mean["steps"], lora_mean["ppl"],
                 color=lora_color, linewidth=2.5, label="LoRA")
    
    if "dendritic_runs" in results:
        dendritic_mean = compute_mean_curve(results["dendritic_runs"])
        ax1.plot(dendritic_mean["steps"], dendritic_mean["ppl"],
                 color=dendritic_color, linewidth=2.5, label="Dendritic")
    
    if "dendritic_stack_runs" in results:
        stack_mean = compute_mean_curve(results["dendritic_stack_runs"])
        ax1.plot(stack_mean["steps"], stack_mean["ppl"],
                 color=stack_color, linewidth=2.5, label="Dendritic Stack")
    
    # Baseline line
    baseline_ppl = results["statistical_analysis"]["baseline_ppl"]
    ax1.axhline(y=baseline_ppl, color=baseline_color, linestyle="--",
                label=f"Baseline ({baseline_ppl:.1f})")
    
    ax1.set_xlabel("Training Step")
    ax1.set_ylabel("Perplexity")
    ax1.set_title("Finetuning Curves")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bar chart
    analysis = results["statistical_analysis"]
    methods = ["Baseline"]
    means = [baseline_ppl]
    stds = [0]
    colors = [baseline_color]
    
    # Add results for all methods present in analysis
    for method, color in [
        ("lora", lora_color),
        ("dendritic", dendritic_color),
        ("dendritic_stack", stack_color)
    ]:
        if method in analysis:
            methods.append(method.replace("_", " ").title())
            means.append(analysis[method]["final_ppl_mean"])
            stds.append(analysis[method]["final_ppl_std"])
            colors.append(color)
    
    bars = ax2.bar(methods, means, yerr=stds, capsize=5, color=colors, alpha=0.7)
    
    ax2.set_ylabel("Final Perplexity")
    ax2.set_title("Final Performance Comparison")
    
    # Significance annotation - handle missing comparison data
    if "comparison" in analysis:
        comp = analysis["comparison"]
        if comp.get("significant_001", False):
            sig_text = "*** p < 0.01"
        elif comp.get("significant_005", False):
            sig_text = "* p < 0.05"
        else:
            sig_text = "n.s."
        
        # Add comparison text if available
        if "dendritic_better_by_pct" in comp:
            comparison_text = f"Dendritic vs LoRA: {sig_text} (Better by {comp['dendritic_better_by_pct']:.1f}%)"
        else:
            comparison_text = f"Dendritic vs LoRA: {sig_text}"
    else:
        # Fallback if comparison data is missing
        comparison_text = "Statistical comparison not available"
    
    ax2.annotate(comparison_text,
                 xy=(0.5, 0.95), xycoords="axes fraction",
                 ha="center", fontsize=11)


def compute_mean_curve(runs: List[Dict]) -> Dict[str, List[float]]:
    """Compute mean curve across runs."""
    # Align steps
    all_steps = set()
    for run in runs:
        all_steps.update(h["step"] for h in run["loss_history"])
    steps = sorted(all_steps)
    
    # Interpolate values at each step
    ppl_at_step = {s: [] for s in steps}
    for run in runs:
        run_steps = [h["step"] for h in run["loss_history"]]
        run_ppl = [h["perplexity"] for h in run["loss_history"]]
        
        for s in steps:
            if s in run_steps:
                idx = run_steps.index(s)
                ppl_at_step[s].append(run_ppl[idx])
    
    # Convert to native float and handle empty cases
    mean_ppl = [float(np.mean(ppl_at_step[s])) if ppl_at_step[s] else 0.0 for s in steps]
    
    return {"steps": steps, "ppl": mean_ppl}


def generate_latex_table(results_path: str) -> str:
    """Generate LaTeX table from experiment results."""
    with open(results_path) as f:
        results = json.load(f)
    
    analysis = results["statistical_analysis"]
    
    if "baseline_runs" in results:
        # Pretraining table
        return generate_pretraining_table(results)
    else:
        # Finetuning table
        return generate_finetuning_table(results)


def generate_pretraining_table(results: Dict) -> str:
    """Generate LaTeX table for pretraining results with all three model comparisons."""
    analysis = results["statistical_analysis"]
    
    table = r"""
\begin{table}[h]
\centering
\caption{Pretraining Performance Comparison: Baseline vs Dendritic MLP vs Dendritic Stack}
\begin{tabular}{lcccc}
\toprule
Method & Final PPL & Best PPL & $\Delta$ PPL & Improvement \\
\midrule
"""
    
    baseline = analysis["baseline"]
    dendritic = analysis["dendritic"]
    stack = analysis["stack"]
    comp = analysis["comparison"]
    
    table += f"Baseline & {baseline['final_ppl_mean']:.2f} $\\pm$ {baseline['final_ppl_std']:.2f} & "
    table += f"{baseline['best_ppl_mean']:.2f} $\\pm$ {baseline['best_ppl_std']:.2f} & - & - \\\\\n"
    
    table += f"Dendritic & {dendritic['final_ppl_mean']:.2f} $\\pm$ {dendritic['final_ppl_std']:.2f} & "
    table += f"{dendritic['best_ppl_mean']:.2f} $\\pm$ {dendritic['best_ppl_std']:.2f} & "
    table += f"{comp['baseline_vs_dendritic']['ppl_difference']:.2f} & {comp['baseline_vs_dendritic']['ppl_improvement_pct']:.1f}\\% \\\\\n"
    
    table += f"Dendritic Stack & {stack['final_ppl_mean']:.2f} $\\pm$ {stack['final_ppl_std']:.2f} & "
    table += f"{stack['best_ppl_mean']:.2f} $\\pm$ {stack['best_ppl_std']:.2f} & "
    table += f"{comp['baseline_vs_stack']['ppl_difference']:.2f} & {comp['baseline_vs_stack']['ppl_improvement_pct']:.1f}\\% \\\\\n"
    
    # Significance markers for all three comparisons
    p_val_dendritic = comp['baseline_vs_dendritic']['paired_ttest']['p_value']
    p_val_stack = comp['baseline_vs_stack']['paired_ttest']['p_value']
    p_val_dendritic_vs_stack = comp['dendritic_vs_stack']['paired_ttest']['p_value']
    
    sig_marker_dendritic = "***" if p_val_dendritic < 0.01 else ("*" if p_val_dendritic < 0.05 else "")
    sig_marker_stack = "***" if p_val_stack < 0.01 else ("*" if p_val_stack < 0.05 else "")
    sig_marker_dendritic_vs_stack = "***" if p_val_dendritic_vs_stack < 0.01 else ("*" if p_val_dendritic_vs_stack < 0.05 else "")
    
    table += r"""
\bottomrule
\end{tabular}
"""
    table += f"\\\\[0.5em]\n\\footnotesize{{Paired t-tests: Baseline vs Dendritic: $t = {comp['baseline_vs_dendritic']['paired_ttest']['t_statistic']:.3f}$, "
    table += f"$p = {p_val_dendritic:.4f}${sig_marker_dendritic} (Cohen's $d = {comp['baseline_vs_dendritic']['cohens_d']:.3f}$); "
    table += f"Baseline vs Stack: $t = {comp['baseline_vs_stack']['paired_ttest']['t_statistic']:.3f}$, "
    table += f"$p = {p_val_stack:.4f}${sig_marker_stack} (Cohen's $d = {comp['baseline_vs_stack']['cohens_d']:.3f}$); "
    table += f"Dendritic vs Stack: $t = {comp['dendritic_vs_stack']['paired_ttest']['t_statistic']:.3f}$, "
    table += f"$p = {p_val_dendritic_vs_stack:.4f}${sig_marker_dendritic_vs_stack} (Cohen's $d = {comp['dendritic_vs_stack']['cohens_d']:.3f}$).}}\n"
    table += r"\end{table}"
    
    return table


def generate_finetuning_table(results: Dict) -> str:
    """Generate LaTeX table for finetuning results."""
    analysis = results["statistical_analysis"]
    params = results["matched_params"]
    
    table = r"""
\begin{table}[h]
\centering
\caption{Finetuning Comparison: LoRA vs Dendritic Enhancement}
\begin{tabular}{lccccc}
\toprule
Method & Trainable Params & Final PPL & $\Delta$ vs Baseline & vs LoRA \\
\midrule
"""
    
    table += f"Baseline & 0 & {analysis['baseline_ppl']:.2f} & - & - \\\\\n"
    
    lora = analysis["lora"]
    dendritic = analysis["dendritic"]
    comp = analysis["comparison"]
    
    table += f"LoRA (r={params['lora_rank']}) & {params['lora']:,} & "
    table += f"{lora['final_ppl_mean']:.2f} $\\pm$ {lora['final_ppl_std']:.2f} & "
    table += f"{lora['improvement_over_baseline_pct']:.1f}\\% & - \\\\\n"
    
    table += f"Dendritic & {params['dendritic']:,} & "
    table += f"{dendritic['final_ppl_mean']:.2f} $\\pm$ {dendritic['final_ppl_std']:.2f} & "
    table += f"{dendritic['improvement_over_baseline_pct']:.1f}\\% & "
    table += f"+{comp['dendritic_better_by_pct']:.1f}\\% \\\\\n"
    
    p_val = comp['paired_ttest']['p_value']
    sig_marker = "***" if p_val < 0.01 else ("*" if p_val < 0.05 else "")
    
    table += r"""
\bottomrule
\end{tabular}
"""
    table += f"\\\\[0.5em]\n\\footnotesize{{Dendritic vs LoRA: $t = {comp['paired_ttest']['t_statistic']:.3f}$, "
    table += f"$p = {p_val:.4f}${sig_marker}. Cohen's $d = {comp['cohens_d']:.3f}$.}}\n"
    table += r"\end{table}"
    
    return table


if __name__ == "__main__":
    plot_training_curves(r"results\pretraining_comparison\pretraining_experiment_20251214_234241.json", show=True)
    # import argparse
    # parser = argparse.ArgumentParser(description='Visualize experiment results')
    # parser.add_argument('results_path',
    #                     help='Path to experiment results JSON file')
    # parser.add_argument('--plot', action='store_true',
    #                     help='Generate training curves plot')
    # parser.add_argument('--output_plot', type=str, default=None,
    #                     help='Output path for plot image')
    # parser.add_argument('--table', action='store_true',
    #                     help='Generate LaTeX table')
    # parser.add_argument('--output_table', type=str, default=None,
    #                     help='Output path for LaTeX table')
    
    # args = parser.parse_args()
    
    # if args.plot:
    #     plot_training_curves(
    #         args.results_path,
    #         output_path=args.output_plot,
    #         show=args.output_plot is None
    #     )
    
    # if args.table:
    #     table = generate_latex_table(args.results_path)
    #     if args.output_table:
    #         with open(args.output_table, 'w') as f:
    #             f.write(table)
    #     else:
    #         print(table)