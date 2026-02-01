import json
import glob
import os
from tabulate import tabulate
import pandas as pd


def load_result_files(results_dir="results"):
    """Load all JSON result files from the results directory"""
    result_files = glob.glob(os.path.join(results_dir, "*.json"))
    results = []
    for file_path in result_files:
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                # Extract key metrics for comparison
                row = {
                    "experiment_id": data.get("experiment_id", "N/A"),
                    "method": data.get("method", "N/A"),
                    "timestamp": data.get("timestamp", "N/A"),
                    "final_ppl": data.get("metrics", {}).get("final_ppl", float("nan")),
                    "best_ppl": data.get("metrics", {}).get(
                        "best_eval_ppl", float("nan")
                    ),
                    "baseline_ppl": data.get("metrics", {}).get(
                        "baseline_ppl", float("nan")
                    ),
                    "rel_improvement": data.get("metrics", {}).get(
                        "relative_improvement", float("nan")
                    ),
                    "train_time_min": data.get("resources", {}).get(
                        "total_time_min", float("nan")
                    ),
                    "trainable_params": data.get("resources", {}).get(
                        "trainable_params", 0
                    ),
                    "total_params": data.get("resources", {}).get("total_params", 0),
                }
                # Calculate improvement over baseline
                if pd.notna(row["baseline_ppl"]) and pd.notna(row["final_ppl"]):
                    row["abs_improvement"] = row["baseline_ppl"] - row["final_ppl"]
                    row["rel_improvement"] = (
                        row["abs_improvement"] / row["baseline_ppl"]
                    )
                else:
                    row["abs_improvement"] = float("nan")
                    row["rel_improvement"] = float("nan")

                results.append(row)

        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
    return results


def compare_experiments():
    """Compare all experiments and display results in a table"""
    results = load_result_files()
    if not results:
        print("No experiment results found in results/ directory")
        return

    # Sort by method and timestamp
    results.sort(key=lambda x: (x["method"], x["timestamp"]))

    # Create pandas dataframe for easier manipulation
    df = pd.DataFrame(results)

    if len(df) == 0:
        print("No valid experiment data to display")
        return

    # Select columns to display
    display_columns = {
        "experiment_id": "Experiment ID",
        "method": "Method",
        "train_time_min": "Time (min)",
        "trainable_params": "Trainable Params",
        "final_ppl": "Final PPL",
        "best_ppl": "Best PPL",
        "baseline_ppl": "Baseline PPL",
        "abs_improvement": "Δ PPL",  # Absolute improvement
        "rel_improvement": "Improv. %",  # Relative improvement percentage
    }

    # Format the dataframe for display
    display_df = df[list(display_columns.keys())].copy()

    # Rename columns for display
    display_df = display_df.rename(columns=display_columns)

    # Format numbers for better readability
    display_df["Trainable Params"] = (
        display_df["Trainable Params"] // 1e6
    )  # Convert to millions
    display_df[["Time (min)", "Final PPL", "Best PPL", "Baseline PPL", "Δ PPL"]] = (
        display_df[
            ["Time (min)", "Final PPL", "Best PPL", "Baseline PPL", "Δ PPL"]
        ].round(2)
    )
    display_df["Improv. %"] = (display_df["Improv. %"] * 100).round(1)

    # Print tabular comparison
    print("\nExperiment Comparison")
    print("=" * 100)
    print(
        tabulate(
            display_df.values.tolist(),
            headers=display_df.columns.tolist(),
            tablefmt="grid",
            floatfmt=".2f",
            numalign="right",
        )
    )

    # Save to markdown file
    markdown_path = "experiment_comparison.md"
    with open(markdown_path, "w") as f:
        f.write("# Experiment Comparison Results\n\n")
        f.write(
            tabulate(
                display_df.values.tolist(),
                headers=display_df.columns.tolist(),
                tablefmt="pipe",
                floatfmt=".2f",
                numalign="right",
            )
        )

    print(f"\nComparison results saved to {markdown_path}")


if __name__ == "__main__":
    compare_experiments()
