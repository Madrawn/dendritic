#!/usr/bin/env python3
"""
Script to automatically plot the latest confidence experiment results.

This script:
1. Finds the most recent confidence experiment results file
2. Generates comprehensive visualizations
3. Saves plots to the current directory
4. Prints summary statistics

Usage:
    python plot_confidence_results.py [results_file]

If no results_file is provided, it automatically finds the latest one.
"""

import sys
import json
from pathlib import Path
from datetime import datetime
import argparse

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from dendritic.experiments.confidence.visualization import (
    plot_loss_curves,
    plot_calibration_curve,
    plot_training_time_comparison,
    generate_summary_statistics,
)


def find_latest_results_file(results_dir="results/confidence_experiments"):
    """
    Find the most recent confidence experiment results file.

    Args:
        results_dir: Directory containing results files

    Returns:
        Path to the most recent results file, or None if no files found
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Error: Results directory '{results_dir}' not found.")
        return None

    # Find all JSON results files (excluding final_results.json for timestamp comparison)
    results_files = []
    for file_path in results_path.glob("*_results.json"):
        if file_path.name == "final_results.json":
            continue

        # Extract timestamp from filename (format: YYYYMMDD_HHMMSS_results.json)
        try:
            timestamp_str = file_path.stem.replace("_results", "")
            file_dt = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            results_files.append((file_dt, file_path))
        except ValueError:
            # If filename doesn't match pattern, use modification time
            file_dt = datetime.fromtimestamp(file_path.stat().st_mtime)
            results_files.append((file_dt, file_path))

    if not results_files:
        # Check for final_results.json as fallback
        final_results = results_path / "final_results.json"
        if final_results.exists():
            print("No timestamped results found, using final_results.json")
            return final_results
        else:
            print("No results files found.")
            return None

    # Sort by timestamp (most recent first)
    results_files.sort(reverse=True)
    latest_dt, latest_file = results_files[0]

    print(f"Found {len(results_files)} results files")
    print(
        f"Latest file: {latest_file.name} (from {latest_dt.strftime('%Y-%m-%d %H:%M:%S')})"
    )

    return latest_file


def plot_results(results_file, output_prefix="confidence"):
    """
    Plot all visualizations for a given results file.

    Args:
        results_file: Path to results JSON file
        output_prefix: Prefix for output plot files
    """
    results_file = Path(results_file)
    if not results_file.exists():
        print(f"Error: Results file '{results_file}' not found.")
        return False

    print(f"\nPlotting results from: {results_file}")
    print("-" * 60)

    try:
        # 1. Plot comprehensive loss curves
        print("1. Generating loss curves plot...")
        loss_curves_path = f"{output_prefix}_loss_curves.png"
        fig1 = plot_loss_curves(
            results=str(results_file),
            output_path=loss_curves_path,
            show=False,  # Don't show interactively when saving
            figsize=(14, 10),
        )
        print(f"   Saved to: {loss_curves_path}")

        # 2. Plot calibration analysis
        print("2. Generating calibration curve plot...")
        calibration_path = f"{output_prefix}_calibration.png"
        fig2 = plot_calibration_curve(
            results=str(results_file),
            output_path=calibration_path,
            show=False,
            figsize=(10, 8),
        )
        print(f"   Saved to: {calibration_path}")

        # 3. Plot training time comparison
        print("3. Generating training time comparison plot...")
        training_time_path = f"{output_prefix}_training_time.png"
        fig3 = plot_training_time_comparison(
            results=str(results_file),
            output_path=training_time_path,
            show=False,
            figsize=(8, 6),
        )
        print(f"   Saved to: {training_time_path}")

        # 4. Generate and print summary statistics
        print("4. Generating summary statistics...")
        stats = generate_summary_statistics(str(results_file))

        print("\n" + "=" * 60)
        print("SUMMARY STATISTICS")
        print("=" * 60)

        for model_type, metrics in stats.items():
            if model_type == "comparison":
                print(f"\nCOMPARISON:")
                print("-" * 40)
                for key, value in metrics.items():
                    if key == "relative_improvement_percent":
                        sign = "+" if value > 0 else ""
                        print(f"  {key}: {sign}{value:.2f}%")
                    else:
                        print(f"  {key}: {value:.4f}")
            elif metrics:
                print(f"\n{model_type.upper()} MODEL:")
                print("-" * 40)
                for key, value in metrics.items():
                    print(f"  {key}: {value:.4f}")

        # 5. Save statistics to JSON file
        stats_path = f"{output_prefix}_summary_stats.json"
        with open(stats_path, "w") as f:
            # Convert numpy types to Python native types for JSON serialization
            def convert_types(obj):
                if isinstance(obj, dict):
                    return {k: convert_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_types(v) for v in obj]
                elif hasattr(obj, "item"):  # numpy scalar
                    return obj.item()
                else:
                    return obj

            json.dump(convert_types(stats), f, indent=2)
        print(f"\nStatistics saved to: {stats_path}")

        print("\n" + "=" * 60)
        print(f"All plots and statistics saved with prefix: '{output_prefix}_'")
        print("=" * 60)

        return True

    except Exception as e:
        print(f"Error plotting results: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Plot confidence experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                         # Plot latest results automatically
  %(prog)s results/confidence_experiments/20260128_093429_results.json
  %(prog)s --prefix my_experiment  # Custom output prefix
        """,
    )
    parser.add_argument(
        "results_file",
        nargs="?",
        help="Path to results JSON file (optional, defaults to latest)",
    )
    parser.add_argument(
        "--prefix",
        default="confidence",
        help="Prefix for output files (default: 'confidence')",
    )
    parser.add_argument(
        "--dir",
        default="results/confidence_experiments",
        help="Results directory to search (default: 'results/confidence_experiments')",
    )

    args = parser.parse_args()

    # Determine which results file to use
    if args.results_file:
        results_file = Path(args.results_file)
        if not results_file.exists():
            print(f"Error: Specified results file '{args.results_file}' not found.")
            sys.exit(1)
    else:
        print("No results file specified, searching for latest...")
        results_file = find_latest_results_file(args.dir)
        if not results_file:
            print("Could not find any results files.")
            sys.exit(1)

    # Plot the results
    success = plot_results(results_file, args.prefix)

    if success:
        print("\n✅ Successfully plotted confidence experiment results!")
        sys.exit(0)
    else:
        print("\n❌ Failed to plot results.")
        sys.exit(1)


if __name__ == "__main__":
    main()
