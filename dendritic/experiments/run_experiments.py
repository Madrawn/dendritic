# dendritic/experiments/run_experiments.py
"""
Main script to run dendritic layer experiments.

Usage:
    python -m dendritic.experiments.run_experiments --experiment pretraining
    python -m dendritic.experiments.run_experiments --experiment finetuning
    python -m dendritic.experiments.run_experiments --experiment both
    python -m dendritic.experiments.run_experiments --experiment confidence
"""

# Standard library imports
import argparse
import logging
import os
from pathlib import Path

import os
import subprocess


def setup_windows_compiler():
    if os.name != "nt":
        return

    # Try multiple Visual Studio versions
    vs_versions = ["2022", "2019", "2017"]
    vs_variants = ["Community", "Professional", "Enterprise", "BuildTools"]

    vcvars_path = None
    for version in vs_versions:
        base_path = rf"C:\Program Files\Microsoft Visual Studio\{version}"
        for variant in vs_variants:
            path = os.path.join(base_path, variant, r"VC\Auxiliary\Build\vcvars64.bat")
            if os.path.exists(path):
                vcvars_path = path
                print(f"Found vcvars at {vcvars_path}")
                break
        if vcvars_path:
            break

    if not vcvars_path:
        print("No Visual Studio vcvars64.bat found.")
        return

    # Disable Conda AutoRun by setting environment variable
    env = os.environ.copy()
    env["CONDA_AUTO_RUN"] = "0"

    # Try up to 3 times with a small delay
    for attempt in range(3):
        try:
            # Original command that worked before
            cmd = f'cmd.exe /d /c "{vcvars_path}" && set'
            output = subprocess.check_output(
                cmd, shell=False, text=True, stderr=subprocess.STDOUT, env=env
            )

            for line in output.splitlines():
                if "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key] = value
            print("Successfully loaded MSVC environment (bypassing Conda AutoRun).")
            return
        except subprocess.CalledProcessError as e:
            print(f"Attempt {attempt + 1} failed: {e.output}")
            if attempt < 2:
                import time

                time.sleep(1)
            else:
                print("Failed to load MSVC after 3 attempts.")
                # Fallback: try to locate cl.exe manually
                import glob

                cl_path = None
                for version in vs_versions:
                    for variant in vs_variants:
                        pattern = rf"C:\Program Files\Microsoft Visual Studio\{version}\{variant}\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe"
                        matches = glob.glob(pattern)
                        if matches:
                            cl_path = matches[0]
                            break
                    if cl_path:
                        break
                if cl_path:
                    cl_dir = os.path.dirname(cl_path)
                    os.environ["PATH"] = cl_dir + ";" + os.environ.get("PATH", "")
                    print(f"Added cl.exe directory to PATH: {cl_dir}")
                else:
                    print("Could not locate cl.exe.")


setup_windows_compiler()
import torch


torch.backends.cuda.matmul.fp32_precision = "ieee"
# torch.set_float32_matmul_precision('medium')
# torch.set_float32_matmul_precision('high')
from transformers.models.gpt2 import GPT2Tokenizer

# Project‑specific imports
from dendritic.experiments.utils.PretrainingConfig import (
    PretrainingConfig,
)
from dendritic.experiments.utils.ExperimentResults import ExperimentResults
from dendritic.experiments.utils.experiment_finetuning import (
    FinetuningConfig,
    FinetuningExperimentResults,
    load_finetuning_data,
)
from dendritic.experiments.utils.experiment_pretraining import (
    train_config_with_models,
)
from dendritic.experiments.utils.sweep import variant_identifier
from dendritic.experiments.utils.experiment_utils import (
    set_random_seed,
    setup_logging,
)

# Confidence experiment imports
from dendritic.experiments.confidence.config import ConfidenceExperimentConfig
from dendritic.experiments.confidence.experiment import ConfidenceAwareExperiment

# Environment configuration
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"  # Fix pygame spam

# Logger instance
logger = logging.getLogger(__name__)


def calculate_required_max_samples(config) -> int:
    """
    Calculate appropriate max_samples based on training configuration.

    Formula:
    1. train_blocks_needed = training_steps * batch_size
    2. test_blocks_needed = min(train_blocks_needed * eval_split_ratio,
                                batch_size * eval_batches)
    3. total_blocks_needed = train_blocks_needed + test_blocks_needed

    When grouped=False (default), each sample produces at most 1 block.
    When grouped=True, samples are concatenated and can produce multiple blocks.

    Returns:
        Appropriate max_samples value
    """
    train_blocks = config.training_steps * config.batch_size
    test_blocks = min(
        int(train_blocks * config.eval_split_ratio),
        config.batch_size * config.eval_batches,
    )
    total_blocks = train_blocks + test_blocks

    if getattr(config, "grouped", False):
        # When grouped=True, samples are concatenated
        # Average blocks per sample depends on average tokens per sample
        # Conservative estimate: avg_tokens_per_sample = 2048
        avg_tokens_per_sample = 2048
        blocks_per_sample = avg_tokens_per_sample / config.max_seq_len
        safety_factor = 1.5  # Account for variability
        required_samples = int(total_blocks / blocks_per_sample * safety_factor)
    else:
        # When grouped=False, each sample produces at most 1 block
        # Need extra samples for those shorter than max_seq_len
        safety_factor = 1.2  # 20% extra
        required_samples = int(total_blocks * safety_factor)

    # Ensure minimum samples
    min_samples = 1000
    return max(required_samples, min_samples)


def run_pretraining_experiment(
    device: str,
    base_config: PretrainingConfig,
    num_workers: int | None = None,
    scheduler_variants: list[PretrainingConfig] | None = None,
    param_grid: dict | None = None,
) -> dict[str, ExperimentResults]:
    """Run the pretraining experiment across configured seeds.

    Parameters
    ----------
    device : str
        Device identifier (e.g., ``'cuda'`` or ``'cpu'``).
    num_workers : int | None, optional
        Number of worker processes for data loading. If ``None`` the default
        is derived from the CLI argument.
    scheduler_variants : list[PretrainingConfig] | None, optional
        List of configuration variants to run. If None, a single default config is used.
    param_grid : dict | None, optional
        Mapping of field names to lists of values to sweep over.
        Field names can be dot‑separated to target nested attributes (e.g.,
        "dropout" or "cohort_scheduler.min_mult").
        If provided, generates configuration variants via Cartesian product.
        Example: {"dropout": [0.0, 0.3], "cohort_scheduler.min_mult": [0.2, 0.3]}
    base_config : PretrainingConfig | None, optional
        Base configuration to use. If None, a default config is created.

    Returns
    -------
    dict[str, ExperimentResults]
        Mapping from variant identifier to experiment results.
    """

    from dendritic.experiments.utils.sweep import generate_scheduler_variants

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Determine which configs to run
    if param_grid is not None:
        variants = generate_scheduler_variants(base_config, param_grid)
    elif scheduler_variants is not None:
        variants = scheduler_variants
    else:
        variants = [base_config]

    results_by_variant: dict[str, ExperimentResults] = {}

    scheduler_variants_list = [cfg for cfg in variants]

    for cfg in scheduler_variants_list:
        logger.info(f"Pretraining config: {cfg}")
        variant_id = variant_identifier(cfg.param_grid)
        logger.info(f"Training variant: {variant_id}")
        wave_results = train_config_with_models(cfg, device, num_workers, tokenizer)
        results_by_variant[variant_id] = wave_results

    # Save consolidated results
    from dendritic.experiments.analysis.analysis import (
        save_consolidated_results,
        print_consolidated_summary,
    )

    output_dir = Path(base_config.output_dir)
    save_consolidated_results(results_by_variant, output_dir)
    print_consolidated_summary(results_by_variant)

    return results_by_variant


def main() -> None:
    """Entry point for the experiment runner.

    Parses command‑line arguments, configures logging, and dispatches
    the requested experiments (pretraining, finetuning, or both).
    """
    parser = argparse.ArgumentParser(description="Run dendritic layer experiments")
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["pretraining", "finetuning", "both", "confidence"],
        default="both",
        help="Which experiment to run",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset identifier for pretraining (e.g., 'wikitext', 'openwebmath')",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of worker processes for data loading (default: 0)",
    )

    args = parser.parse_args()
    # Map textual log level to logging constant; default to INFO if not provided
    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    logger = setup_logging(log_level=log_level_map.get(args.log_level))

    logger.info(f"Running experiment(s): {args.experiment}")
    logger.info(f"Device: {args.device}")

    if args.experiment in ["pretraining", "both"]:
        logger.info("\n" + "=" * 70)
        logger.info("RUNNING PRETRAINING EXPERIMENT")
        logger.info("=" * 70)
        param_grid = {
            "cohort_scheduler.min_mult": [None, 0.0],
            "layer_type": ["standard", "dendritic"],
        }  # Example: sweep over dropout values
        base_config = PretrainingConfig(
            # training_steps=60,
            seeds=[24],  # Use fewer seeds for faster testing,
        )
        run_pretraining_experiment(
            device=args.device,
            base_config=base_config,
            param_grid=param_grid,
        )

    if args.experiment in ["finetuning", "both"]:
        logger.info("\n" + "=" * 70)
        logger.info("RUNNING FINETUNING EXPERIMENT")
        logger.info("=" * 70)
        run_finetuning_experiment_wrapper(args.device, args.num_workers)

    if args.experiment == "confidence":
        logger.info("\n" + "=" * 70)
        logger.info("RUNNING CONFIDENCE-AWARE EXPERIMENT")
        logger.info("=" * 70)
        # Create tokenizer (same as other experiments)
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        # Create configuration with reasonable defaults
        config = ConfidenceExperimentConfig(
            training_steps=650,  # Reasonable for PoC
            seeds=[42],  # , 123, 456],  # Multiple seeds for statistical significance
            batch_size=10,
            vocab_size=50257,  # GPT-2 vocab size
            embed_dim=400,
            num_heads=8,
            num_layers=6,
            max_seq_len=512,
            dropout=0.1,
            scheduler_type="cosine",
            results_dir="results/confidence_experiments",
            dataset="tinystories",
        )

        # Calculate appropriate max_samples based on training configuration
        # When grouped=False (default), each sample produces at most 1 block
        required_max_samples = calculate_required_max_samples(config)
        config.dataset_kwargs = {"max_samples": required_max_samples}

        logger.info(
            f"Calculated max_samples: {required_max_samples} "
            f"(based on {config.training_steps} steps, batch_size={config.batch_size})"
        )

        # Run the experiment
        experiment = ConfidenceAwareExperiment(config)
        results = experiment.run(tokenizer)
        logger.info(
            f"Confidence experiment completed. Results saved to {config.results_dir}"
        )

    logger.info("\nAll experiments complete!")


def run_finetuning_experiment_wrapper(
    device: str, num_workers: int | None = None
) -> FinetuningExperimentResults:
    """Run the finetuning experiment across configured seeds.

    Parameters
    ----------
    device : str
        Device identifier (e.g., ``'cuda'`` or ``'cpu'``).
    num_workers : int | None, optional
        Number of worker processes for data loading. If ``None`` the default
        is derived from the CLI argument.

    Returns
    -------
    FinetuningExperimentResults
        Results of the finetuning experiment.
    """

    # Ensure deterministic behavior
    set_random_seed(42)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    config = FinetuningConfig(
        training_steps=30, batch_size=4, eval_interval=250, seeds=[42, 71, 123]
    )

    train_dl, eval_dl = load_finetuning_data(
        tokenizer,
        max_length=config.max_length,
        batch_size=config.batch_size,
        num_workers=num_workers,
    )

    from dendritic.experiments.utils.experiment_finetuning import (
        run_finetuning_experiment as run_finetuning_exp,
    )

    results: FinetuningExperimentResults | None = None
    try:
        results = run_finetuning_exp(train_dl, eval_dl, config, device)
    finally:
        torch.cuda.empty_cache()
    if results is None:
        raise RuntimeError("Finetuning experiment failed without returning results")
    return results


if __name__ == "__main__":
    main()
