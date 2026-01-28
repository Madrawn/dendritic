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
import multiprocessing
import os
from datetime import datetime
from pathlib import Path
import random
from typing import Any
import numpy as np

# Third‑party imports
import torch

from dendritic.experiments.utils.param_utils import find_matching_hidden_dims

torch.backends.cuda.matmul.fp32_precision = "ieee"
# torch.set_float32_matmul_precision('medium')
# torch.set_float32_matmul_precision('high')
from torch.utils.data import DataLoader
from transformers.models.gpt2 import GPT2Tokenizer

# Project‑specific imports
from dendritic.experiments.utils.PretrainingConfig import (
    CohortSchedulerConfig,
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
    debug_dataset_integrity,
)
from dendritic.dataset_handlers.factory import get_handler

# Confidence experiment imports
from dendritic.experiments.confidence.config import ConfidenceExperimentConfig
from dendritic.experiments.confidence.experiment import ConfidenceAwareExperiment

# Environment configuration
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"  # Fix pygame spam

# Logger instance
logger = logging.getLogger(__name__)


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
            training_steps=1000,  # Reasonable for PoC
            seeds=[42, 123, 456],  # Multiple seeds for statistical significance
            batch_size=4,
            vocab_size=50257,  # GPT-2 vocab size
            embed_dim=256,
            num_heads=8,
            num_layers=6,
            max_seq_len=128,
            dropout=0.1,
            results_dir="results/confidence_experiments",
        )
        
        # Run the experiment
        experiment = ConfidenceAwareExperiment(config)
        results = experiment.run(tokenizer)
        logger.info(f"Confidence experiment completed. Results saved to {config.results_dir}")

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
