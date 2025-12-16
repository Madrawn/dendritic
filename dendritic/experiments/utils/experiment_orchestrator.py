# dendritic/experiments/experiment_orchestrator.py
"""
Orchestrator module for running the pretraining experiment.
This file contains the implementation of `run_pretraining_experiment`,
extracted from `experiment_pretraining.py` to keep the original module lightweight.
"""

import logging
import torch
from pathlib import Path
from typing import Optional, Tuple

from experiments.analysis.analysis import analyze_results, print_experiment_summary, save_experiment_results

from .experiment_pretraining import create_models, train_single_run, evaluate
from .PretrainingConfig import PretrainingConfig
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm

# For statistical analysis

from .TrainingResult import TrainingResult
from .ExperimentResults import ExperimentResults
from .custom_scaler import CohortLRScheduler
from .param_utils import find_matching_hidden_dims

def run_pretraining_experiment(
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    config: Optional[PretrainingConfig] = None,
    device: str = "cuda",
) -> ExperimentResults:
    """
    Run the full pretraining comparison experiment.

    Args:
        train_dataloader: Training data
        eval_dataloader: Evaluation data
        config: Experiment configuration
        device: Device to train on

    Returns:
        ExperimentResults with statistics
    """
    if config is None:
        config = PretrainingConfig()
    assert isinstance(config, PretrainingConfig)
    logging.info("=" * 70)
    logging.info("PRETRAINING EXPERIMENT")
    logging.info("=" * 70)

    # Create output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_variants = []
    for seed in config.seeds:
        logging.info(f"\n--- Seed {seed} ---")

        # Create fresh models for each seed
        baseline_model, dendritic_model, stack_model, baseline_wave_model = (
            create_models(config)
        )
        # Create optimizers for each model
        # Optimizer with warmup
        # List of (model_name, model_instance, results_list)
        model_variants = [
            (f"{name}{opt_fn(model).__class__.__name__}", model, [], opt_fn(model))
            for name, model, _ in [
                ("baseline", baseline_model, []),
                ("baseline_wave", baseline_wave_model, []),
            ]
            for opt_fn in [
                lambda model: torch.optim.SGD(
                    model.parameters(),
                    lr=0.01,
                    momentum=0.9,
                    weight_decay=1e-4,
                    nesterov=True,
                ),
                lambda model: torch.optim.AdamW(
                    model.parameters(),
                    lr=config.learning_rate,  # type: ignore
                    weight_decay=config.weight_decay,  # type: ignore
                    betas=(0.9, 0.95),
                ),
            ]
        ]
        for model_name, model_instance, results_list, optimizer in model_variants:
            logging.info(
                f"Training {model_name} (seed={seed}) with optimizer {optimizer.__class__.__name__}..."
            )
            result = train_single_run(
                model_instance,
                train_dataloader,
                eval_dataloader,
                config,
                model_name,
                seed,
                device,
                optimizer,
            )
            results_list.append(result)
            logging.info(
                f"{model_name} (seed={seed}) final eval loss: {result.final_eval_loss:.4f}, "
                f"ppl: {result.final_perplexity:.2f}"
            )

        # Clear GPU memory
        del baseline_model, dendritic_model, stack_model, baseline_wave_model
        torch.cuda.empty_cache()

    # Statistical analysis
    model_results = {
        model_name: results_list
        for model_name, model_instance, results_list, optimizer in model_variants
    }
    statistical_analysis = analyze_results(model_results)

    # Create results object with dictionary-based storage

    results = ExperimentResults(
        model_results=model_results,
        statistical_analysis=statistical_analysis,
        config=config,
    )

    # Save results
    save_experiment_results(results, output_dir)

    # Print summary
    print_experiment_summary(results)

    return results