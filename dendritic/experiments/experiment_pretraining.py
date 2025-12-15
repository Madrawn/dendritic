# dendritic/experiments/experiment_pretraining.py
"""
Experiment 1: Pretraining Comparison
====================================
Compare fresh training of:
- Baseline GPT-2 with standard MLP
- Dendritic GPT-2 with DendriticMLP (parameter-matched)

Key controls:
- Same total parameter count
- Same training data
- Same hyperparameters (except architecture-specific)
- Multiple random seeds for statistical significance
"""

import logging
import torch
from pathlib import Path
from typing import Optional, Tuple

from .MiniGPT import MiniGPT
from .PretrainingConfig import PretrainingConfig
def _build_model(
    config: PretrainingConfig,
    hidden_dim: int,
    mlp_type: str,
    dropout: float,
    poly_rank: Optional[int] = None,
) -> MiniGPT:
    """Helper to construct a MiniGPT model with given parameters."""
    return MiniGPT(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        max_seq_len=config.max_seq_len,
        hidden_dim=hidden_dim,
        mlp_type=mlp_type,
        dropout=dropout,
        **({} if poly_rank is None else {"poly_rank": poly_rank}),
    )
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm

# For statistical analysis

from .analysis import (
    analyze_results,
    print_experiment_summary,
    save_experiment_results,
)
from .TrainingResult import TrainingResult
from .ExperimentResults import ExperimentResults
from .custom_scaler import CohortLRScheduler
from .param_utils import find_matching_hidden_dims


def create_models(
    config: PretrainingConfig,
) -> Tuple[MiniGPT, MiniGPT, MiniGPT, MiniGPT]:
    """Create baseline, dendritic, and stack models with matched parameters."""
    baseline_hidden, dendritic_hidden, stack_hidden = find_matching_hidden_dims(config)

    logging.info(f"Baseline hidden dim: {baseline_hidden}")
    logging.info(f"Dendritic hidden dim: {dendritic_hidden}")
    logging.info(f"Dendritic Stack hidden dim: {stack_hidden}")

    baseline_model = _build_model(
        config,
        hidden_dim=baseline_hidden,
        mlp_type="baseline",
        dropout=config.dropout,
    )
    baseline_wave_model = _build_model(
        config,
        hidden_dim=baseline_hidden,
        mlp_type="baseline_wave",
        dropout=0,
    )
    dendritic_model = _build_model(
        config,
        hidden_dim=dendritic_hidden,
        mlp_type="dendritic",
        dropout=config.dropout,
        poly_rank=config.poly_rank,
    )
    stack_model = _build_model(
        config,
        hidden_dim=stack_hidden,
        mlp_type="dendritic_stack",
        dropout=config.dropout,
        poly_rank=config.poly_rank,
    )

    # Verify parameter matches
    from .param_utils import verify_param_match

    # Verify baseline vs dendritic
    matched, details = verify_param_match(
        baseline_model, dendritic_model, tolerance=0.02
    )
    logging.info(
        f"Baseline vs Dendritic: {matched} (diff: {details['relative_diff']:.2%})"
    )

    # Verify baseline vs stack
    matched_stack, details_stack = verify_param_match(
        baseline_model, stack_model, tolerance=0.02
    )
    logging.info(
        f"Baseline vs DendriticStack: {matched_stack} (diff: {details_stack['relative_diff']:.2%})"
    )

    if not matched or not matched_stack:
        logging.warning("Parameters not matched within 2% tolerance!")

    # Store computed hidden dims in config
    config.baseline_hidden_dim = baseline_hidden
    config.dendritic_hidden_dim = dendritic_hidden
    config.dendritic_stack_hidden_dim = stack_hidden

    return baseline_model, dendritic_model, stack_model, baseline_wave_model


def train_single_run(
    model: MiniGPT,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    config: PretrainingConfig,
    model_type: str,
    seed: int,
    device: str,
    optimizer: torch.optim.Optimizer,
) -> TrainingResult:
    """Train a single model and return results."""
    import time

    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model.to(device)

    # Scheduler selection
    if config.scheduler_type == "cosine":
        # Linear warmup then cosine decay
        def lr_lambda(step: int) -> float:
            if step < config.warmup_steps:
                return step / config.warmup_steps
            progress = (step - config.warmup_steps) / (
                config.training_steps - config.warmup_steps
            )
            return 0.5 * (1 + np.cos(np.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif config.scheduler_type == "plateau":
        # ReduceLROnPlateau scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=config.plateau_factor,
            patience=config.plateau_patience,
            threshold=config.plateau_threshold,
            cooldown=config.plateau_cooldown,
            min_lr=config.plateau_min_lr,
        )
    else:
        raise ValueError(f"Unknown scheduler type: {config.scheduler_type}")

    # Training state
    best_eval_loss = float("inf")
    no_improvement_count = 0
    early_stop_patience = config.plateau_patience * config.early_stop_multiplier
    loss_history = []
    train_iter = iter(train_dataloader)
    total_train_loss = 0.0
    logging_step_count = 0
    start_time = time.time()
    loss = None
    progress = tqdm(range(config.training_steps), desc=f"{model_type} seed={seed}")
    cohort_scheduler = CohortLRScheduler(min_mult=0.5, max_mult=1.0, device=device)
    for step in progress:
        model.train()

        # Get batch
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dataloader)
            batch = next(train_iter)

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        outputs = model(input_ids, labels=labels)
        loss = outputs["loss"]

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        if model_type == "baseline_wave":
            cohort_scheduler.apply_to_gradients(model)
            cohort_scheduler.step()

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()

        # Step the scheduler based on type
        if config.scheduler_type == "cosine":
            assert isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)
            scheduler.step()
        # For plateau scheduler, we step with eval_loss during evaluation

        # Update progress
        progress.set_postfix(
            {
                "loss": f"{loss.item():.4f}, no_improvement_count: {no_improvement_count}, lr: {optimizer.param_groups[0]['lr']:.6f}" 
            }
        )

        # Evaluation
        # Evaluation
        if (step + 1) % config.eval_interval == 0:
            eval_loss = evaluate(model, eval_dataloader, config.eval_batches, device)
            perplexity = np.exp(eval_loss)
            avg_train_loss = total_train_loss / (logging_step_count + 1)

            # Check for improvement based on scheduler type
            if config.scheduler_type == "plateau":
                improvement = best_eval_loss - eval_loss
                if improvement > config.plateau_threshold or best_eval_loss == float(
                    "inf"
                ):
                    best_eval_loss = eval_loss
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
            else:
                # cosine scheduler: any improvement counts
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                # no_improvement_count is not used for cosine, keep it zero
                no_improvement_count = 0

            # Step the plateau scheduler with evaluation loss
            if config.scheduler_type == "plateau":
                assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
                scheduler.step(eval_loss)

            loss_history.append(
                {
                    "step": step + 1,
                    "train_loss": avg_train_loss,
                    "eval_loss": eval_loss,
                    "perplexity": perplexity,
                    "lr": (
                        scheduler.get_last_lr()[0]
                        if config.scheduler_type == "cosine"
                        else optimizer.param_groups[0]["lr"]
                    ),
                }
            )

            logging.info(
                f"{model_type} seed={seed} step={step+1}: "
                f"train_loss={avg_train_loss:.4f}, eval_loss={eval_loss:.4f}, "
                f"ppl={perplexity:.2f} lr={loss_history[-1]['lr']:.6f}"
            )
            # Reset the running loss and step count for the next interval
            total_train_loss = 0.0
            logging_step_count = 0

            # Early stopping check (only for plateau scheduler)
            if (
                config.scheduler_type == "plateau"
                and no_improvement_count >= early_stop_patience
            ):
                logging.info(
                    f"{model_type} seed={seed}: Early stopping triggered after "
                    f"{no_improvement_count} evaluations without improvement."
                )
                break
    training_time = time.time() - start_time

    # Final evaluation
    final_eval_loss = evaluate(model, eval_dataloader, None, device)  # Full eval

    # Get polynomial stats for dendritic layers
    from dendritic.enhancement import get_polynomial_stats

    polynomial_stats = get_polynomial_stats(model)

    # Use last loss if available, else use final_eval_loss
    final_train_loss = loss.item() if loss is not None else final_eval_loss

    return TrainingResult(
        model_type=model_type,
        seed=seed,
        final_train_loss=final_train_loss,
        final_eval_loss=final_eval_loss,
        final_perplexity=float(np.exp(final_eval_loss)),
        best_eval_loss=best_eval_loss,
        best_perplexity=float(np.exp(best_eval_loss)),
        loss_history=loss_history,
        training_time=training_time,
        config=config.__dict__,
        polynomial_stats=polynomial_stats,
    )


def evaluate(
    model: MiniGPT, dataloader: DataLoader, max_batches: Optional[int], device: str
) -> float:
    """Evaluate model and return mean loss."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if max_batches and i >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, labels=labels)

            # Count non-masked tokens
            non_masked = (labels != -100).sum().item()
            total_loss += outputs["loss"].item() * non_masked
            total_tokens += non_masked

    return total_loss / total_tokens if total_tokens > 0 else float("nan")


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
