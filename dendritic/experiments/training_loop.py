import time
import logging

import torch
import numpy as np
from tqdm import tqdm

from .MiniGPT import MiniGPT
from .PretrainingConfig import PretrainingConfig
from torch.utils.data import DataLoader
from .TrainingResult import TrainingResult
from .custom_scaler import CohortLRScheduler
from .experiment_pretraining import evaluate  # Reuse the evaluate helper from the original module


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

        # Update progress display
        progress.set_postfix(
            {
                "loss": f"{loss.item():.4f}, no_improvement_count: {no_improvement_count}, lr: {optimizer.param_groups[0]['lr']:.6f}"
            }
        )

        # Evaluation
        if (step + 1) % config.eval_interval == 0:
            eval_loss = evaluate(model, eval_dataloader, config.eval_batches, device)
            perplexity = np.exp(eval_loss)
            avg_train_loss = total_train_loss / (logging_step_count + 1)

            # Check for improvement based on scheduler type
            if config.scheduler_type == "plateau":
                improvement = best_eval_loss - eval_loss
                if improvement > config.plateau_threshold or best_eval_loss == float("inf"):
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