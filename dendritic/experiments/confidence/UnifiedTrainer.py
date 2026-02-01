"""
Unified training infrastructure for confidence-aware experiments.

This module provides a unified training framework that combines shared logic
from confidence and standard model training, while keeping model-specific
differences in separate strategy classes.
"""

from dendritic.experiments.utils.TrainingResult import TrainingResult
from dendritic.experiments.utils.experiment_utils import set_random_seed
from dendritic.experiments.confidence.TrainingStrategy import TrainingStrategy
from dendritic.experiments.confidence.config import ConfidenceExperimentConfig
from dendritic.experiments.confidence.results import ConfidenceTrainingResult


import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


import gc
import logging
import os
import time
from collections.abc import Generator
import os
import subprocess


import torch


class UnifiedTrainer:
    """Unified trainer that works with any training strategy."""

    def __init__(
        self,
        config: ConfidenceExperimentConfig,
        strategy: TrainingStrategy,
        model_type: str,
    ):
        """
        Initialize unified trainer.

        Args:
            config: Experiment configuration
            strategy: Training strategy to use
            model_type: Type of model (for logging)
        """
        self.config = config
        self.strategy = strategy
        self.model_type = model_type
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # CUDA performance optimizations (ported from pretraining experiment)
        self.use_cuda = torch.cuda.is_available() and self.device.startswith("cuda")
        if self.use_cuda:
            torch.set_float32_matmul_precision("medium")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

    def prefetching_cycle(self, dataloader, device: str) -> Generator:
        """
        Robust prefetching iterator (ported from pretraining experiment).

        Handles Windows iterator reset internally and prefetches to device.
        """
        while True:
            try:
                for batch in dataloader:
                    # Prepare batch using strategy
                    prepared_batch = self.strategy.prepare_batch(batch, device)
                    yield prepared_batch
            except StopIteration:
                # Windows workaround: reset iterator
                continue
            except RuntimeError as e:
                if "DataLoader worker" in str(e):
                    # Windows-specific issue with worker processes
                    logging.warning(
                        f"DataLoader worker issue: {e}. Restarting iterator."
                    )
                    continue
                raise

    def _evaluate_with_iterator(
        self,
        model: nn.Module,
        dataloader: Generator,
        max_batches: int | None,
        device: str,
    ) -> float:
        """
        Evaluate model using prefetching iterator (ported from pretraining).

        Returns mean loss.
        """
        model.eval()
        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            i = 0
            for batch in tqdm(
                dataloader, total=max_batches, desc="Evaluating", leave=False
            ):
                if max_batches and i >= max_batches:
                    break
                i += 1

                # Evaluation step using strategy
                step_result = self.strategy.evaluation_step(model, batch, device)

                # Extract loss (handle different loss keys)
                loss = step_result.get("loss_lm", step_result.get("loss"))
                if loss is None:
                    raise ValueError("Evaluation step must return a loss")

                # For language modeling loss, we count tokens
                # This is simplified - pretraining has more complex token counting
                total_loss += loss.item()
                total_tokens += 1  # Simplified - pretraining counts actual tokens

        return total_loss / total_tokens if total_tokens > 0 else float("nan")

    def _evaluate(
        self, model: nn.Module, eval_loader, device: str, num_batches: int
    ) -> float:
        """
        Evaluate model on validation set.

        Args:
            model: Model to evaluate
            eval_loader: Evaluation DataLoader
            device: Device to run on
            num_batches: Number of batches to evaluate

        Returns:
            Average evaluation loss
        """
        model.eval()
        total_loss = 0.0
        total_batches = 0

        with torch.no_grad():
            eval_iter = iter(eval_loader)
            for i in range(num_batches):
                try:
                    batch = next(eval_iter)
                except StopIteration:
                    eval_iter = iter(eval_loader)
                    batch = next(eval_iter)

                # Prepare batch using strategy
                prepared_batch = self.strategy.prepare_batch(batch, device)

                # Evaluation step using strategy
                step_result = self.strategy.evaluation_step(
                    model, prepared_batch, device
                )

                total_loss += step_result["loss"].item()
                total_batches += 1

        model.train()
        return total_loss / total_batches if total_batches > 0 else float("inf")

    def train(
        self, model: nn.Module, train_loader, eval_loader, seed: int
    ) -> TrainingResult | ConfidenceTrainingResult:
        """
        Unified training loop.

        Args:
            model: Model to train
            train_loader: Training DataLoader
            eval_loader: Evaluation DataLoader
            seed: Random seed

        Returns:
            Training result appropriate for the strategy
        """
        # Set seed
        set_random_seed(seed)

        # ========== 1. CUDA OPTIMIZATIONS ==========
        if self.use_cuda:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Move model to device
        model = model.to(self.device)

        # ========== 2. MODEL COMPILATION (optional) ==========
        # Windows -> default, Linux -> reduce-overhead
        if self.device.startswith("cuda"):
            compile_mode = "default" if os.name == "nt" else "reduce-overhead"
            try:
                logging.info(f"Compiling model with mode='{compile_mode}'...")
                model = torch.compile(model, mode=compile_mode)  # type: ignore
            except Exception as e:
                logging.warning(f"Compilation failed: {e}. Falling back to eager.")
                if hasattr(model, "_orig_mod"):
                    model = model._orig_mod  # type: ignore

        # ========== 3. OPTIMIZER ==========
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        # ========== 4. SCHEDULER (ported from pretraining) ==========
        warmup_steps = getattr(self.config, "warmup_steps", 0)
        training_steps = int(self.config.training_steps)

        if self.config.scheduler_type == "cosine":

            def lr_lambda(step: int) -> float:
                if step < warmup_steps:
                    return float(step) / float(max(1, warmup_steps))
                progress = float(step - warmup_steps) / float(
                    max(1, training_steps - warmup_steps)
                )
                return 0.5 * (1.0 + np.cos(np.pi * progress))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        elif self.config.scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=getattr(self.config, "plateau_factor", 0.5),
                patience=getattr(self.config, "plateau_patience", 5),
                threshold=getattr(self.config, "plateau_threshold", 1e-4),
            )
        else:
            scheduler = None

        # ========== 5. AMP & GRADSCALER ==========
        # Enable scaler for safety (prevents PPL explosion)
        bf16_supported = self.use_cuda and torch.cuda.is_bf16_supported()
        amp_dtype = torch.bfloat16 if bf16_supported else torch.float16
        scaler = torch.amp.GradScaler("cuda", enabled=self.use_cuda)  # type: ignore

        # ========== 6. TRAINING STATE ==========
        best_eval_loss = float("inf")
        loss_history = []
        strategy_metrics = {
            "confidence_loss_history": [],
            "token_loss_history": [],
            "confidence_predictions": [],
            "actual_future_losses": [],
        }

        # ========== 7. PREFETCHING ITERATOR ==========
        train_iter = self.prefetching_cycle(train_loader, self.device)
        eval_iter = self.prefetching_cycle(eval_loader, self.device)

        # ========== 8. TRAINING LOOP SETUP ==========
        model.train()
        progress = tqdm(
            range(training_steps),
            desc=f"{self.model_type} model seed={seed}",
        )
        start_time = time.time()

        # Initialize loss tracking (ported from pretraining)
        queued_loss = None
        avg_train_loss_tensor = torch.tensor(15.0, device="cpu")
        # Separate tracking for lm_loss and confidence_loss
        avg_train_lm_loss_tensor = torch.tensor(15.0, device="cpu")
        avg_train_conf_loss_tensor = torch.tensor(0.0, device="cpu")
        avg_eval_loss = 15.0
        no_improvement_count = 0
        loss = None

        for step in progress:
            # Robust fetch (handles Windows iterator reset internally)
            batch = next(train_iter)

            # Forward with AMP
            with torch.amp.autocast("cuda", enabled=self.use_cuda, dtype=amp_dtype):  # type: ignore
                step_result = self.strategy.training_step(model, batch, self.device)
                loss = step_result.get("total_loss", step_result.get("loss"))
                if loss is None:
                    raise ValueError("Training step must return a loss")

            # Backward with GradScaler
            optimizer.zero_grad(set_to_none=True)  # Efficient gradient zeroing
            scaler.scale(loss).backward()

            # --- Unscale BEFORE manipulating gradients ---
            scaler.unscale_(optimizer)

            # Gradient clipping (after unscaling)
            clip_grad_norm = getattr(self.config, "max_grad_norm", 1.0)
            if clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

            # Optimizer Step (Scaler checks for NaNs here)
            scaler.step(optimizer)
            scaler.update()

            # Scheduler Step
            if scheduler:
                if self.config.scheduler_type == "cosine":
                    scheduler.step()
                elif self.config.scheduler_type == "plateau":
                    # Plateau scheduler needs metrics, will be called during evaluation
                    pass

            # Track strategy-specific metrics
            if "loss_confidence" in step_result:
                strategy_metrics["confidence_loss_history"].append(
                    step_result["loss_confidence"].item()
                )
            if "loss_lm" in step_result:
                strategy_metrics["token_loss_history"].append(
                    step_result["loss_lm"].item()
                )
            if "pred_conf_t" in step_result:
                strategy_metrics["confidence_predictions"].append(
                    step_result["pred_conf_t"].mean().item()
                )

            # Accumulate Loss (GPU resident) - ported from pretraining
            avg_train_loss_tensor = (
                avg_train_loss_tensor * 0.9 + 0.1 * loss.detach().cpu()
            )

            # Accumulate separate losses if available
            if "loss_lm" in step_result:
                avg_train_lm_loss_tensor = (
                    avg_train_lm_loss_tensor * 0.9
                    + 0.1 * step_result["loss_lm"].detach().cpu()
                )
            if "loss_confidence" in step_result:
                avg_train_conf_loss_tensor = (
                    avg_train_conf_loss_tensor * 0.9
                    + 0.1 * step_result["loss_confidence"].detach().cpu()
                )

            # Update Progress Bar (Periodic Sync) - ported from pretraining
            if step % 10 == 0:
                # Use lm_loss for display if available, otherwise total loss
                if "loss_lm" in step_result:
                    current_loss_tensor = step_result["loss_lm"].detach().cpu()
                else:
                    current_loss_tensor = loss.detach().cpu()

                if queued_loss is not None:
                    # This .item() will be instant
                    progress.set_postfix(
                        {
                            "loss": f"{queued_loss.item():.4f}",
                            "lr": f"{optimizer.param_groups[0]['lr']:.6f}",
                        }
                    )

                # Update the queue
                queued_loss = current_loss_tensor

            # Evaluation Loop (ported from pretraining)
            eval_interval = getattr(self.config, "eval_interval", 100)
            if (step + 1) % eval_interval == 0:
                avg_train_loss = avg_train_loss_tensor.item()

                model.eval()
                with torch.no_grad():
                    eval_loss = self._evaluate_with_iterator(
                        model,
                        eval_iter,
                        getattr(self.config, "eval_batches", 10),
                        self.device,
                    )
                model.train()

                avg_eval_loss = avg_eval_loss * 0.9 + 0.1 * eval_loss
                perplexity = np.exp(eval_loss)

                if avg_eval_loss < best_eval_loss:
                    best_eval_loss = avg_eval_loss
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                # Plateau scheduler step
                if self.config.scheduler_type == "plateau" and scheduler is not None:
                    scheduler.step(
                        avg_eval_loss
                    )  # ReduceLROnPlateau expects metrics parameter

                # Record loss history
                history_entry = {
                    "step": step + 1,
                    "train_loss": avg_train_loss,
                    "eval_loss": eval_loss,
                    "perplexity": perplexity,
                    "lr": optimizer.param_groups[0]["lr"],
                }

                # Add strategy-specific metrics if available
                if "loss_lm" in step_result:
                    history_entry["train_loss_lm"] = step_result["loss_lm"].item()
                if "loss_confidence" in step_result:
                    history_entry["train_loss_conf"] = step_result[
                        "loss_confidence"
                    ].item()

                loss_history.append(history_entry)

                # Format logging based on model type
                if "loss_confidence" in step_result:
                    # Confidence model: show separate losses
                    avg_train_lm_loss = avg_train_lm_loss_tensor.item()
                    avg_train_conf_loss = avg_train_conf_loss_tensor.item()
                    logging.info(
                        f"{self.model_type} seed={seed} step={step+1}: "
                        f"train_lm={avg_train_lm_loss:.4f}, train_conf={avg_train_conf_loss:.4f}, "
                        f"avg_eval_loss={avg_eval_loss:.4f}, ppl={perplexity:.2f}"
                    )
                else:
                    # Standard model: keep original format
                    logging.info(
                        f"{self.model_type} seed={seed} step={step+1}: "
                        f"train={avg_train_loss:.4f}, avg_eval_loss={avg_eval_loss:.4f}, ppl={perplexity:.2f}"
                    )

                # Early stopping check (ported from pretraining)
                plateau_patience = getattr(self.config, "plateau_patience", 5)
                early_stop_multiplier = getattr(self.config, "early_stop_multiplier", 3)
                if (
                    self.config.scheduler_type == "plateau"
                    and no_improvement_count >= plateau_patience * early_stop_multiplier
                ):
                    logging.info("Early stopping triggered.")
                    break

        training_time = time.time() - start_time
        final_train_loss = loss.item() if loss is not None else float("inf")

        # Final Eval (ported from pretraining)
        model.eval()
        with torch.no_grad():
            final_eval_loss = self._evaluate_with_iterator(
                model, eval_iter, self.config.eval_batches or 10, self.device
            )

        # Memory cleanup (ported from pretraining)
        eval_iter.close()
        train_iter.close()
        del (
            eval_iter,
            train_iter,
            eval_loader,
            train_loader,
            model,
            optimizer,
            scheduler,
        )
        gc.collect()
        if self.use_cuda:
            torch.cuda.empty_cache()

        # Create result using strategy
        return self.strategy.create_result(
            model_type=self.model_type,
            seed=seed,
            final_train_loss=final_train_loss,
            final_eval_loss=final_eval_loss,
            best_eval_loss=best_eval_loss,
            loss_history=loss_history,
            training_time=training_time,
            additional_metrics=strategy_metrics,
        )
