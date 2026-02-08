"""
Unified training infrastructure for confidence-aware experiments.

This module provides a unified training framework that combines shared logic
from confidence and standard model training, while keeping model-specific
differences in separate strategy classes.
"""

from typing import cast
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
from collections.abc import Generator, Iterable
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
        tokenizer=None,  # Optional tokenizer for sampling
        results_dir=None,  # Directory for saving sampled tokens
    ):
        """
        Initialize unified trainer.

        Args:
            config: Experiment configuration
            strategy: Training strategy to use
            model_type: Type of model (for logging)
            tokenizer: Optional tokenizer for sampling during evaluation
            results_dir: Directory for saving sampled tokens
        """
        self.config = config
        self.strategy = strategy
        self.model_type = model_type
        self.tokenizer = tokenizer
        self.results_dir = results_dir or config.results_dir
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
                    logging.warning(f"DataLoader worker issue: {e}. Restarting iterator.")
                    continue
                raise

    def _evaluate_with_iterator(
        self,
        model: nn.Module,
        dataloader: Iterable,
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
            for batch in tqdm(dataloader, total=max_batches, desc="Evaluating", leave=False):
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

    def _evaluate(self, model: nn.Module, eval_loader: Iterable, device: str, num_batches: int) -> float:
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
                step_result = self.strategy.evaluation_step(model, prepared_batch, device)

                total_loss += step_result["loss"].item()
                total_batches += 1

        model.train()
        return total_loss / total_batches if total_batches > 0 else float("inf")

    def _sample_and_log(self, model: nn.Module, step: int, eval_loss: float, seed: int):
        """
        Sample tokens from model and log results.

        Args:
            model: Model to sample from
            step: Current training step
            eval_loss: Current evaluation loss
            seed: Random seed for reproducibility
        """
        if self.tokenizer is None:
            logging.warning("No tokenizer provided, skipping sampling")
            return

        try:
            from dendritic.experiments.confidence.sampling_utils import (
                sample_model_output,
                SamplingConfig,
            )

            # Determine if we should use confidence-aware sampling
            # Only for ConfidenceAwareGPT models
            use_confidence = hasattr(model, "confidence_predictor")

            sampling_config = SamplingConfig(
                device=self.device,
                max_new_tokens=self.config.sampling_max_tokens,
                temperature=self.config.sampling_temperature,
                top_p=self.config.sampling_top_p,
                use_confidence=use_confidence,
                include_confidence_formatting=True,
            )
            generated, confidence_predictions, formatted_tokens_with_confidence = sample_model_output(
                model=model,
                tokenizer=self.tokenizer,
                prompt=self.config.sampling_prompt,
                config=sampling_config,
            )

            # Log truncated version to console
            truncated = generated[:100] + "..." if len(generated) > 100 else generated
            log_message = f"{self.model_type} seed={seed} step={step}: eval_loss={eval_loss:.4f}, sampled: {truncated}"

            # Add confidence predictions to log if available
            if confidence_predictions is not None and len(confidence_predictions) > 0:
                avg_conf = sum(confidence_predictions) / len(confidence_predictions)
                log_message += f", avg_conf={avg_conf:.4f}"

            logging.info(log_message)

            # Save full version to text file
            self._save_sampled_tokens_to_file(
                step=step,
                seed=seed,
                model_type=self.model_type,
                eval_loss=eval_loss,
                sampled_text=generated,
                confidence_predictions=confidence_predictions,
                formatted_tokens_with_confidence=formatted_tokens_with_confidence,
            )

        except Exception as e:
            logging.warning(f"Sampling failed: {e}")

    def _save_sampled_tokens_to_file(
        self,
        step,
        seed,
        model_type,
        eval_loss,
        sampled_text,
        confidence_predictions=None,
        formatted_tokens_with_confidence=None,
    ):
        """
        Save sampled tokens to text file alongside JSON results.

        Args:
            step: Training step
            seed: Random seed
            model_type: Type of model
            eval_loss: Evaluation loss
            sampled_text: Full sampled text
            confidence_predictions: Optional list of confidence predictions
        """
        from datetime import datetime
        from pathlib import Path

        # Ensure results directory exists
        results_path = Path(self.results_dir)
        results_path.mkdir(parents=True, exist_ok=True)

        # Create text file name based on timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        text_file = results_path / f"{timestamp}_{model_type}_seed{seed}_sampled_tokens.txt"

        # Append to file
        with open(text_file, "a", encoding="utf-8") as f:
            f.write(f"=== Step {step} (Seed: {seed}, Model: {model_type}) ===\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Eval Loss: {eval_loss:.4f}\n")
            f.write(f"Sampled Tokens: {sampled_text}\n")

            # Add confidence predictions if available
            if confidence_predictions is not None and len(confidence_predictions) > 0:
                f.write(f"Confidence Predictions: {confidence_predictions}\n")
                f.write(f"Avg Confidence: {sum(confidence_predictions) / len(confidence_predictions):.4f}\n")

                # Add formatted tokens with confidence if available
                if formatted_tokens_with_confidence is not None:
                    f.write(f"Sampled Tokens with Confidence: {formatted_tokens_with_confidence}\n")

            f.write("=" * 40 + "\n\n")

    def train(
        self, model: nn.Module, train_loader: Iterable, eval_loader: Iterable, seed: int
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
        # torch.cuda.memory._record_memory_history(max_entries=100000)
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
        optimizer = torch.optim.Adafactor(
            model.parameters(),
            # lr=self.config.learning_rate,
            # weight_decay=self.config.weight_decay,
        )

        # ========== 4. SCHEDULER (ported from pretraining) ==========
        warmup_steps = getattr(self.config, "warmup_steps", 0)
        training_steps = int(self.config.training_steps)

        if self.config.scheduler_type == "cosine":

            def lr_lambda(step: int) -> float:
                if step < warmup_steps:
                    return float(step) / float(max(1, warmup_steps))
                progress = float(step - warmup_steps) / float(max(1, training_steps - warmup_steps))
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

        # Track minimal improvement early exit (using smoothed avg_eval_loss)
        last_avg_eval_loss = None

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
                    scheduler = cast(torch.optim.lr_scheduler.LambdaLR, scheduler)
                    scheduler.step()
                elif self.config.scheduler_type == "plateau":
                    # Plateau scheduler needs metrics, will be called during evaluation
                    pass

            # Track strategy-specific metrics
            if "loss_confidence" in step_result:
                strategy_metrics["confidence_loss_history"].append(step_result["loss_confidence"].item())
            if "loss_lm" in step_result:
                strategy_metrics["token_loss_history"].append(step_result["loss_lm"].item())
            if "pred_conf_t" in step_result:
                strategy_metrics["confidence_predictions"].append(step_result["pred_conf_t"].mean().item())

            # Accumulate Loss (GPU resident) - ported from pretraining
            avg_train_loss_tensor = avg_train_loss_tensor * 0.9 + 0.1 * loss.detach().cpu()

            # Accumulate separate losses if available
            if "loss_lm" in step_result:
                avg_train_lm_loss_tensor = avg_train_lm_loss_tensor * 0.9 + 0.1 * step_result["loss_lm"].detach().cpu()
            if "loss_confidence" in step_result:
                avg_train_conf_loss_tensor = (
                    avg_train_conf_loss_tensor * 0.9 + 0.1 * step_result["loss_confidence"].detach().cpu()
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
                    progress.set_postfix({
                        "loss": f"{queued_loss.item():.4f}",
                        "lr": f"{optimizer.param_groups[0]['lr']:.6f}",
                    })

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
                # torch.cuda.memory._dump_snapshot("snappershop.pickle")
                # torch.cuda.memory._record_memory_history(enabled=None)

                model.train()

                # Sample tokens from model during evaluation
                self._sample_and_log(model, step + 1, eval_loss, seed)

                smoothing_factor = getattr(self.config, "eval_smoothing_factor", 0.5)
                avg_eval_loss = avg_eval_loss * (1 - smoothing_factor) + smoothing_factor * eval_loss
                perplexity = np.exp(eval_loss)

                if avg_eval_loss < best_eval_loss:
                    best_eval_loss = avg_eval_loss
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                # Plateau scheduler step
                if self.config.scheduler_type == "plateau" and scheduler is not None:
                    scheduler = cast(torch.optim.lr_scheduler.ReduceLROnPlateau, scheduler)
                    scheduler.step(avg_eval_loss)  # ReduceLROnPlateau expects metrics parameter

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
                    history_entry["train_loss_conf"] = step_result["loss_confidence"].item()

                loss_history.append(history_entry)

                # Format logging based on model type
                if "loss_confidence" in step_result:
                    # Confidence model: show separate losses
                    avg_train_lm_loss = avg_train_lm_loss_tensor.item()
                    avg_train_conf_loss = avg_train_conf_loss_tensor.item()
                    logging.info(
                        f"{self.model_type} seed={seed} step={step + 1}: "
                        f"train_lm={avg_train_lm_loss:.4f}, train_conf={avg_train_conf_loss:.4f}, "
                        f"avg_eval_loss={avg_eval_loss:.4f}, ppl={perplexity:.2f}"
                    )
                else:
                    # Standard model: keep original format
                    logging.info(
                        f"{self.model_type} seed={seed} step={step + 1}: "
                        f"train={avg_train_loss:.4f}, avg_eval_loss={avg_eval_loss:.4f}, ppl={perplexity:.2f}"
                    )

                # Minimal improvement early exit check (using smoothed avg_eval_loss)
                min_improvement = getattr(self.config, "min_eval_improvement", None)
                if min_improvement is not None and last_avg_eval_loss is not None:
                    improvement = last_avg_eval_loss - avg_eval_loss
                    if improvement < min_improvement:
                        logging.info(
                            f"Early stopping triggered: smoothed eval loss improvement "
                            f"({improvement:.6f}) below threshold ({min_improvement})"
                        )
                        break
                last_avg_eval_loss = avg_eval_loss

        training_time = time.time() - start_time
        final_train_loss = loss.item() if loss is not None else float("inf")

        # Final Eval (ported from pretraining)
        model.eval()
        with torch.no_grad():
            final_eval_loss = self._evaluate_with_iterator(
                model, eval_iter, self.config.eval_batches or 10, self.device
            )

        # Sample tokens from model after final evaluation
        self._sample_and_log(model, training_steps, final_eval_loss, seed)

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
