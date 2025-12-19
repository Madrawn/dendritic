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
from dataclasses import dataclass

from dendritic.experiments.analysis.analysis import (
    analyze_results,
    print_experiment_summary,
    save_experiment_results,
)
from dendritic.experiments.models.MiniGPT import MiniGPT

from .PretrainingConfig import PretrainingConfig, CohortSchedulerConfig

from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm

# For statistical analysis


from .TrainingResult import TrainingResult
from .ExperimentResults import ExperimentResults
from .custom_scaler import CohortLRScheduler
from .param_utils import find_matching_hidden_dims

@dataclass
class ModelVariant:
    name: str
    model: torch.nn.Module
    results: list
    optimizer: torch.optim.Optimizer
# ----------------------------------------------------------------------
# PretrainingExperiment class – encapsulates the pretraining workflow
# ----------------------------------------------------------------------
class PretrainingExperiment:
    """
    Encapsulates the pretraining experiment workflow.

    Responsibilities:
    - Store the configuration and derived hidden dimensions.
    - Build models with appropriate parameters.
    - Train a single model run.
    - Evaluate models.
    - Execute the full experiment (formerly `run_pretraining_experiment`).

    """

    def __init__(self, config: PretrainingConfig | None = None):
        """
        Initialize the experiment with a configuration.

        Args:
            config: Optional PretrainingConfig. If None, a default config is created.
        """
        self.config = config if config is not None else PretrainingConfig()
        # Hidden dimensions will be computed when models are created
        self.baseline_hidden: int | None = None
        self.dendritic_hidden: int | None = None
        self.stack_hidden: int | None = None

    def _create_cohort_scheduler(
        self,
        config: CohortSchedulerConfig,
        device: str,
    ) -> CohortLRScheduler:
        """Create a CohortLRScheduler from a CohortSchedulerConfig."""
        return CohortLRScheduler(
            min_mult=config.min_mult,
            max_mult=config.max_mult,
            device=device,
        )

    def _build_model(
        self,
        hidden_dim: int,
        mlp_type: str,
        dropout: float,
        poly_rank: int | None = None,
    ) -> MiniGPT:
        """
        Helper to construct a MiniGPT model with given parameters.
        """
        return MiniGPT(
            vocab_size=self.config.vocab_size,
            embed_dim=self.config.embed_dim,
            num_heads=self.config.num_heads,
            num_layers=self.config.num_layers,
            max_seq_len=self.config.max_seq_len,
            hidden_dim=hidden_dim,
            mlp_type=mlp_type,
            dropout=dropout,
            **({} if poly_rank is None else {"poly_rank": poly_rank}),
        )

# Remove duplicate _build_model (the class method is already defined)




    def create_models(self) -> tuple[MiniGPT, MiniGPT, MiniGPT, MiniGPT]:
        """
        Create baseline, dendritic, and stack models with matched parameters.
        Stores hidden dimensions on the instance for later use.
        """
        baseline_hidden, dendritic_hidden, stack_hidden = find_matching_hidden_dims(self.config)

        logging.info(f"Baseline hidden dim: {baseline_hidden}")
        logging.info(f"Dendritic hidden dim: {dendritic_hidden}")
        logging.info(f"Dendritic Stack hidden dim: {stack_hidden}")

        baseline_model = self._build_model(
            hidden_dim=baseline_hidden,
            mlp_type="baseline",
            dropout=self.config.dropout,
        )
        baseline_wave_model = self._build_model(
            hidden_dim=baseline_hidden,
            mlp_type="baseline_wave",
            dropout=0,
        )
        dendritic_model = self._build_model(
            hidden_dim=dendritic_hidden,
            mlp_type="dendritic",
            dropout=self.config.dropout,
            poly_rank=self.config.poly_rank,
        )
        stack_model = self._build_model(
            hidden_dim=stack_hidden,
            mlp_type="dendritic_stack",
            dropout=self.config.dropout,
            poly_rank=self.config.poly_rank,
        )

        # Verify parameter matches
        from .param_utils import verify_param_match

        matched, details = verify_param_match(
            baseline_model, dendritic_model, tolerance=0.02
        )
        logging.info(
            f"Baseline vs Dendritic: {matched} (diff: {details['relative_diff']:.2%})"
        )

        matched_stack, details_stack = verify_param_match(
            baseline_model, stack_model, tolerance=0.02
        )
        logging.info(
            f"Baseline vs DendriticStack: {matched_stack} (diff: {details_stack['relative_diff']:.2%})"
        )

        if not matched or not matched_stack:
            logging.warning("Parameters not matched within 2% tolerance!")

        # Store hidden dimensions on the instance
        self.baseline_hidden = baseline_hidden
        self.dendritic_hidden = dendritic_hidden
        self.stack_hidden = stack_hidden

        return baseline_model, dendritic_model, stack_model, baseline_wave_model


    def train_single_run(
        self,
        model: torch.nn.Module,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
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
        if self.config.scheduler_type == "cosine":
            # Linear warmup then cosine decay
            def lr_lambda(step: int) -> float:
                if step < self.config.warmup_steps:
                    return step / self.config.warmup_steps
                progress = (step - self.config.warmup_steps) / (
                    self.config.training_steps - self.config.warmup_steps
                )
                return 0.5 * (1 + np.cos(np.pi * progress))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        elif self.config.scheduler_type == "plateau":
            # ReduceLROnPlateau scheduler
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.config.plateau_factor,
                patience=self.config.plateau_patience,
                threshold=self.config.plateau_threshold,
                cooldown=self.config.plateau_cooldown,
                min_lr=self.config.plateau_min_lr,
            )
        else:
            raise ValueError(f"Unknown scheduler type: {self.config.scheduler_type}")

        # Training state
        best_eval_loss = float("inf")
        no_improvement_count = 0
        early_stop_patience = self.config.plateau_patience * self.config.early_stop_multiplier
        loss_history = []
        train_iter = iter(train_dataloader)
        total_train_loss = 0.0
        logging_step_count = 0
        start_time = time.time()
        loss = None
        progress = tqdm(range(self.config.training_steps), desc=f"{model_type} seed={seed}")
        if self.config.cohort_scheduler is not None:
            cohort_scheduler = self._create_cohort_scheduler(
                self.config.cohort_scheduler, device
            )
        else:
            cohort_scheduler = None
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
            if cohort_scheduler is not None:
                cohort_scheduler.apply_to_gradients(model)
                cohort_scheduler.step()

            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
            optimizer.step()

            # Step the scheduler based on type
            if self.config.scheduler_type == "cosine":
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
            if (step + 1) % self.config.eval_interval == 0:
                eval_loss = self.evaluate(model, eval_dataloader, self.config.eval_batches, device)
                perplexity = np.exp(eval_loss)
                avg_train_loss = total_train_loss / (logging_step_count + 1)

                # Check for improvement based on scheduler type
                if self.config.scheduler_type == "plateau":
                    improvement = best_eval_loss - eval_loss
                    if improvement > self.config.plateau_threshold or best_eval_loss == float(
                        "inf"
                    ):
                        best_eval_loss = eval_loss
                        no_improvement_count = 0
                        logging.info(
                            f"Plateau improvement: best_eval_loss updated to {best_eval_loss:.4f} "
                            f"(improvement={improvement:.4f} > threshold={self.config.plateau_threshold})"
                        )
                    else:
                        no_improvement_count += 1
                        logging.info(
                            f"Plateau no improvement: best_eval_loss remains {best_eval_loss:.4f} "
                            f"(improvement={improvement:.4f} <= threshold={self.config.plateau_threshold}), "
                            f"no_improvement_count={no_improvement_count}"
                        )
                else:
                    # cosine scheduler: any improvement counts
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                    # no_improvement_count is not used for cosine, keep it zero
                    no_improvement_count = 0

                # Step the plateau scheduler with evaluation loss
                if self.config.scheduler_type == "plateau":
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
                            if self.config.scheduler_type == "cosine"
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
                    self.config.scheduler_type == "plateau"
                    and no_improvement_count >= early_stop_patience
                ):
                    logging.info(
                        f"{model_type} seed={seed}: Early stopping triggered after "
                        f"{no_improvement_count} evaluations without improvement. "
                        f"early_stop_patience={early_stop_patience}"
                    )
                    break
        training_time = time.time() - start_time

        # Final evaluation
        final_eval_loss = self.evaluate(model, eval_dataloader, None, device)  # Full eval

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
            config=self.config.__dict__,
            polynomial_stats=polynomial_stats,
        )


    def evaluate(
        self,
        model: torch.nn.Module,
        dataloader: DataLoader,
        max_batches: int | None,
        device: str,
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

    def run(
        self,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        model_variants: list[ModelVariant],
        device: str = "cuda",
    ) -> ExperimentResults:
        """
        Execute the full pretraining experiment (formerly `run_pretraining_experiment`).
        """
        # Ensure output directory exists
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for seed in self.config.seeds:
            logging.info(f"\n--- Seed {seed} ---")

            for variant in model_variants:
                logging.info(
                    f"Training {variant.name} (seed={seed}) with optimizer {variant.optimizer.__class__.__name__}..."
                )
                result = self.train_single_run(
                    variant.model,
                    train_dataloader,
                    eval_dataloader,
                    variant.name,
                    seed,
                    device,
                    variant.optimizer,
                )
                variant.results.append(result)
                logging.info(
                    f"{variant.name} (seed={seed}) final eval loss: {result.final_eval_loss:.4f}, "
                    f"ppl: {result.final_perplexity:.2f}"
                )

            # Clear GPU memory
            torch.cuda.empty_cache()

        # Statistical analysis
        model_results = {
            variant.name: variant.results
            for variant in model_variants
        }
        statistical_analysis = analyze_results(model_results)

        # Create results object
        results = ExperimentResults(
            model_results=model_results,
            statistical_analysis=statistical_analysis,
            config=self.config,
        )

        # Save and print summary
        save_experiment_results(results, output_dir)
        print_experiment_summary(results)

        return results

