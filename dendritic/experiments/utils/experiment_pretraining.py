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
import time
from typing import cast
import torch
torch.set_float32_matmul_precision('high')

from pathlib import Path
from dataclasses import dataclass, field

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

logging.getLogger().setLevel(logging.INFO)


@dataclass
class ModelVariant:
    name: str
    model: torch.nn.Module

    optimizer: torch.optim.Optimizer
    results: list = field(default_factory=list)


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
            sharpness=config.sharpness,
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
        baseline_hidden, dendritic_hidden, stack_hidden = find_matching_hidden_dims(
            self.config
        )

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
        """Train a single model and return results (Optimized & Clean Logs)."""

        # Set seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = model.to(device)

        # Compile model (optional, safe wrap)
        try:
            model = cast(torch.nn.Module, torch.compile(model))
        except Exception:
            pass

        # Scheduler selection
        if self.config.scheduler_type == "cosine":

            def lr_lambda(step: int) -> float:
                if step < self.config.warmup_steps:
                    return step / self.config.warmup_steps
                progress = (step - self.config.warmup_steps) / (
                    self.config.training_steps - self.config.warmup_steps
                )
                return 0.5 * (1 + np.cos(np.pi * progress))

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        elif self.config.scheduler_type == "plateau":
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

        # FIX: Use new torch.amp API + ignore Pylance errors
        use_amp = device.startswith("cuda")
        # Pylance stubs are outdated for PyTorch 2.4+, so we ignore the export error
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)  # type: ignore

        # Training state
        best_eval_loss = float("inf")
        no_improvement_count = 0
        early_stop_patience = (
            self.config.plateau_patience * self.config.early_stop_multiplier
        )
        loss_history = []

        def cycle(iterable):
            iterator = iter(iterable)
            while True:
                try:
                    yield next(iterator)
                except StopIteration:
                    iterator = iter(iterable)
                    yield next(iterator)

        train_iter = cycle(train_dataloader)

        total_train_loss = 0.0
        logging_step_count = 0
        start_time = time.time()
        loss_val = 0.0

        if self.config.cohort_scheduler is not None:
            cohort_scheduler = self._create_cohort_scheduler(
                self.config.cohort_scheduler, device
            )
        else:
            cohort_scheduler = None

        progress = tqdm(
            range(self.config.training_steps), desc=f"{model_type} seed={seed}"
        )

        model.train()

        for step in progress:
            batch = next(train_iter)

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            # FIX: Use new torch.amp API with device type 'cuda'
            with torch.amp.autocast("cuda", enabled=use_amp):  # type: ignore
                outputs = model(input_ids, labels=labels)
                loss = outputs["loss"]

            optimizer.zero_grad(set_to_none=True)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            if cohort_scheduler is not None:
                cohort_scheduler.apply_to_gradients(model)
                cohort_scheduler.step()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), self.config.max_grad_norm
            )

            scaler.step(optimizer)
            scaler.update()

            # Pylance assertion to fix "metrics argument missing"
            if self.config.scheduler_type == "cosine":
                assert isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)
                scheduler.step()

            total_train_loss += loss.item()
            logging_step_count += 1

            if step % 10 == 0:
                current_lr = optimizer.param_groups[0]["lr"]
                loss_val = loss.item()
                progress.set_postfix(
                    {
                        "loss": f"{loss_val:.4f}",
                        "no_imp": no_improvement_count,
                        "lr": f"{current_lr:.6f}",
                    }
                )

            # Evaluation
            if (step + 1) % self.config.eval_interval == 0:
                logging.info(f"Evaluating at step {step+1}")

                model.eval()
                with torch.no_grad():
                    eval_loss = self.evaluate(
                        model, eval_dataloader, self.config.eval_batches, device
                    )
                model.train()

                perplexity = np.exp(eval_loss)
                avg_train_loss = (
                    total_train_loss / logging_step_count
                    if logging_step_count > 0
                    else 0.0
                )

                if self.config.scheduler_type == "plateau":
                    improvement = best_eval_loss - eval_loss
                    if (
                        improvement > self.config.plateau_threshold
                        or best_eval_loss == float("inf")
                    ):
                        best_eval_loss = eval_loss
                        no_improvement_count = 0
                        logging.info(f"Plateau improvement: {best_eval_loss:.4f}")
                    else:
                        no_improvement_count += 1
                        logging.info(
                            f"Plateau no improvement. Count: {no_improvement_count}"
                        )

                    assert isinstance(
                        scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                    )
                    scheduler.step(eval_loss)
                else:
                    if eval_loss < best_eval_loss:
                        best_eval_loss = eval_loss
                    no_improvement_count = 0

                loss_history.append(
                    {
                        "step": step + 1,
                        "train_loss": avg_train_loss,
                        "eval_loss": eval_loss,
                        "perplexity": perplexity,
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                )

                logging.info(
                    f"{model_type} seed={seed} step={step+1}: "
                    f"train={avg_train_loss:.4f}, eval={eval_loss:.4f}, "
                    f"ppl={perplexity:.2f}, lr={optimizer.param_groups[0]['lr']:.6f}"
                )

                total_train_loss = 0.0
                logging_step_count = 0

                if (
                    self.config.scheduler_type == "plateau"
                    and no_improvement_count >= early_stop_patience
                ):
                    logging.info("Early stopping triggered.")
                    break

        training_time = time.time() - start_time

        model.eval()
        with torch.no_grad():
            final_eval_loss = self.evaluate(model, eval_dataloader, None, device)

        from dendritic.enhancement import get_polynomial_stats

        polynomial_stats = get_polynomial_stats(model)

        return TrainingResult(
            model_type=model_type,
            seed=seed,
            final_train_loss=loss_val,
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
        model_results = {variant.name: variant.results for variant in model_variants}
        statistical_analysis = analyze_results(model_results)

        # Create results object
        results = ExperimentResults(
            model_results=model_results,
            statistical_analysis=statistical_analysis,
            config=self.config,
        )

        # Save and print summary moved to caller
        # save_experiment_results(results, output_dir)
        # print_experiment_summary(results)

        return results
