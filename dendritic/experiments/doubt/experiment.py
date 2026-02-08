# ruff: noqa: PLR6301
"""
Doubt-aware GPT experiment implementation.

This module implements the DoubtAwareExperiment class for comparing
DoubtAwareGPT (with two-pass lookahead training) vs standard MiniGPT.
"""

import logging
from datetime import datetime
from typing import Dict, List, Tuple
import torch
import numpy as np
from pathlib import Path

from dendritic.experiments.models.DoubtAwareGPT import DoubtAwareGPT
from dendritic.experiments.models.MiniGPT import MiniGPT
from dendritic.experiments.models.ModelConfig import ModelConfig
from dendritic.experiments.utils.TrainingResult import TrainingResult
from dendritic.experiments.utils.param_utils import find_matching_hidden_dims
from dendritic.experiments.utils.loss_utils import (
    compute_sequence_language_modeling_loss,
)

from dendritic.experiments.doubt.config import DoubtExperimentConfig
from dendritic.experiments.doubt.data_loader import prepare_doubt_data
from dendritic.experiments.doubt.results import (
    DoubtTrainingResult,
    DoubtExperimentResults,
    save_results,
)
from dendritic.experiments.doubt.UnifiedTrainer import (
    UnifiedTrainer,
)
from dendritic.experiments.doubt.DoubtTrainingStrategy import (
    DoubtTrainingStrategy,
)
from dendritic.experiments.doubt.StandardTrainingStrategy import (
    StandardTrainingStrategy,
)


class DoubtAwareExperiment:
    """
    Experiment comparing DoubtAwareGPT vs standard MiniGPT.

    This class implements the two-pass lookahead training for DoubtAwareGPT
    and standard training for MiniGPT, with proper parameter logging and
    results tracking.
    """

    def __init__(self, config: DoubtExperimentConfig):
        """
        Initialize the experiment with configuration.

        Args:
            config: DoubtExperimentConfig containing experiment parameters
        """
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Create output directory
        self.results_dir = Path(config.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(self.results_dir / "experiment.log"),
                logging.StreamHandler(),
            ],
        )

    def create_models(self) -> Tuple[MiniGPT, DoubtAwareGPT]:
        """
        Create both model variants with parameter matching.

        Returns:
            Tuple of (standard_model, doubt_model)
        """
        # Get matching hidden dimensions
        baseline_hidden, dendritic_hidden = find_matching_hidden_dims(self.config)

        # Create standard MiniGPT with ModelConfig
        standard_config = ModelConfig(
            vocab_size=self.config.vocab_size,
            embed_dim=self.config.embed_dim,
            num_heads=self.config.num_heads,
            num_layers=self.config.num_layers,
            max_seq_len=self.config.max_seq_len,
            hidden_dim=baseline_hidden,
            mlp_type="standard",
            dropout=self.config.dropout,
            poly_rank=16,
            poly_degree=3,
        )
        standard_model = MiniGPT(standard_config)

        # Create DoubtAwareGPT with ModelConfig
        # Doubt model uses same sequence length as standard model
        doubt_config = ModelConfig(
            vocab_size=self.config.vocab_size,
            embed_dim=self.config.embed_dim,
            num_heads=self.config.num_heads,
            num_layers=self.config.num_layers,
            max_seq_len=self.config.max_seq_len,
            hidden_dim=baseline_hidden,  # Use same hidden dim for fair comparison
            mlp_type="standard",  # DoubtAwareGPT uses MetaAwareBlock internally
            dropout=self.config.dropout,
            poly_rank=16,
            poly_degree=3,
        )
        doubt_model = DoubtAwareGPT(doubt_config)

        # Log parameter counts
        std_params = sum(p.numel() for p in standard_model.parameters())
        doubt_params = sum(p.numel() for p in doubt_model.parameters())

        logging.info(f"Standard model parameters: {std_params:,}")
        logging.info(f"Doubt model parameters: {doubt_params:,}")
        logging.info(
            f"Parameter difference: {doubt_params - std_params:,} ({(doubt_params - std_params) / std_params * 100:.2f}%)"
        )

        return standard_model, doubt_model

    def train_doubt_model(
        self,
        model: DoubtAwareGPT,
        train_loader,
        eval_loader,
        device: str,
        seed: int,
        tokenizer=None,  # Optional tokenizer for sampling
    ) -> DoubtTrainingResult:
        """
        Two-pass training loop using unified trainer.

        Args:
            model: DoubtAwareGPT model
            train_loader: DataLoader yielding (tokens_t, tokens_t_plus_1)
            eval_loader: DataLoader for evaluation
            device: Device to train on
            seed: Random seed for reproducibility
            tokenizer: Optional tokenizer for sampling during evaluation

        Returns:
            DoubtTrainingResult with training metrics
        """
        strategy = DoubtTrainingStrategy(self.config)
        trainer = UnifiedTrainer(
            self.config,
            strategy,
            "doubt",
            tokenizer=tokenizer,
            results_dir=self.results_dir,
        )
        result = trainer.train(model, train_loader, eval_loader, seed)
        # Type cast since DoubtTrainingStrategy returns DoubtTrainingResult
        return result  # type: ignore

    def train_standard_model(
        self,
        model: MiniGPT,
        train_loader,
        eval_loader,
        device: str,
        seed: int,
        tokenizer=None,
    ) -> TrainingResult:
        """
        Standard training loop using unified trainer.

        Args:
            model: Standard MiniGPT model
            train_loader: DataLoader yielding (tokens_t, tokens_t_plus_1)
            eval_loader: DataLoader for evaluation
            device: Device to train on
            seed: Random seed for reproducibility
            tokenizer: Optional tokenizer for sampling during evaluation

        Returns:
            TrainingResult with training metrics
        """
        strategy = StandardTrainingStrategy(self.config)
        trainer = UnifiedTrainer(
            self.config,
            strategy,
            "standard",
            tokenizer=tokenizer,
            results_dir=self.results_dir,
        )
        result = trainer.train(model, train_loader, eval_loader, seed)
        # Type cast since StandardTrainingStrategy returns TrainingResult
        return result  # type: ignore

    def _evaluate_doubt_model(self, model: DoubtAwareGPT, eval_loader, device: str, num_batches: int) -> float:
        """Evaluate doubt model on validation set."""
        model.eval()
        total_loss = 0.0
        total_batches = 0

        with torch.no_grad():
            for i, (tokens_t, tokens_t_plus_1) in enumerate(eval_loader):
                if i >= num_batches:
                    break

                tokens_t = tokens_t.to(device)
                tokens_t_plus_1 = tokens_t_plus_1.to(device)

                # Initialize doubt
                batch_size = tokens_t.shape[0]
                seq_len = tokens_t.shape[1]
                prev_doubt = torch.zeros(
                    (batch_size, seq_len, 1),
                    device=device,
                    dtype=model.tok_emb.weight.dtype,
                )

                # Two-pass evaluation
                result = DoubtAwareGPT.two_pass_training_step(
                    model=model,
                    tokens_t=tokens_t,
                    tokens_t_plus_1=tokens_t_plus_1,
                    alpha=self.config.doubt_alpha,
                )

                total_loss += result["loss_lm"].item()
                total_batches += 1

        model.train()
        return total_loss / total_batches if total_batches > 0 else float("inf")

    def _evaluate_standard_model(self, model: MiniGPT, eval_loader, device: str, num_batches: int) -> float:
        """Evaluate standard model on validation set."""
        model.eval()
        total_loss = 0.0
        total_batches = 0

        with torch.no_grad():
            for i, (tokens_t, tokens_t_plus_1, _) in enumerate(eval_loader):
                if i >= num_batches:
                    break

                input_ids = tokens_t.to(device)
                # For standard model, we need sequence labels (shifted by 1)
                # tokens_t has shape [batch_size, seq_len]
                # tokens_t_plus_1 has shape [batch_size] (single token at position seq_len)
                # Create sequence labels: tokens_t[:, 1:] concatenated with tokens_t_plus_1
                seq_labels = torch.cat(
                    [
                        tokens_t[:, 1:].to(device),
                        tokens_t_plus_1.unsqueeze(1).to(device),
                    ],
                    dim=1,
                )  # shape [batch_size, seq_len]

                # Model no longer computes loss internally
                logits = model(input_ids)
                loss = compute_sequence_language_modeling_loss(logits, seq_labels, reduction="mean")
                total_loss += loss.item()
                total_batches += 1

        model.train()
        return total_loss / total_batches if total_batches > 0 else float("inf")

    def run(self, tokenizer) -> DoubtExperimentResults:
        """
        Run full experiment comparing both models.

        Args:
            tokenizer: Tokenizer for the dataset

        Returns:
            DoubtExperimentResults with results from both models
        """
        logging.info("Starting doubt-aware experiment")
        logging.info(f"Configuration: {self.config}")

        # Prepare data
        logging.info("Preparing data...")
        dataloaders = prepare_doubt_data(
            config=self.config,
            tokenizer=tokenizer,
            dataset_kwargs=self.config.dataset_kwargs,
        )

        train_loader = dataloaders["train"]
        eval_loader = dataloaders["eval"]

        # Create models
        logging.info("Creating models...")
        standard_model, doubt_model = self.create_models()

        # Get parameter counts
        std_params = sum(p.numel() for p in standard_model.parameters())
        doubt_params = sum(p.numel() for p in doubt_model.parameters())

        parameter_counts = {"standard": std_params, "doubt": doubt_params}

        # Run experiments for each seed
        standard_results = {}
        doubt_results = {}
        training_times = {"standard": [], "doubt": []}

        for seed in self.config.seeds:
            logging.info(f"Running experiment with seed={seed}")

            # Train doubt model
            logging.info(f"Training doubt model (seed={seed})...")
            doubt_result = self.train_doubt_model(
                doubt_model,
                train_loader,
                eval_loader,
                self.device,
                seed,
                tokenizer,
            )
            doubt_results[str(seed)] = [doubt_result]
            training_times["doubt"].append(doubt_result.training_time)
            # Train standard model
            logging.info(f"Training standard model (seed={seed})...")
            standard_result = self.train_standard_model(
                standard_model, train_loader, eval_loader, self.device, seed, tokenizer
            )
            standard_results[str(seed)] = [standard_result]
            training_times["standard"].append(standard_result.training_time)

            # Save intermediate results
            if self.config.save_interval > 0 and (seed % self.config.save_interval == 0):
                self._save_intermediate_results(standard_results, doubt_results, parameter_counts, seed)

        # Calculate average training times
        avg_training_time = {
            "standard": (float(np.mean(training_times["standard"])) if training_times["standard"] else 0.0),
            "doubt": (float(np.mean(training_times["doubt"])) if training_times["doubt"] else 0.0),
        }

        # Create final results object
        results = DoubtExperimentResults(
            standard_model_results=standard_results,
            doubt_model_results=doubt_results,
            config=self.config,
            timestamp=datetime.now().isoformat(),
            training_time=avg_training_time,
            parameter_counts=parameter_counts,
        )

        # Save final results using the new serialization
        save_results(results, self.results_dir)

        logging.info("Experiment completed successfully")
        return results

    def _save_intermediate_results(
        self,
        standard_results: Dict[str, List[TrainingResult]],
        doubt_results: Dict[str, List[DoubtTrainingResult]],
        parameter_counts: Dict[str, int],
        seed: int,
    ):
        """Save intermediate results to disk."""
        # Calculate training times from results
        training_time = {"standard": 0.0, "doubt": 0.0}
        if standard_results:
            # Get average training time for standard models
            std_times = [r.training_time for seed_list in standard_results.values() for r in seed_list]
            if std_times:
                training_time["standard"] = float(np.mean(std_times))

        if doubt_results:
            # Get average training time for doubt models
            doubt_times = [r.training_time for seed_list in doubt_results.values() for r in seed_list]
            if doubt_times:
                training_time["doubt"] = float(np.mean(doubt_times))

        intermediate_results = DoubtExperimentResults(
            standard_model_results=standard_results,
            doubt_model_results=doubt_results,
            config=self.config,
            timestamp=datetime.now().isoformat(),
            training_time=training_time,
            parameter_counts=parameter_counts,
        )

        # Use the new serialization function
        filename = f"intermediate_results_seed_{seed}.json"
        save_path = save_results(intermediate_results, self.results_dir, filename)
        logging.info(f"Saved intermediate results to {save_path}")

    def _save_results(self, results: DoubtExperimentResults):
        """Save final results to disk."""
        # Use the new serialization function
        save_results(results, self.results_dir, "final_results.json")
