"""
Self-conditioned GPT experiment implementation.

This module implements the SelfConditionedExperiment class for comparing
SelfConditionedGPT (with self-conditioning) vs standard MiniGPT.
Both models are trained using the standard training strategy.
"""

import logging
from datetime import datetime
from typing import Dict, List, Tuple
import torch
import numpy as np
from pathlib import Path

from dendritic.experiments.models.SelfConditionedGPT import SelfConditionedGPT
from dendritic.experiments.models.MiniGPT import MiniGPT
from dendritic.experiments.models.ModelConfig import ModelConfig
from dendritic.experiments.utils.TrainingResult import TrainingResult
from dendritic.experiments.utils.param_utils import find_matching_hidden_dims

from dendritic.experiments.self_conditioned.config import SelfConditionedExperimentConfig
from dendritic.experiments.self_conditioned.data_loader import prepare_self_conditioned_data
from dendritic.experiments.self_conditioned.results import (
    SelfConditionedExperimentResults,
    save_results,
)
from dendritic.experiments.doubt.UnifiedTrainer import UnifiedTrainer
from dendritic.experiments.doubt.StandardTrainingStrategy import StandardTrainingStrategy


class SelfConditionedExperiment:
    """
    Experiment comparing MiniGPT vs SelfConditionedGPT.

    Both models are trained with standard next-token prediction.
    SelfConditionedGPT uses an internal two-pass forward to generate
    its own conditioning signal, but from the trainer's perspective it
    is a standard model returning logits.
    """

    def __init__(self, config: SelfConditionedExperimentConfig):
        """
        Initialize the experiment with configuration.

        Args:
            config: SelfConditionedExperimentConfig containing experiment parameters
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

    def create_models(self) -> Tuple[MiniGPT, SelfConditionedGPT]:
        """
        Create both model variants with parameter matching.

        Returns:
            Tuple of (standard_model, self_conditioned_model)
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

        # Create SelfConditionedGPT with ModelConfig
        self_cond_config = ModelConfig(
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
        self_conditioned_model = SelfConditionedGPT(
            config=self_cond_config,
            bound_fn=self.config.bound_fn,
            take_meta=self.config.take_meta,
        )

        # Log parameter counts
        std_params = sum(p.numel() for p in standard_model.parameters())
        self_cond_params = sum(p.numel() for p in self_conditioned_model.parameters())

        logging.info(f"Standard model parameters: {std_params:,}")
        logging.info(f"Self-conditioned model parameters: {self_cond_params:,}")
        logging.info(
            f"Parameter difference: {self_cond_params - std_params:,} ({(self_cond_params - std_params) / std_params * 100:.2f}%)"
        )

        return standard_model, self_conditioned_model

    def train_self_conditioned_model(
        self,
        model: SelfConditionedGPT,
        train_loader,
        eval_loader,
        device: str,
        seed: int,
        tokenizer=None,
    ) -> TrainingResult:
        """
        Standard training loop for self-conditioned model.

        Args:
            model: SelfConditionedGPT model
            train_loader: DataLoader yielding (tokens_t, tokens_t_plus_1)
            eval_loader: DataLoader for evaluation
            device: Device to train on
            seed: Random seed for reproducibility
            tokenizer: Optional tokenizer for sampling during evaluation

        Returns:
            TrainingResult with training metrics
        """
        strategy = StandardTrainingStrategy(self.config)  # type: ignore
        trainer = UnifiedTrainer(
            self.config,
            strategy,
            "self_conditioned",
            tokenizer=tokenizer,
            results_dir=self.results_dir,
        )
        result = trainer.train(model, train_loader, eval_loader, seed)
        # StandardTrainingStrategy returns TrainingResult
        return result

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
        Standard training loop for baseline model.

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
        strategy = StandardTrainingStrategy(self.config)  # type: ignore
        trainer = UnifiedTrainer(
            self.config,
            strategy,
            "standard",
            tokenizer=tokenizer,
            results_dir=self.results_dir,
        )
        result = trainer.train(model, train_loader, eval_loader, seed)
        return result

    def run(self, tokenizer) -> SelfConditionedExperimentResults:
        """
        Run full experiment comparing both models.

        Args:
            tokenizer: Tokenizer for the dataset

        Returns:
            SelfConditionedExperimentResults with results from both models
        """
        logging.info("Starting self-conditioned experiment")
        logging.info(f"Configuration: {self.config}")

        # Prepare data
        logging.info("Preparing data...")
        dataloaders = prepare_self_conditioned_data(
            config=self.config,
            tokenizer=tokenizer,
            dataset_kwargs=self.config.dataset_kwargs,
        )

        train_loader = dataloaders["train"]
        eval_loader = dataloaders["eval"]

        # Create models
        logging.info("Creating models...")
        standard_model, self_conditioned_model = self.create_models()

        # Get parameter counts
        std_params = sum(p.numel() for p in standard_model.parameters())
        self_cond_params = sum(p.numel() for p in self_conditioned_model.parameters())

        parameter_counts = {"standard": std_params, "self_conditioned": self_cond_params}

        # Run experiments for each seed
        standard_results = {}
        self_conditioned_results = {}
        training_times = {"standard": [], "self_conditioned": []}

        for seed in self.config.seeds:
            logging.info(f"Running experiment with seed={seed}")

            # Train self-conditioned model
            logging.info(f"Training self-conditioned model (seed={seed})...")
            self_cond_result = self.train_self_conditioned_model(
                self_conditioned_model,
                train_loader,
                eval_loader,
                self.device,
                seed,
                tokenizer,
            )
            self_conditioned_results[str(seed)] = [self_cond_result]
            training_times["self_conditioned"].append(self_cond_result.training_time)

            # Train standard model
            logging.info(f"Training standard model (seed={seed})...")
            standard_result = self.train_standard_model(
                standard_model, train_loader, eval_loader, self.device, seed, tokenizer
            )
            standard_results[str(seed)] = [standard_result]
            training_times["standard"].append(standard_result.training_time)

            # Save intermediate results
            if self.config.save_interval > 0 and (seed % self.config.save_interval == 0):
                self._save_intermediate_results(standard_results, self_conditioned_results, parameter_counts, seed)

        # Calculate average training times
        avg_training_time = {
            "standard": (float(np.mean(training_times["standard"])) if training_times["standard"] else 0.0),
            "self_conditioned": (
                float(np.mean(training_times["self_conditioned"])) if training_times["self_conditioned"] else 0.0
            ),
        }

        # Create final results object
        results = SelfConditionedExperimentResults(
            standard_model_results=standard_results,
            self_conditioned_model_results=self_conditioned_results,
            config=self.config,
            timestamp=datetime.now().isoformat(),
            training_time=avg_training_time,
            parameter_counts=parameter_counts,
        )

        # Save final results
        save_results(results, self.results_dir)

        logging.info("Experiment completed successfully")
        return results

    def _save_intermediate_results(
        self,
        standard_results: Dict[str, List[TrainingResult]],
        self_conditioned_results: Dict[str, List[TrainingResult]],
        parameter_counts: Dict[str, int],
        seed: int,
    ):
        """Save intermediate results to disk."""
        training_time = {"standard": 0.0, "self_conditioned": 0.0}
        if standard_results:
            std_times = [r.training_time for seed_list in standard_results.values() for r in seed_list]
            if std_times:
                training_time["standard"] = float(np.mean(std_times))
        if self_conditioned_results:
            sc_times = [r.training_time for seed_list in self_conditioned_results.values() for r in seed_list]
            if sc_times:
                training_time["self_conditioned"] = float(np.mean(sc_times))

        intermediate_results = SelfConditionedExperimentResults(
            standard_model_results=standard_results,
            self_conditioned_model_results=self_conditioned_results,
            config=self.config,
            timestamp=datetime.now().isoformat(),
            training_time=training_time,
            parameter_counts=parameter_counts,
        )

        filename = f"intermediate_results_seed_{seed}.json"
        save_path = save_results(intermediate_results, self.results_dir, filename)
        logging.info(f"Saved intermediate results to {save_path}")
