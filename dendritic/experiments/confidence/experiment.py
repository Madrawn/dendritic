"""
Confidence-aware GPT experiment implementation.

This module implements the ConfidenceAwareExperiment class for comparing
ConfidenceAwareGPT (with two-pass lookahead training) vs standard MiniGPT.
"""

import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass, field
from pathlib import Path

from dendritic.experiments.models.MiniGPT import MiniGPT, ConfidenceAwareGPT
from dendritic.experiments.utils.TrainingResult import TrainingResult
from dendritic.experiments.utils.PretrainingConfig import PretrainingConfig
from dendritic.experiments.utils.param_utils import find_matching_hidden_dims
from dendritic.experiments.utils.experiment_utils import set_random_seed
from dendritic.experiments.utils.loss_utils import compute_language_modeling_loss

from dendritic.experiments.confidence.config import ConfidenceExperimentConfig
from dendritic.experiments.confidence.data_loader import prepare_confidence_data
from dendritic.experiments.confidence.results import (
    ConfidenceTrainingResult,
    ConfidenceExperimentResults,
    save_results,
    load_results,
    create_results_filename,
)
from dendritic.experiments.confidence.UnifiedTrainer import (
    UnifiedTrainer,
)
from dendritic.experiments.confidence.ConfidenceTrainingStrategy import (
    ConfidenceTrainingStrategy,
)
from dendritic.experiments.confidence.StandardTrainingStrategy import (
    StandardTrainingStrategy,
)


class ConfidenceAwareExperiment:
    """
    Experiment comparing ConfidenceAwareGPT vs standard MiniGPT.

    This class implements the two-pass lookahead training for ConfidenceAwareGPT
    and standard training for MiniGPT, with proper parameter logging and
    results tracking.
    """

    def __init__(self, config: ConfidenceExperimentConfig):
        """
        Initialize the experiment with configuration.

        Args:
            config: ConfidenceExperimentConfig containing experiment parameters
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

    def create_models(self) -> Tuple[MiniGPT, ConfidenceAwareGPT]:
        """
        Create both model variants with parameter matching.

        Returns:
            Tuple of (standard_model, confidence_model)
        """
        # Get matching hidden dimensions
        baseline_hidden, dendritic_hidden = find_matching_hidden_dims(self.config)

        # Create standard MiniGPT
        standard_model = MiniGPT(
            vocab_size=self.config.vocab_size,
            embed_dim=self.config.embed_dim,
            num_heads=self.config.num_heads,
            num_layers=self.config.num_layers,
            max_seq_len=self.config.max_seq_len,
            hidden_dim=baseline_hidden,
            mlp_type="standard",
            dropout=self.config.dropout,
        )

        # Create ConfidenceAwareGPT
        # Confidence model needs to handle sequences of length max_seq_len + 1
        # for lookahead training (hypothetical sequence with appended token)
        confidence_max_seq_len = self.config.max_seq_len + 1
        confidence_model = ConfidenceAwareGPT(
            vocab_size=self.config.vocab_size,
            embed_dim=self.config.embed_dim,
            num_heads=self.config.num_heads,
            num_layers=self.config.num_layers,
            max_seq_len=confidence_max_seq_len,
            hidden_dim=baseline_hidden,  # Use same hidden dim for fair comparison
            mlp_type="standard",  # ConfidenceAwareGPT uses MetaAwareBlock internally
            dropout=self.config.dropout,
        )

        # Log parameter counts
        std_params = sum(p.numel() for p in standard_model.parameters())
        conf_params = sum(p.numel() for p in confidence_model.parameters())

        logging.info(f"Standard model parameters: {std_params:,}")
        logging.info(f"Confidence model parameters: {conf_params:,}")
        logging.info(
            f"Parameter difference: {conf_params - std_params:,} "
            f"({(conf_params - std_params) / std_params * 100:.2f}%)"
        )

        return standard_model, confidence_model

    def train_confidence_model(
        self,
        model: ConfidenceAwareGPT,
        train_loader,
        eval_loader,
        device: str,
        seed: int,
    ) -> ConfidenceTrainingResult:
        """
        Two-pass training loop using unified trainer.

        Args:
            model: ConfidenceAwareGPT model
            train_loader: DataLoader yielding (tokens_t, tokens_t_plus_1, tokens_t_plus_2)
            eval_loader: DataLoader for evaluation
            device: Device to train on
            seed: Random seed for reproducibility

        Returns:
            ConfidenceTrainingResult with training metrics
        """
        strategy = ConfidenceTrainingStrategy(self.config)
        trainer = UnifiedTrainer(self.config, strategy, "confidence")
        result = trainer.train(model, train_loader, eval_loader, seed)
        # Type cast since ConfidenceTrainingStrategy returns ConfidenceTrainingResult
        return result  # type: ignore

    def train_standard_model(
        self, model: MiniGPT, train_loader, eval_loader, device: str, seed: int
    ) -> TrainingResult:
        """
        Standard training loop using unified trainer.

        Args:
            model: Standard MiniGPT model
            train_loader: DataLoader yielding (tokens_t, tokens_t_plus_1, tokens_t_plus_2)
            eval_loader: DataLoader for evaluation
            device: Device to train on
            seed: Random seed for reproducibility

        Returns:
            TrainingResult with training metrics
        """
        strategy = StandardTrainingStrategy(self.config)
        trainer = UnifiedTrainer(self.config, strategy, "standard")
        result = trainer.train(model, train_loader, eval_loader, seed)
        # Type cast since StandardTrainingStrategy returns TrainingResult
        return result  # type: ignore

    def _evaluate_confidence_model(
        self, model: ConfidenceAwareGPT, eval_loader, device: str, num_batches: int
    ) -> float:
        """Evaluate confidence model on validation set."""
        model.eval()
        total_loss = 0.0
        total_batches = 0

        with torch.no_grad():
            for i, (tokens_t, tokens_t_plus_1, tokens_t_plus_2) in enumerate(
                eval_loader
            ):
                if i >= num_batches:
                    break

                tokens_t = tokens_t.to(device)
                tokens_t_plus_1 = tokens_t_plus_1.to(device)
                tokens_t_plus_2 = tokens_t_plus_2.to(device)

                # Initialize confidence
                batch_size = tokens_t.shape[0]
                seq_len = tokens_t.shape[1]
                prev_conf = torch.zeros(
                    (batch_size, seq_len, 1),
                    device=device,
                    dtype=model.tok_emb.weight.dtype,
                )

                # Two-pass evaluation
                result = ConfidenceAwareGPT.two_pass_training_step(
                    model=model,
                    tokens_t=tokens_t,
                    tokens_t_plus_1=tokens_t_plus_1,
                    alpha=self.config.confidence_alpha,
                )

                total_loss += result["loss_lm"].item()
                total_batches += 1

        model.train()
        return total_loss / total_batches if total_batches > 0 else float("inf")

    def _evaluate_standard_model(
        self, model: MiniGPT, eval_loader, device: str, num_batches: int
    ) -> float:
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
                loss = compute_language_modeling_loss(logits, seq_labels)
                total_loss += loss.item()
                total_batches += 1

        model.train()
        return total_loss / total_batches if total_batches > 0 else float("inf")

    def run(self, tokenizer) -> ConfidenceExperimentResults:
        """
        Run full experiment comparing both models.

        Args:
            tokenizer: Tokenizer for the dataset

        Returns:
            ConfidenceExperimentResults with results from both models
        """
        logging.info("Starting confidence-aware experiment")
        logging.info(f"Configuration: {self.config}")

        # Prepare data
        logging.info("Preparing data...")
        dataloaders = prepare_confidence_data(
            config=self.config,
            tokenizer=tokenizer,
            dataset_kwargs=self.config.dataset_kwargs,
        )

        train_loader = dataloaders["train"]
        eval_loader = dataloaders["eval"]

        # Create models
        logging.info("Creating models...")
        standard_model, confidence_model = self.create_models()

        # Get parameter counts
        std_params = sum(p.numel() for p in standard_model.parameters())
        conf_params = sum(p.numel() for p in confidence_model.parameters())

        parameter_counts = {"standard": std_params, "confidence": conf_params}

        # Run experiments for each seed
        standard_results = {}
        confidence_results = {}
        training_times = {"standard": [], "confidence": []}

        for seed in self.config.seeds:
            logging.info(f"Running experiment with seed={seed}")

            # Train confidence model
            logging.info(f"Training confidence model (seed={seed})...")
            confidence_result = self.train_confidence_model(
                confidence_model, train_loader, eval_loader, self.device, seed
            )
            confidence_results[str(seed)] = [confidence_result]
            training_times["confidence"].append(confidence_result.training_time)
            # Train standard model
            logging.info(f"Training standard model (seed={seed})...")
            standard_result = self.train_standard_model(
                standard_model, train_loader, eval_loader, self.device, seed
            )
            standard_results[str(seed)] = [standard_result]
            training_times["standard"].append(standard_result.training_time)

            # Save intermediate results
            if self.config.save_interval > 0 and (
                seed % self.config.save_interval == 0
            ):
                self._save_intermediate_results(
                    standard_results, confidence_results, parameter_counts, seed
                )

        # Calculate average training times
        avg_training_time = {
            "standard": (
                float(np.mean(training_times["standard"]))
                if training_times["standard"]
                else 0.0
            ),
            "confidence": (
                float(np.mean(training_times["confidence"]))
                if training_times["confidence"]
                else 0.0
            ),
        }

        # Create final results object
        results = ConfidenceExperimentResults(
            standard_model_results=standard_results,
            confidence_model_results=confidence_results,
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
        confidence_results: Dict[str, List[ConfidenceTrainingResult]],
        parameter_counts: Dict[str, int],
        seed: int,
    ):
        """Save intermediate results to disk."""
        # Calculate training times from results
        training_time = {"standard": 0.0, "confidence": 0.0}
        if standard_results:
            # Get average training time for standard models
            std_times = [
                r.training_time
                for seed_list in standard_results.values()
                for r in seed_list
            ]
            if std_times:
                training_time["standard"] = float(np.mean(std_times))

        if confidence_results:
            # Get average training time for confidence models
            conf_times = [
                r.training_time
                for seed_list in confidence_results.values()
                for r in seed_list
            ]
            if conf_times:
                training_time["confidence"] = float(np.mean(conf_times))

        intermediate_results = ConfidenceExperimentResults(
            standard_model_results=standard_results,
            confidence_model_results=confidence_results,
            config=self.config,
            timestamp=datetime.now().isoformat(),
            training_time=training_time,
            parameter_counts=parameter_counts,
        )

        # Use the new serialization function
        filename = f"intermediate_results_seed_{seed}.json"
        save_path = save_results(intermediate_results, self.results_dir, filename)
        logging.info(f"Saved intermediate results to {save_path}")

    def _save_results(self, results: ConfidenceExperimentResults):
        """Save final results to disk."""
        # Use the new serialization function
        save_results(results, self.results_dir, "final_results.json")
