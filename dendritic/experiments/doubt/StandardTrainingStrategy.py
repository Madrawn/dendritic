from dendritic.experiments.models.doubt_conditioning.SelfConditionedGPT import SelfConditionedGPT
from dendritic.experiments.models.doubt_conditioning.DoubtAwareGPT import DoubtAwareGPT
from dendritic.experiments.utils.TrainingResult import TrainingResult
from dendritic.experiments.doubt.TrainingStrategy import TrainingStrategy
from dendritic.experiments.utils.loss_utils import (
    compute_sequence_language_modeling_loss,
)

import tqdm

import numpy as np
import torch
import torch.nn as nn


from typing import Any, cast


class StandardTrainingStrategy(TrainingStrategy):
    """Strategy for standard next-token prediction training."""

    def training_step(
        self, model: nn.Module, batch: tuple[torch.Tensor, ...], device: str, **kwargs
    ) -> dict[str, torch.Tensor]:
        """
        Perform standard training step.

        Args:
            model: MiniGPT model
            batch: Tuple of (tokens_t, tokens_t_plus_1)
            device: Device to run on

        Returns:
            Dictionary with loss metrics
        """
        tokens_t, tokens_t_plus_1 = batch

        # Prepare input and labels
        input_ids = tokens_t
        # Create sequence labels: tokens_t[:, 1:] concatenated with tokens_t_plus_1
        seq_labels = torch.cat(
            [
                tokens_t[:, 1:],
                tokens_t_plus_1.unsqueeze(1),
            ],
            dim=1,
        )  # shape [batch_size, seq_len]

        # Forward pass (model no longer computes loss internally)
        logits = model(input_ids)

        # Compute loss externally (no shifting since seq_labels already aligned)
        loss = compute_sequence_language_modeling_loss(logits, seq_labels, reduction="mean")

        # Return loss_lm for consistency with evaluation logic
        return {"loss_lm": loss, "loss": loss}

    def evaluation_step(
        self, model: nn.Module, batch: tuple[torch.Tensor, ...], device: str, **kwargs
    ) -> dict[str, torch.Tensor]:
        """
        Perform evaluation step for standard model.

        Args:
            model: MiniGPT model
            batch: Tuple of (tokens_t, tokens_t_plus_1)
            device: Device to run on

        Returns:
            Dictionary with loss metrics
        """
        # Move tensors to device
        batch = self.prepare_batch(batch, device)
        tokens_t, tokens_t_plus_1 = batch

        # Prepare input and labels
        input_ids = tokens_t
        # Create sequence labels: tokens_t[:, 1:] concatenated with tokens_t_plus_1
        seq_labels = torch.cat(
            [
                tokens_t[:, 1:],
                tokens_t_plus_1.unsqueeze(1),
            ],
            dim=1,
        )

        # Forward pass (model no longer computes loss internally)
        logits = model(input_ids)
        # if hasattr(model, "core") and hasattr(model.core, "loss_predictor"):
        # Use forward_with_diagnostics to get everything at once
        # model = cast(SelfConditionedGPT, model)
        # diagnostics = model.forward_with_diagnostics(input_ids)
        # logits_with_doubt = diagnostics["logits"]
        # logits_pass1 = diagnostics["pass1_logits"]
        # doubt_signal = diagnostics["doubt_signal"]

        # tqdm.tqdm.write(
        #     f"Logit difference (pass1 vs pass2): {str((logits_pass1 - logits_with_doubt).abs().mean())}"
        # )
        # tqdm.tqdm.write(
        #     (
        #         "Doubt signal stats - mean: "
        #         + str(doubt_signal.mean().item())
        #         + " std: "
        #         + str(doubt_signal.std().item())
        #     )
        # )
        # Compute loss externally (no shifting since seq_labels already aligned)
        loss = compute_sequence_language_modeling_loss(logits, seq_labels, reduction="mean")

        return {"loss": loss}

    def prepare_batch(self, batch: tuple[torch.Tensor, ...], device: str) -> tuple[torch.Tensor, ...]:
        """
        Prepare batch for standard training.

        Args:
            batch: Tuple of (tokens_t, tokens_t_plus_1)
            device: Device to move tensors to

        Returns:
            Prepared batch
        """
        tokens_t, tokens_t_plus_1 = batch
        return (
            tokens_t.to(device),
            tokens_t_plus_1.to(device),
        )

    def create_result(
        self,
        model_type: str,
        seed: int,
        final_train_loss: float,
        final_eval_loss: float,
        best_eval_loss: float,
        loss_history: list[dict[str, Any]],
        training_time: float,
        additional_metrics: dict[str, Any] | None = None,
    ) -> TrainingResult:
        """
        Create TrainingResult for standard model.

        Args:
            model_type: Type of model (should be "standard")
            seed: Random seed used
            final_train_loss: Final training loss
            final_eval_loss: Final evaluation loss
            best_eval_loss: Best evaluation loss
            loss_history: History of loss metrics
            training_time: Total training time
            additional_metrics: Additional metrics (unused for standard)

        Returns:
            TrainingResult
        """
        return TrainingResult(
            model_type=model_type,
            seed=seed,
            final_train_loss=final_train_loss,
            final_eval_loss=final_eval_loss,
            final_perplexity=np.exp(final_eval_loss),
            best_eval_loss=best_eval_loss,
            best_perplexity=np.exp(best_eval_loss),
            loss_history=loss_history,
            training_time=training_time,
            config=self.config.__dict__,
        )
