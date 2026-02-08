from dendritic.experiments.doubt.TrainingStrategy import TrainingStrategy
from dendritic.experiments.doubt.config import DoubtExperimentConfig
from dendritic.experiments.doubt.results import DoubtTrainingResult
from dendritic.experiments.models.DoubtAwareGPT import DoubtAwareGPT
from dendritic.experiments.utils.loss_utils import (
    compute_sequence_language_modeling_loss,
)


import numpy as np
import torch
from torch import nn


from typing import Any, cast


class DoubtTrainingStrategy(TrainingStrategy):
    """Strategy for doubt-aware two-pass training."""

    def __init__(self, config: DoubtExperimentConfig):
        super().__init__(config)
        self.prev_doubt = None

    def training_step(self, model: DoubtAwareGPT | nn.Module, batch, device, **kwargs):
        tokens_t, tokens_t_plus_1 = batch
        model = cast(DoubtAwareGPT, model)
        result = model.two_pass_training_step(
            model=model,
            tokens_t=tokens_t,
            tokens_t_plus_1=tokens_t_plus_1,
            alpha=self.config.doubt_alpha,
        )

        return result

    def evaluation_step(
        self, model: nn.Module, batch: tuple[torch.Tensor, ...], device: str, **kwargs
    ) -> dict[str, torch.Tensor]:
        """
        Perform evaluation step for doubt model.

        Args:
            model: DoubtAwareGPT model
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

        # Forward pass with default doubt scalars (zeros)
        # Model no longer computes loss internally
        outputs = model(input_ids, doubt_scalars=None)

        # Compute language modeling loss externally (no shifting since seq_labels already aligned)
        loss = compute_sequence_language_modeling_loss(outputs["logits"], seq_labels, reduction="mean")

        # Return loss_lm for consistency with training step
        # Note: Doubt loss is not computed during evaluation because it requires
        # two-pass lookahead (future losses) which are not available in evaluation mode.
        return {
            "loss_lm": loss,
            "loss": loss,
            "doubt_loss": torch.tensor(0.0, device=loss.device),  # No doubt loss in evaluation (intentional design)
        }

    def prepare_batch(self, batch: tuple[torch.Tensor, ...], device: str) -> tuple[torch.Tensor, ...]:
        """
        Prepare batch for doubt training.

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
    ) -> DoubtTrainingResult:
        """
        Create DoubtTrainingResult with doubt metrics.

        Args:
            model_type: Type of model (should be "doubt")
            seed: Random seed used
            final_train_loss: Final training loss
            final_eval_loss: Final evaluation loss
            best_eval_loss: Best evaluation loss
            loss_history: History of loss metrics
            training_time: Total training time
            additional_metrics: Additional doubt metrics

        Returns:
            DoubtTrainingResult
        """
        if additional_metrics is None:
            additional_metrics = {}

        return DoubtTrainingResult(
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
            doubt_loss_history=additional_metrics.get("doubt_loss_history", []),
            token_loss_history=additional_metrics.get("token_loss_history", []),
            loss_predictions=additional_metrics.get("loss_predictions", []),
            actual_future_losses=additional_metrics.get("actual_future_losses", []),
        )
