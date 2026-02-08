from dendritic.experiments.doubt.config import DoubtExperimentConfig
from dendritic.experiments.doubt.results import DoubtTrainingResult
from dendritic.experiments.utils.TrainingResult import TrainingResult


import torch
import torch.nn as nn


from abc import ABC, abstractmethod
from typing import Any

from dendritic.experiments.models.DoubtAwareGPT import DoubtAwareGPT


class TrainingStrategy(ABC):
    """Abstract base class for training strategies."""

    def __init__(self, config: DoubtExperimentConfig):
        self.config = config

    @abstractmethod
    def training_step(
        self,
        model: nn.Module | DoubtAwareGPT,
        batch: tuple[torch.Tensor, ...],
        device: str,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """
        Perform a single training step.

        Args:
            model: Model to train
            batch: Batch of data (format depends on strategy)
            device: Device to run on
            **kwargs: Additional strategy-specific arguments

        Returns:
            Dictionary with loss metrics
        """
        pass

    @abstractmethod
    def evaluation_step(
        self, model: nn.Module, batch: tuple[torch.Tensor, ...], device: str, **kwargs
    ) -> dict[str, torch.Tensor]:
        """
        Perform a single evaluation step.

        Args:
            model: Model to evaluate
            batch: Batch of data
            device: Device to run on
            **kwargs: Additional strategy-specific arguments

        Returns:
            Dictionary with loss metrics
        """
        pass

    @abstractmethod
    def prepare_batch(self, batch: tuple[torch.Tensor, ...], device: str) -> tuple[torch.Tensor, ...]:
        """
        Prepare batch for training/evaluation.

        Args:
            batch: Raw batch from DataLoader
            device: Device to move tensors to

        Returns:
            Prepared batch
        """
        pass

    @abstractmethod
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
    ) -> TrainingResult | DoubtTrainingResult:
        """
        Create appropriate result object for this strategy.

        Args:
            model_type: Type of model (e.g., "standard", "doubt")
            seed: Random seed used
            final_train_loss: Final training loss
            final_eval_loss: Final evaluation loss
            best_eval_loss: Best evaluation loss
            loss_history: History of loss metrics
            training_time: Total training time
            additional_metrics: Additional strategy-specific metrics

        Returns:
            TrainingResult or subclass
        """
        pass
