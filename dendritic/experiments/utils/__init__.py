"""
Utilities for experiments.
"""

# Note: check_seq_len module has run() function, not check_seq_len
from .check_seq_len import run as check_seq_len
from .custom_scaler import CohortLRScheduler as CustomScaler
from .experiment_finetuning import run_finetuning_experiment
from ..run_experiments import run_pretraining_experiment
from .experiment_utils import (
    set_random_seed,
    setup_logging,
    debug_dataset_integrity,
)
from .ExperimentResults import ExperimentResults
from .loss_utils import (
    compute_language_modeling_loss,
    compute_confidence_loss,
    compute_sequence_language_modeling_loss,
    compute_total_confidence_aware_loss,
)
from .param_utils import count_parameters
from .PretrainingConfig import PretrainingConfig
from .sweep import generate_scheduler_variants as run_sweep
from .TrainingResult import TrainingResult
from .visualization import plot_training_curves as plot_training_history

__all__ = [
    "check_seq_len",
    "CustomScaler",
    "run_finetuning_experiment",
    "run_pretraining_experiment",
    "set_random_seed",
    "setup_logging",
    "debug_dataset_integrity",
    "ExperimentResults",
    "compute_language_modeling_loss",
    "compute_confidence_loss",
    "compute_sequence_language_modeling_loss",
    "compute_total_confidence_aware_loss",
    "count_parameters",
    "PretrainingConfig",
    "run_sweep",
    "TrainingResult",
    "plot_training_history",
]
