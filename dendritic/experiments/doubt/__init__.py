"""
Doubt-aware GPT experiment module.

This module provides components for running doubt-aware GPT experiments
comparing DoubtAwareGPT (with two-pass lookahead training) vs standard MiniGPT.
"""

from .config import DoubtExperimentConfig
from .data_loader import prepare_doubt_data
from .experiment import DoubtAwareExperiment
from .results import (
    DoubtTrainingResult,
    DoubtExperimentResults,
    save_results,
    load_results,
    create_results_filename,
)
from .visualization import (
    plot_loss_curves,
    plot_calibration_curve,
    plot_training_time_comparison,
    generate_summary_statistics,
)

__all__ = [
    "DoubtExperimentConfig",
    "prepare_doubt_data",
    "DoubtAwareExperiment",
    "DoubtTrainingResult",
    "DoubtExperimentResults",
    "save_results",
    "load_results",
    "create_results_filename",
    "plot_loss_curves",
    "plot_calibration_curve",
    "plot_training_time_comparison",
    "generate_summary_statistics",
]
