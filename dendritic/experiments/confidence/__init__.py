"""
Confidence-aware GPT experiment module.

This module provides components for running confidence-aware GPT experiments
comparing ConfidenceAwareGPT (with two-pass lookahead training) vs standard MiniGPT.
"""

from .config import ConfidenceExperimentConfig
from .data_loader import prepare_confidence_data
from .experiment import ConfidenceAwareExperiment
from .results import (
    ConfidenceTrainingResult,
    ConfidenceExperimentResults,
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
    "ConfidenceExperimentConfig",
    "prepare_confidence_data",
    "ConfidenceAwareExperiment",
    "ConfidenceTrainingResult",
    "ConfidenceExperimentResults",
    "save_results",
    "load_results",
    "create_results_filename",
    "plot_loss_curves",
    "plot_calibration_curve",
    "plot_training_time_comparison",
    "generate_summary_statistics",
]