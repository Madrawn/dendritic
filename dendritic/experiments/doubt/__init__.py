"""
Doubt-aware GPT experiment module.

This module provides components for running doubt-aware GPT experiments comparing DoubtAwareGPT vs standard MiniGPT.
"""

from .config import DoubtExperimentConfig
from .data_loader import prepare_doubt_data
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
