"""
Configuration for confidence-aware GPT experiments.
"""

from dataclasses import dataclass
from dendritic.experiments.utils.PretrainingConfig import (
    PretrainingConfig,
)


@dataclass
class ConfidenceExperimentConfig(PretrainingConfig):
    """Configuration for confidence-aware experiments.

    Extends PretrainingConfig with confidence-specific parameters for
    two-pass lookahead training.
    """

    # Confidence-specific parameters
    confidence_alpha: float = 1.0  # Weight for confidence loss
    lookahead_steps: int = 2  # Number of steps to look ahead (fixed at 2 for now)
    confidence_init_bias: float = 2.0  # Initial bias for confidence predictor

    # Experiment tracking
    results_dir: str = "results/confidence_experiments"
    save_interval: int = 100  # Save intermediate results every N steps

    # Token sampling configuration
    sampling_prompt: str = "Once upon a time"  # Prompt for sampling during evaluation
    sampling_temperature: float = 0.8  # Temperature for sampling
    sampling_top_p: float = 0.95  # Top-p (nucleus) sampling parameter
    sampling_max_tokens: int = 50  # Maximum tokens to sample during evaluation

    # Miscellaneous
    eval_smoothing_factor: float = 0.5  # Smoothing factor for eval loss moving average (for early exit)

    def __post_init__(self):
        """Initialize derived fields after dataclass initialization."""
        # Ensure results_dir is a string path
        if not self.results_dir:
            self.results_dir = "results/confidence_experiments"
