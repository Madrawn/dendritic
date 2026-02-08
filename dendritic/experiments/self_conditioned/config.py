from dataclasses import dataclass, field
from dendritic.experiments.utils.PretrainingConfig import PretrainingConfig


@dataclass
class SelfConditionedExperimentConfig(PretrainingConfig):
    """Configuration for self-conditioned experiments.

    Extends PretrainingConfig with parameters specific to SelfConditionedGPT.
    """

    # Self-conditioned specific parameters
    bound_fn: str = "tanh"  # Bound function for the doubt signal: "tanh", "sigmoid", "softsign", "relu", "none"
    take_meta: int = 3  # Number of layers to use for meta prediction

    # Experiment tracking
    results_dir: str = "results/self_conditioned_experiments"
    save_interval: int = 100  # Save intermediate results every N steps

    # Token sampling configuration
    sampling_prompt: str = "Once upon a time"
    sampling_temperature: float = 0.8
    sampling_top_p: float = 0.95
    sampling_max_tokens: int = 50

    # Miscellaneous
    eval_smoothing_factor: float = 0.5  # Smoothing factor for eval loss moving average

    def __post_init__(self):
        """Initialize derived fields after dataclass initialization."""
        if not self.results_dir:
            self.results_dir = "results/self_conditioned_experiments"
