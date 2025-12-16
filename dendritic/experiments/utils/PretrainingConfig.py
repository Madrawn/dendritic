from dataclasses import dataclass, field


@dataclass
class PretrainingConfig:
    """Configuration for pretraining experiment."""
    # Model architecture
    vocab_size: int = 50257
    embed_dim: int = 384        # Smaller for faster experiments
    num_heads: int = 6
    num_layers: int = 6
    max_seq_len: int = 256
    dropout: float = 0.1

    # Dendritic-specific
    poly_rank: int = 16
    dendritic_dropout: float = 0.1

    # Training
    training_steps: int = 10000
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = training_steps // 20
    max_grad_norm: float = 1.0
    scheduler_type: str = "cosine"  # "cosine" or "plateau"
    
    # ReduceOnPlateau specific parameters
    plateau_patience: int = 10
    plateau_factor: float = 0.5
    plateau_threshold: float = 1e-4
    plateau_cooldown: int = 0
    plateau_min_lr: float = 1e-6
    early_stop_multiplier: int = 2  # multiplier for plateau_patience to trigger early stopping

    # Evaluation
    eval_interval: int = max(training_steps // 10, 1)
    eval_batches: int = 100

    # Experiment
    seeds: list[int] = field(default_factory=lambda: [42, 123, 456, 789, 1011])
    output_dir: str = "results/pretraining_comparison"

    # Computed fields (set in __post_init__)
    baseline_hidden_dim: int = 0
    dendritic_hidden_dim: int = 0
    dendritic_stack_hidden_dim: int = 0

    def __post_init__(self):
        # We'll compute these after calculating non-MLP params
        pass