from dataclasses import dataclass, field
from typing import Literal, get_type_hints


# Configuration for optional Cohort LR Scheduler
@dataclass
class CohortSchedulerConfig:
    """Parameters for the optional CohortLRScheduler."""

    min_mult: float = 0.5  # Minimum multiplier for LR scaling
    max_mult: float = 1.0  # Maximum multiplier for LR scaling
    sharpness: float = (
        1.0  # Sharpness of the cosine peak (higher = narrower high LR band)
    )
    device: str = "cpu"  # Device for scheduler tensors
    apply_to_gradients: bool = (
        True  # Whether to modify gradients (default current behavior)
    )


class AutoVivifyMixin:
    def set_deep(self, branch_name: str, leaf_name: str, value):
        """
        Sets self.branch_name.leaf_name = value.
        If self.branch_name is None, it creates it using the type hint.
        """
        # 1. Check if the branch (e.g., 'dev_conf') already exists
        current_branch = getattr(self, branch_name)

        if current_branch is None:
            # 2. It's None, so we need to create it.
            # We look at the class type hints to find out what 'dev_conf' should be.
            hints = get_type_hints(self.__class__)
            target_type = hints.get(branch_name)

            if not target_type:
                raise ValueError(
                    f"Field '{branch_name}' not defined in {self.__class__.__name__}"
                )

            # 3. Handle Optional[Type] (which is effectively Union[Type, NoneType])
            # If the type is Optional[DeveloperConf], we need to extract DeveloperConf
            if hasattr(target_type, "__args__"):
                # Usually the first arg is the actual class, the second is NoneType
                target_cls = target_type.__args__[0]
            else:
                target_cls = target_type

            # 4. Instantiate the class (assumes it has a parameter-less init)
            current_branch = target_cls()
            setattr(self, branch_name, current_branch)

        # 5. Set the final value
        setattr(current_branch, leaf_name, value)


@dataclass
class PretrainingConfig(AutoVivifyMixin):
    """Configuration for pretraining experiment."""

    # Model architecture
    vocab_size: int = 50257
    embed_dim: int = 384  # Smaller for faster experiments
    num_heads: int = 6
    num_layers: int = 6
    max_seq_len: int = 1024
    dropout: float = 0.0
    layer_type: Literal["standard", "dendritic"] = "standard"
    hidden_dim: int = 128

    # Dendritic-specific
    poly_rank: int = 16
    poly_degree: int = 3
    dendritic_dropout: float = 0.0

    # Training
    training_steps: int = 21000
    batch_size: int = 6
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    scheduler_type: str = "plateau"  # "cosine" or "plateau"
    eval_split_ratio: float = 0.1

    # ReduceOnPlateau specific parameters
    plateau_patience: int = 4
    plateau_factor: float = 0.5
    plateau_threshold: float = 1e-3
    plateau_cooldown: int = 0
    plateau_min_lr: float = 1e-6
    early_stop_multiplier: int = (
        2  # multiplier for plateau_patience to trigger early stopping
    )

    # Evaluation
    effective_eval_interval: int | None = (
        None  # If None, computed as training_steps // 20
    )

    eval_batches: int = 15

    # Experiment
    seeds: list[int] = field(default_factory=lambda: [42, 123, 456, 789, 1011])
    # Optional Cohort scheduler configuration; if None, original behavior is used
    cohort_scheduler: CohortSchedulerConfig | None = None
    output_dir: str = "results/pretraining_comparison"

    # Dataset configuration
    dataset: str = "openwebmath"
    grouped: bool = False
    group_separator: Literal["EOS_token", "EOS_BOS_tokens"] | str = "EOS_token"
    dataset_kwargs: dict = field(default_factory=dict)

    baseline_hidden_dim: int = 0
    dendritic_hidden_dim: int = 0
    dendritic_stack_hidden_dim: int = 0
    param_grid: dict = field(default_factory=dict)

    @property
    def warmup_steps(self) -> int:
        """This acts as a getter and is calculated on demand."""
        return self.training_steps // 20

    @property
    def eval_interval(self) -> int:
        """Handle the 'dynamic default' logic without __post_init__."""
        if self.effective_eval_interval is not None:
            return self.effective_eval_interval
        return max(self.training_steps // 20, 1)
