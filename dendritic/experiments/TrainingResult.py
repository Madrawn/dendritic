from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class TrainingResult:
    """Results from a single training run."""
    model_type: str
    seed: int
    final_train_loss: float
    final_eval_loss: float
    final_perplexity: float
    best_eval_loss: float
    best_perplexity: float
    loss_history: List[Dict[str, Any]]
    training_time: float
    config: Dict[str, Any]
    polynomial_stats: Dict[str, Any] = field(default_factory=dict)