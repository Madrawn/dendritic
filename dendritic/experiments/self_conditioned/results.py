"""
Results data structures and serialization for self-conditioned experiments.

This module defines the dataclasses for storing experiment results and provides
serialization/deserialization utilities for saving and loading results.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

from dendritic.experiments.utils.TrainingResult import TrainingResult
from dendritic.experiments.self_conditioned.config import SelfConditionedExperimentConfig


@dataclass
class SelfConditionedExperimentResults:
    """Results from a self-conditioned experiment."""

    standard_model_results: Dict[str, List[TrainingResult]]  # By seed
    self_conditioned_model_results: Dict[str, List[TrainingResult]]  # By seed
    config: SelfConditionedExperimentConfig
    timestamp: str
    training_time: Dict[str, float]  # Training time per model type
    parameter_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to a serializable dictionary."""
        result_dict = asdict(self)
        result_dict["config"] = asdict(self.config)
        return _convert_to_serializable(result_dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SelfConditionedExperimentResults":
        """Create SelfConditionedExperimentResults from a dictionary."""
        config_dict = data.get("config", {})
        config = SelfConditionedExperimentConfig(**config_dict)

        standard_results = {}
        for seed, result_list in data.get("standard_model_results", {}).items():
            standard_results[seed] = [
                TrainingResult(**result) if isinstance(result, dict) else result for result in result_list
            ]

        self_cond_results = {}
        for seed, result_list in data.get("self_conditioned_model_results", {}).items():
            self_cond_results[seed] = [
                TrainingResult(**result) if isinstance(result, dict) else result for result in result_list
            ]

        return cls(
            standard_model_results=standard_results,
            self_conditioned_model_results=self_cond_results,
            config=config,
            timestamp=data.get("timestamp", ""),
            training_time=data.get("training_time", {}),
            parameter_counts=data.get("parameter_counts", {}),
        )


def save_results(
    results: SelfConditionedExperimentResults,
    results_dir: Path,
    filename: Optional[str] = None,
) -> Path:
    """Save experiment results to JSON file."""
    results_dir.mkdir(parents=True, exist_ok=True)
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_results.json"
    save_path = results_dir / filename
    serializable_dict = results.to_dict()
    with open(save_path, "w") as f:
        json.dump(serializable_dict, f, indent=2)
    return save_path


def load_results(filepath: Path) -> SelfConditionedExperimentResults:
    """Load experiment results from JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)
    return SelfConditionedExperimentResults.from_dict(data)


def _convert_to_serializable(obj: Any) -> Any:
    """Recursively convert objects to JSON-serializable types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "dtype"):
        if np.issubdtype(obj.dtype, np.integer):
            return int(obj)
        elif np.issubdtype(obj.dtype, np.floating):
            return float(obj)
        elif np.issubdtype(obj.dtype, np.bool_):
            return bool(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if hasattr(obj, "__dict__"):
        return _convert_to_serializable(obj.__dict__)
    elif isinstance(obj, dict):
        return {key: _convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_serializable(item) for item in obj]
    else:
        return obj
