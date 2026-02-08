"""
Results data structures and serialization for doubt-aware experiments.

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
from dendritic.experiments.doubt.config import DoubtExperimentConfig


@dataclass
class DoubtTrainingResult(TrainingResult):
    """Extended training results with loss prediction metrics."""

    doubt_loss_history: List[float] = field(default_factory=list)
    token_loss_history: List[float] = field(default_factory=list)
    loss_predictions: List[float] = field(default_factory=list)
    actual_future_losses: List[float] = field(default_factory=list)


@dataclass
class DoubtExperimentResults:
    """Results from a doubt-aware experiment."""

    standard_model_results: Dict[str, List[TrainingResult]]  # By seed
    doubt_model_results: Dict[str, List[DoubtTrainingResult]]  # By seed
    config: DoubtExperimentConfig
    timestamp: str
    training_time: Dict[str, float]  # Training time per model type
    parameter_counts: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to a serializable dictionary."""
        # Convert dataclass to dict
        result_dict = asdict(self)

        # Convert config to dict
        result_dict["config"] = asdict(self.config)

        # Convert numpy types to Python native types
        return _convert_to_serializable(result_dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DoubtExperimentResults":
        """Create DoubtExperimentResults from a dictionary."""
        # Convert config dict back to DoubtExperimentConfig
        config_dict = data.get("config", {})
        config = DoubtExperimentConfig(**config_dict)

        # Reconstruct standard model results
        standard_results = {}
        for seed, result_list in data.get("standard_model_results", {}).items():
            standard_results[seed] = [
                TrainingResult(**result) if isinstance(result, dict) else result for result in result_list
            ]

        # Reconstruct doubt model results
        doubt_results = {}
        for seed, result_list in data.get("doubt_model_results", {}).items():
            doubt_results[seed] = [
                (DoubtTrainingResult(**result) if isinstance(result, dict) else result) for result in result_list
            ]

        # Reconstruct results
        return cls(
            standard_model_results=standard_results,
            doubt_model_results=doubt_results,
            config=config,
            timestamp=data.get("timestamp", ""),
            training_time=data.get("training_time", {}),
            parameter_counts=data.get("parameter_counts", {}),
        )


# Backward compatibility alias
DoubtTrainingResultWithPredictions = DoubtTrainingResult


def save_results(
    results: DoubtExperimentResults,
    results_dir: Path,
    filename: Optional[str] = None,
) -> Path:
    """
    Save experiment results to JSON file.

    Args:
        results: Experiment results to save
        results_dir: Directory to save results in
        filename: Optional filename (default: {timestamp}_results.json)

    Returns:
        Path to saved file
    """
    # Ensure results directory exists
    results_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_results.json"

    save_path = results_dir / filename

    # Convert to serializable dict
    serializable_dict = results.to_dict()

    # Save to JSON
    with open(save_path, "w") as f:
        json.dump(serializable_dict, f, indent=2)

    return save_path


def load_results(filepath: Path) -> DoubtExperimentResults:
    """
    Load experiment results from JSON file.

    Args:
        filepath: Path to JSON file containing results

    Returns:
        Loaded DoubtExperimentResults
    """
    with open(filepath, "r") as f:
        data = json.load(f)

    return DoubtExperimentResults.from_dict(data)


def _convert_to_serializable(obj: Any) -> Any:
    """
    Recursively convert objects to JSON-serializable types.

    Handles numpy types, dataclasses, and other non-serializable objects.
    """
    # Handle numpy arrays first (they also have dtype attribute)
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # Handle numpy scalar types
    if hasattr(obj, "dtype"):
        if np.issubdtype(obj.dtype, np.integer):
            return int(obj)
        elif np.issubdtype(obj.dtype, np.floating):
            return float(obj)
        elif np.issubdtype(obj.dtype, np.bool_):
            return bool(obj)

    # Handle other numpy types
    if isinstance(obj, np.generic):
        return obj.item()

    if hasattr(obj, "__dict__"):
        # Handle dataclasses and other objects with __dict__
        return _convert_to_serializable(obj.__dict__)
    elif isinstance(obj, dict):
        return {key: _convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_serializable(item) for item in obj]
    else:
        return obj


def create_results_filename(prefix: str = "doubt") -> str:
    """
    Create a standardized filename for results.

    Args:
        prefix: Optional prefix for the filename

    Returns:
        Filename in format: {prefix}_{timestamp}.json
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.json"
