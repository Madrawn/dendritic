from dendritic.experiments.utils.PretrainingConfig import PretrainingConfig
from dendritic.experiments.utils.TrainingResult import TrainingResult


from dataclasses import dataclass
from typing import Any, Dict, List

@dataclass
class ExperimentResults:
    """Aggregated results from pretraining experiment."""
    model_results: Dict[str, List[TrainingResult]]
    statistical_analysis: Dict[str, Any]
    config: PretrainingConfig

    def get_model_results(self, model_name: str) -> List[TrainingResult]:
        """Get results for a specific model type."""
        if model_name not in self.model_results:
            raise ValueError(f"Unknown model type: {model_name}")
        return self.model_results[model_name]
