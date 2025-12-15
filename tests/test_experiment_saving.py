import json
import numpy as np
import pytest
from dendritic.experiments.analysis import save_experiment_results
from pathlib import Path

from dendritic.experiments.ExperimentResults import ExperimentResults
from dendritic.experiments.PretrainingConfig import PretrainingConfig
from dendritic.experiments.TrainingResult import TrainingResult

def test_save_experiment_results_handles_numpy_types(tmp_path):
    """Test that save_experiment_results handles numpy types correctly."""
    # Create dummy data with numpy types that caused the original error
    config = PretrainingConfig()
    statistical_analysis = {
        "comparison": {
            "significant_005": np.bool_(True),
            "significant_001": np.bool_(False)
        }
    }
    
    results = ExperimentResults(
        model_results={
            "baseline": [],
            "dendritic": []
        },
        statistical_analysis=statistical_analysis,
        config=config
    )
    
    # This should not raise TypeError
    save_experiment_results(results, Path(tmp_path))
    
    # Verify the saved file
    output_file = list(Path(tmp_path).glob("*.json"))[0]
    with open(output_file, "r") as f:
        data = json.load(f)
        assert data["statistical_analysis"]["comparison"]["significant_005"] is True
        assert data["statistical_analysis"]["comparison"]["significant_001"] is False