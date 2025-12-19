import json
import numpy as np
import pytest
from pathlib import Path
from dendritic.experiments.utils.ExperimentResults import ExperimentResults
from dendritic.experiments.utils.PretrainingConfig import PretrainingConfig, CohortSchedulerConfig
from dendritic.experiments.analysis.analysis import save_experiment_results


@pytest.mark.unit
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


@pytest.mark.unit
def test_save_experiment_results_with_cohort_scheduler(tmp_path):
    """Test that save_experiment_results can serialize config with CohortSchedulerConfig."""
    config = PretrainingConfig(
        cohort_scheduler=CohortSchedulerConfig(min_mult=0.3, max_mult=1.0, sharpness=2.0)
    )
    statistical_analysis = {
        "baseline": {
            "final_ppl_mean": 100.0,
            "final_ppl_std": 5.0,
        }
    }
    results = ExperimentResults(
        model_results={
            "baseline": [],
        },
        statistical_analysis=statistical_analysis,
        config=config
    )
    # This should not raise TypeError
    save_experiment_results(results, Path(tmp_path))
    # Verify the saved file contains cohort_scheduler
    output_file = list(Path(tmp_path).glob("*.json"))[0]
    with open(output_file, "r") as f:
        data = json.load(f)
        assert "cohort_scheduler" in data["config"]
        # Ensure values are present
        cohort = data["config"]["cohort_scheduler"]
        assert cohort["min_mult"] == 0.3
        assert cohort["max_mult"] == 1.0
        assert cohort["sharpness"] == 2.0