"""
Tests for the flexible cohort scheduler refactor.
"""

from unittest import mock

import pytest

from dendritic.experiments.utils.PretrainingConfig import (
    PretrainingConfig,
    CohortSchedulerConfig,
)
from dendritic.experiments.utils.sweep import generate_scheduler_variants
from dendritic.experiments.run_experiments import run_pretraining_experiment


@pytest.mark.unit
def test_pretraining_config_cohort_scheduler():
    """Ensure PretrainingConfig can store a CohortSchedulerConfig."""
    scheduler_cfg = CohortSchedulerConfig(
        min_mult=0.4, max_mult=0.9, device="cpu", apply_to_gradients=False
    )
    cfg = PretrainingConfig(
        training_steps=10,
        batch_size=2,
        seeds=[1],
        cohort_scheduler=scheduler_cfg,
    )
    assert cfg.cohort_scheduler is not None
    assert cfg.cohort_scheduler.min_mult == 0.4
    assert cfg.cohort_scheduler.max_mult == 0.9
    assert cfg.cohort_scheduler.device == "cpu"
    assert cfg.cohort_scheduler.apply_to_gradients is False


@pytest.mark.unit
def test_generate_scheduler_variants():
    """Check that the sweep utility creates the correct number of variants."""
    base_cfg = PretrainingConfig(
        training_steps=10,
        batch_size=2,
        seeds=[1],
    )
    assert base_cfg.cohort_scheduler is None # Should be None by default
    param_grid = {"cohort_scheduler.min_mult": [0.4, 0.5], "cohort_scheduler.max_mult": [0.9, 1.0]}
    variants = generate_scheduler_variants(base_cfg, param_grid)

    # 2 * 2 = 4 combinations
    assert len(variants) == 4
    for v in variants:
        assert isinstance(v.cohort_scheduler, CohortSchedulerConfig)
        # Ensure the values come from the grid
        assert v.cohort_scheduler.min_mult in param_grid["cohort_scheduler.min_mult"]
        assert v.cohort_scheduler.max_mult in param_grid["cohort_scheduler.max_mult"]




@pytest.mark.unit
def test_run_pretraining_experiment_multiple_variants(monkeypatch):
    """
    Run the experiment with multiple configs using a mocked PretrainingExperiment
    to avoid heavy training.
    """
    import torch
    # Minimal base config
    base_cfg = PretrainingConfig(
        training_steps=1,
        batch_size=1,
        seeds=[1],
    )
    param_grid = {"cohort_scheduler.min_mult": [0.4], "cohort_scheduler.max_mult": [0.9]}
    variants = generate_scheduler_variants(base_cfg, param_grid)

    # Mock ExperimentResults to return a simple object
    dummy_result = mock.MagicMock()
    dummy_result.final_eval_loss = 0.0
    # Set config to avoid recursion during serialization
    dummy_result.config = variants[0]

    # Create a mock model with parameters to avoid empty optimizer
    def create_mock_model():
        model = mock.MagicMock()
        # Add a dummy parameter
        dummy_param = torch.nn.Parameter(torch.tensor(1.0))
        model.parameters = mock.Mock(return_value=[dummy_param])
        model.named_parameters = mock.Mock(return_value=[("dummy", dummy_param)])
        return model

    # Mock PretrainingExperiment to bypass actual training
    class DummyExperiment:
        def __init__(self, config):
            self.config = config

        def run(self, *args, **kwargs):
            return dummy_result

        def create_models(self):
            # Return dummy models with parameters
            return (create_mock_model(), create_mock_model(), create_mock_model(), create_mock_model())

    monkeypatch.setattr(
        "dendritic.experiments.utils.experiment_pretraining.PretrainingExperiment",
        DummyExperiment,
    )
    
    # Mock torch.optim.AdamW to avoid empty parameter list error
    mock_optimizer = mock.MagicMock()
    def mock_adamw(params, **kwargs):
        if len(params) == 0:
            # If params empty, add a dummy parameter
            params = [torch.nn.Parameter(torch.tensor(1.0))]
        return mock_optimizer
    monkeypatch.setattr(torch.optim, "AdamW", mock_adamw)

    # Mock load_pretraining_data to avoid real dataset loading
    def mock_load_pretraining_data(tokenizer, config, max_length=256, num_workers=None):
        # Create a dummy dataset with a single batch
        class DummyDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 10  # non-zero
            def __getitem__(self, idx):
                return {
                    "input_ids": torch.randint(0, config.vocab_size, (max_length,)),
                    "labels": torch.randint(0, config.vocab_size, (max_length,)),
                }
        train_dataset = DummyDataset()
        eval_dataset = DummyDataset()
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=config.batch_size, shuffle=True
        )
        eval_dataloader = torch.utils.data.DataLoader(
            eval_dataset, batch_size=config.batch_size, shuffle=False
        )
        return train_dataloader, eval_dataloader
    
    monkeypatch.setattr(
        "dendritic.experiments.run_experiments.load_pretraining_data",
        mock_load_pretraining_data,
    )
    
    # Mock analysis functions to avoid side effects
    monkeypatch.setattr(
        "dendritic.experiments.analysis.analysis.save_consolidated_results",
        mock.MagicMock(),
    )
    monkeypatch.setattr(
        "dendritic.experiments.analysis.analysis.print_consolidated_summary",
        mock.MagicMock(),
    )

    results = run_pretraining_experiment(device="cpu", scheduler_variants=variants)

    # Should contain two entries: baseline (no scheduler) and the scheduler variant
    assert isinstance(results, dict)
    assert len(results) == 2
    expected_keys = {"baseline", "min0.4_max0.9_sharp1.0"}
    assert set(results.keys()) == expected_keys
    for key in expected_keys:
        assert results[key] is dummy_result