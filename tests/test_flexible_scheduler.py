"""
Tests for the flexible cohort scheduler refactor.
"""

import builtins
from copy import deepcopy
from unittest import mock

import pytest

from dendritic.experiments.utils.PretrainingConfig import (
    PretrainingConfig,
    CohortSchedulerConfig,
)
from dendritic.experiments.utils.sweep import generate_scheduler_variants
from dendritic.experiments.run_experiments import _variant_identifier, run_pretraining_experiment


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


def test_generate_scheduler_variants():
    """Check that the sweep utility creates the correct number of variants."""
    base_cfg = PretrainingConfig(
        training_steps=10,
        batch_size=2,
        seeds=[1],
    )
    param_grid = {"min_mult": [0.4, 0.5], "max_mult": [0.9, 1.0]}
    variants = generate_scheduler_variants(base_cfg, param_grid)

    # 2 * 2 = 4 combinations
    assert len(variants) == 4
    for v in variants:
        assert isinstance(v.cohort_scheduler, CohortSchedulerConfig)
        # Ensure the values come from the grid
        assert v.cohort_scheduler.min_mult in param_grid["min_mult"]
        assert v.cohort_scheduler.max_mult in param_grid["max_mult"]


def test_variant_identifier():
    """Validate the human‑readable identifier generation."""
    cfg_no_sched = PretrainingConfig(training_steps=10, batch_size=2, seeds=[1])
    assert _variant_identifier(cfg_no_sched) == "no_scheduler"

    scheduler_cfg = CohortSchedulerConfig(min_mult=0.4, max_mult=0.9)
    cfg_with_sched = deepcopy(cfg_no_sched)
    cfg_with_sched.cohort_scheduler = scheduler_cfg
    assert _variant_identifier(cfg_with_sched) == "min0.4_max0.9"


def test_run_pretraining_experiment_multiple_variants(monkeypatch):
    """
    Run the experiment with multiple configs using a mocked PretrainingExperiment
    to avoid heavy training.
    """
    # Minimal base config
    base_cfg = PretrainingConfig(
        training_steps=1,
        batch_size=1,
        seeds=[1],
    )
    param_grid = {"min_mult": [0.4], "max_mult": [0.9]}
    variants = generate_scheduler_variants(base_cfg, param_grid)

    # Mock ExperimentResults to return a simple object
    dummy_result = mock.MagicMock()
    dummy_result.final_eval_loss = 0.0

    # Mock PretrainingExperiment to bypass actual training
    class DummyExperiment:
        def __init__(self, config):
            self.config = config

        def run(self, *args, **kwargs):
            return dummy_result

        def create_models(self):
            # Return dummy models (the actual values are not used)
            return (mock.MagicMock(), mock.MagicMock(), mock.MagicMock(), mock.MagicMock())

    monkeypatch.setattr(
        "dendritic.experiments.utils.experiment_pretraining.PretrainingExperiment",
        DummyExperiment,
    )

    results = run_pretraining_experiment(device="cpu", scheduler_variants=variants)

    # Should contain a single entry because we only generated one variant
    assert isinstance(results, dict)
    assert len(results) == 1
    key = next(iter(results))
    assert key == "min0.4_max0.9"
    assert results[key] is dummy_result