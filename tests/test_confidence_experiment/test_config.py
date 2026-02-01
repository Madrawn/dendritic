"""
Unit tests for ConfidenceExperimentConfig.
"""

import pytest
import json
from dataclasses import asdict

from dendritic.experiments.confidence.config import ConfidenceExperimentConfig
from dendritic.experiments.utils.PretrainingConfig import (
    PretrainingConfig,
    CohortSchedulerConfig,
)


@pytest.mark.unit
def test_config_instantiation_defaults():
    """Test that ConfidenceExperimentConfig can be instantiated with default values."""
    config = ConfidenceExperimentConfig()

    # Check that default values are set
    assert config.confidence_alpha == 1.0
    assert config.lookahead_steps == 2
    assert config.confidence_init_bias == 2.0
    assert config.results_dir == "results/confidence_experiments"
    assert config.save_interval == 100

    # Check that parent class defaults are inherited
    assert config.vocab_size == 50257
    assert config.embed_dim == 384
    assert config.max_seq_len == 1024
    assert config.dataset == "openwebmath"


@pytest.mark.unit
def test_config_custom_values():
    """Test that ConfidenceExperimentConfig accepts custom values."""
    config = ConfidenceExperimentConfig(
        confidence_alpha=0.5,
        lookahead_steps=3,
        confidence_init_bias=1.5,
        results_dir="custom/results",
        save_interval=50,
        vocab_size=10000,
        embed_dim=256,
        max_seq_len=512,
        dataset="wikitext",
    )

    assert config.confidence_alpha == 0.5
    assert config.lookahead_steps == 3
    assert config.confidence_init_bias == 1.5
    assert config.results_dir == "custom/results"
    assert config.save_interval == 50
    assert config.vocab_size == 10000
    assert config.embed_dim == 256
    assert config.max_seq_len == 512
    assert config.dataset == "wikitext"


@pytest.mark.unit
def test_config_inheritance():
    """Test that ConfidenceExperimentConfig properly inherits from PretrainingConfig."""
    config = ConfidenceExperimentConfig()

    # Check isinstance
    assert isinstance(config, PretrainingConfig)

    # Check that all parent fields are accessible
    parent_fields = {
        field.name for field in PretrainingConfig.__dataclass_fields__.values()
    }
    child_fields = {
        field.name for field in ConfidenceExperimentConfig.__dataclass_fields__.values()
    }

    # All parent fields should be in child fields
    assert parent_fields.issubset(child_fields)

    # Child should have additional fields
    additional_fields = child_fields - parent_fields
    expected_additional = {
        "confidence_alpha",
        "lookahead_steps",
        "confidence_init_bias",
        "results_dir",
        "save_interval",
    }
    assert additional_fields == expected_additional


@pytest.mark.unit
def test_config_post_init():
    """Test that __post_init__ sets default results_dir if empty."""
    # Test with empty results_dir
    config = ConfidenceExperimentConfig(results_dir="")
    assert config.results_dir == "results/confidence_experiments"

    # Test with non-empty results_dir
    config = ConfidenceExperimentConfig(results_dir="my/results")
    assert config.results_dir == "my/results"

    # Test that parent __post_init__ is called
    config = ConfidenceExperimentConfig()
    assert config.eval_interval == max(config.training_steps // 20, 1)


@pytest.mark.unit
def test_config_serialization():
    """Test that ConfidenceExperimentConfig can be serialized to JSON."""
    config = ConfidenceExperimentConfig(
        confidence_alpha=0.7,
        lookahead_steps=2,
        confidence_init_bias=1.8,
        results_dir="test/results",
        save_interval=200,
    )

    # Convert to dict
    config_dict = asdict(config)

    # Check that all fields are present and have correct types
    assert isinstance(config_dict, dict)
    assert config_dict["confidence_alpha"] == 0.7
    assert config_dict["lookahead_steps"] == 2
    assert config_dict["confidence_init_bias"] == 1.8
    assert config_dict["results_dir"] == "test/results"
    assert config_dict["save_interval"] == 200

    # Convert to JSON (should not raise)
    json_str = json.dumps(config_dict)
    assert isinstance(json_str, str)

    # Parse back
    parsed_dict = json.loads(json_str)
    assert parsed_dict["confidence_alpha"] == 0.7


@pytest.mark.unit
def test_config_cohort_scheduler():
    """Test that cohort_scheduler field works correctly."""
    # Test with None (default)
    config = ConfidenceExperimentConfig()
    assert config.cohort_scheduler is None

    # Test with CohortSchedulerConfig
    cohort_config = CohortSchedulerConfig(
        min_mult=0.3,
        max_mult=1.2,
        sharpness=2.0,
        device="cuda",
        apply_to_gradients=False,
    )
    config = ConfidenceExperimentConfig(cohort_scheduler=cohort_config)
    assert config.cohort_scheduler is not None
    assert config.cohort_scheduler.min_mult == 0.3
    assert config.cohort_scheduler.max_mult == 1.2
    assert config.cohort_scheduler.sharpness == 2.0
    assert config.cohort_scheduler.device == "cuda"
    assert config.cohort_scheduler.apply_to_gradients is False


@pytest.mark.unit
def test_config_validation():
    """Test that confidence-specific parameters accept various values."""
    # Test that lookahead_steps accepts zero (no validation currently)
    config = ConfidenceExperimentConfig(lookahead_steps=0)
    assert config.lookahead_steps == 0

    # Test that lookahead_steps accepts negative values (no validation currently)
    config = ConfidenceExperimentConfig(lookahead_steps=-1)
    assert config.lookahead_steps == -1

    # Test that confidence_alpha can be zero
    config = ConfidenceExperimentConfig(confidence_alpha=0.0)
    assert config.confidence_alpha == 0.0

    # Test negative confidence_alpha (should be allowed, though unusual)
    config = ConfidenceExperimentConfig(confidence_alpha=-0.5)
    assert config.confidence_alpha == -0.5


@pytest.mark.unit
def test_config_field_types():
    """Test that field types are correct."""
    config = ConfidenceExperimentConfig()

    # Check types of confidence-specific fields
    assert isinstance(config.confidence_alpha, float)
    assert isinstance(config.lookahead_steps, int)
    assert isinstance(config.confidence_init_bias, float)
    assert isinstance(config.results_dir, str)
    assert isinstance(config.save_interval, int)

    # Check parent field types
    assert isinstance(config.vocab_size, int)
    assert isinstance(config.embed_dim, int)
    assert isinstance(config.max_seq_len, int)
    assert isinstance(config.dataset, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
