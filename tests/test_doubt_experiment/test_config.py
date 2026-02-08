"""
Unit tests for DoubtExperimentConfig.
"""

import pytest
import json
from dataclasses import asdict

from dendritic.experiments.doubt.config import DoubtExperimentConfig
from dendritic.experiments.utils.PretrainingConfig import (
    PretrainingConfig,
    CohortSchedulerConfig,
)


@pytest.mark.unit
def test_config_instantiation_defaults():
    """Test that DoubtExperimentConfig can be instantiated with default values."""
    config = DoubtExperimentConfig()

    # Check that default values are set
    assert config.doubt_alpha == 1.0
    assert config.lookahead_steps == 2
    assert config.doubt_init_bias == 2.0
    assert config.results_dir == "results/doubt_experiments"
    assert config.save_interval == 100

    # Check that parent class defaults are inherited
    assert config.vocab_size == 50257
    assert config.embed_dim == 384
    assert config.max_seq_len == 1024
    assert config.dataset == "openwebmath"


@pytest.mark.unit
def test_config_custom_values():
    """Test that DoubtExperimentConfig accepts custom values."""
    config = DoubtExperimentConfig(
        doubt_alpha=0.5,
        lookahead_steps=3,
        doubt_init_bias=1.5,
        results_dir="custom/results",
        save_interval=50,
        vocab_size=10000,
        embed_dim=256,
        max_seq_len=512,
        dataset="wikitext",
    )

    assert config.doubt_alpha == 0.5
    assert config.lookahead_steps == 3
    assert config.doubt_init_bias == 1.5
    assert config.results_dir == "custom/results"
    assert config.save_interval == 50
    assert config.vocab_size == 10000
    assert config.embed_dim == 256
    assert config.max_seq_len == 512
    assert config.dataset == "wikitext"


@pytest.mark.unit
def test_config_inheritance():
    """Test that DoubtExperimentConfig properly inherits from PretrainingConfig."""
    config = DoubtExperimentConfig()

    # Check isinstance
    assert isinstance(config, PretrainingConfig)

    # Check that all parent fields are accessible
    parent_fields = {field.name for field in PretrainingConfig.__dataclass_fields__.values()}
    child_fields = {field.name for field in DoubtExperimentConfig.__dataclass_fields__.values()}

    # All parent fields should be in child fields
    assert parent_fields.issubset(child_fields)

    # Child should have additional fields
    additional_fields = child_fields - parent_fields
    expected_additional = {
        "doubt_alpha",
        "lookahead_steps",
        "doubt_init_bias",
        "results_dir",
        "save_interval",
        "sampling_top_p",
        "sampling_prompt",
        "sampling_max_tokens",
        "sampling_temperature",
        "eval_smoothing_factor",
    }
    assert additional_fields == expected_additional


@pytest.mark.unit
def test_config_post_init():
    """Test that __post_init__ sets default results_dir if empty."""
    # Test with empty results_dir
    config = DoubtExperimentConfig(results_dir="")
    assert config.results_dir == "results/doubt_experiments"

    # Test with non-empty results_dir
    config = DoubtExperimentConfig(results_dir="my/results")
    assert config.results_dir == "my/results"

    # Test that parent __post_init__ is called
    config = DoubtExperimentConfig()
    assert config.eval_interval == max(config.training_steps // config.training_steps_factor, 1)


@pytest.mark.unit
def test_config_serialization():
    """Test that DoubtExperimentConfig can be serialized to JSON."""
    config = DoubtExperimentConfig(
        doubt_alpha=0.7,
        lookahead_steps=2,
        doubt_init_bias=1.8,
        results_dir="test/results",
        save_interval=200,
    )

    # Convert to dict
    config_dict = asdict(config)

    # Check that all fields are present and have correct types
    assert isinstance(config_dict, dict)
    assert config_dict["doubt_alpha"] == 0.7
    assert config_dict["lookahead_steps"] == 2
    assert config_dict["doubt_init_bias"] == 1.8
    assert config_dict["results_dir"] == "test/results"
    assert config_dict["save_interval"] == 200

    # Convert to JSON (should not raise)
    json_str = json.dumps(config_dict)
    assert isinstance(json_str, str)

    # Parse back
    parsed_dict = json.loads(json_str)
    assert parsed_dict["doubt_alpha"] == 0.7


@pytest.mark.unit
def test_config_cohort_scheduler():
    """Test that cohort_scheduler field works correctly."""
    # Test with None (default)
    config = DoubtExperimentConfig()
    assert config.cohort_scheduler is None

    # Test with CohortSchedulerConfig
    cohort_config = CohortSchedulerConfig(
        min_mult=0.3,
        max_mult=1.2,
        sharpness=2.0,
        device="cuda",
        apply_to_gradients=False,
    )
    config = DoubtExperimentConfig(cohort_scheduler=cohort_config)
    assert config.cohort_scheduler is not None
    assert config.cohort_scheduler.min_mult == 0.3
    assert config.cohort_scheduler.max_mult == 1.2
    assert config.cohort_scheduler.sharpness == 2.0
    assert config.cohort_scheduler.device == "cuda"
    assert config.cohort_scheduler.apply_to_gradients is False


@pytest.mark.unit
def test_config_validation():
    """Test that doubt-specific parameters accept various values."""
    # Test that lookahead_steps accepts zero (no validation currently)
    config = DoubtExperimentConfig(lookahead_steps=0)
    assert config.lookahead_steps == 0

    # Test that lookahead_steps accepts negative values (no validation currently)
    config = DoubtExperimentConfig(lookahead_steps=-1)
    assert config.lookahead_steps == -1

    # Test that doubt_alpha can be zero
    config = DoubtExperimentConfig(doubt_alpha=0.0)
    assert config.doubt_alpha == 0.0

    # Test negative doubt_alpha (should be allowed, though unusual)
    config = DoubtExperimentConfig(doubt_alpha=-0.5)
    assert config.doubt_alpha == -0.5


@pytest.mark.unit
def test_config_field_types():
    """Test that field types are correct."""
    config = DoubtExperimentConfig()

    # Check types of doubt-specific fields
    assert isinstance(config.doubt_alpha, float)
    assert isinstance(config.lookahead_steps, int)
    assert isinstance(config.doubt_init_bias, float)
    assert isinstance(config.results_dir, str)
    assert isinstance(config.save_interval, int)

    # Check parent field types
    assert isinstance(config.vocab_size, int)
    assert isinstance(config.embed_dim, int)
    assert isinstance(config.max_seq_len, int)
    assert isinstance(config.dataset, str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
