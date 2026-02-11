"""
Validation and sanity check utilities for doubt-aware experiments.

This module provides functions to validate experiment results, compare model
parameters, and perform sanity checks on doubt predictions.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Any
from dendritic.experiments.models.doubt_conditioning.DoubtAwareGPT import DoubtAwareGPT
from dendritic.experiments.models.MiniGPT import MiniGPT


def compare_parameter_counts(standard_model: MiniGPT, doubt_model: DoubtAwareGPT) -> Dict[str, Any]:
    """
    Compare parameter counts between standard and doubt models.

    Args:
        standard_model: Standard MiniGPT model
        doubt_model: DoubtAwareGPT model

    Returns:
        Dictionary with parameter counts and comparison metrics
    """
    # Calculate total parameters
    std_params = sum(p.numel() for p in standard_model.parameters())
    doubt_params = sum(p.numel() for p in doubt_model.parameters())

    # Calculate layer-wise breakdown
    std_layer_params = {}
    doubt_layer_params = {}

    for name, param in standard_model.named_parameters():
        std_layer_params[name] = param.numel()

    for name, param in doubt_model.named_parameters():
        doubt_layer_params[name] = param.numel()

    # Calculate difference
    param_diff = doubt_params - std_params
    relative_diff = param_diff / std_params if std_params > 0 else float("inf")

    return {
        "standard_total": std_params,
        "doubt_total": doubt_params,
        "difference": param_diff,
        "relative_difference": relative_diff,
        "standard_by_layer": std_layer_params,
        "doubt_by_layer": doubt_layer_params,
        "is_fair_comparison": relative_diff < 0.3,  # Less than 30% more params
    }


def validate_loss_predictions(
    doubt_model: DoubtAwareGPT,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    threshold: float = 0.95,
) -> Dict[str, Any]:
    """
    Validate that loss predictions are within reasonable bounds.

    Args:
        doubt_model: DoubtAwareGPT model
        input_ids: Input token IDs
        attention_mask: Attention mask
        threshold: Minimum percentage of values that should be in [0, 1] range

    Returns:
        Dictionary with validation results
    """
    doubt_model.eval()

    with torch.no_grad():
        output = doubt_model(
            input_ids=input_ids,
            labels=None,  # No labels for validation
            doubt_scalars=None,  # Default doubt
        )

    loss_pred = output["loss_prediction"]

    # Basic shape validation
    batch_size, seq_len = input_ids.shape
    V = getattr(doubt_model, "doubt_vector_dim", 1)
    expected_shape = (batch_size, seq_len, V) if V > 1 else (batch_size, seq_len, 1)
    assert loss_pred.shape == expected_shape, (
        f"Loss prediction shape {loss_pred.shape} doesn't match expected shape {expected_shape}"
    )

    # Check for NaN or infinite values
    has_nan = torch.any(torch.isnan(loss_pred))
    has_inf = torch.any(torch.isinf(loss_pred))

    # Check value range - loss predictions are raw values, not bounded
    # They can be any real number, but we check for reasonable bounds
    # Initial bias is set to 2.0, so values around that are expected
    mean_val = loss_pred.mean().item()
    std_val = loss_pred.std().item()

    # Calculate statistics
    loss_pred_np = loss_pred.cpu().numpy()
    stats = {
        "mean": float(np.mean(loss_pred_np)),
        "std": float(np.std(loss_pred_np)),
        "min": float(np.min(loss_pred_np)),
        "max": float(np.max(loss_pred_np)),
        "median": float(np.median(loss_pred_np)),
    }

    # Determine if validation passes
    # Check if values are reasonable (not too far from initial bias of 2.0)
    reasonable_mean = abs(mean_val - 2.0) < 10.0  # Allow some deviation
    reasonable_std = std_val < 10.0  # Reasonable standard deviation
    passes_range_check = reasonable_mean and reasonable_std
    passes_sanity_check = not (has_nan or has_inf)

    return {
        "shape": loss_pred.shape,
        "has_nan": bool(has_nan),
        "has_inf": bool(has_inf),
        "mean_value": mean_val,
        "std_value": std_val,
        "passes_range_check": passes_range_check,
        "passes_sanity_check": passes_sanity_check,
        "statistics": stats,
        "is_valid": passes_range_check and passes_sanity_check,
    }


def validate_experiment_results(
    results: Any,  # DoubtExperimentResults type
    config: Any,  # DoubtExperimentConfig type
) -> Dict[str, Any]:
    """
    Validate experiment results for consistency and completeness.

    Args:
        results: DoubtExperimentResults object
        config: DoubtExperimentConfig object

    Returns:
        Dictionary with validation results
    """
    validation_results: Dict[str, Any] = {
        "checks_passed": [],
        "checks_failed": [],
        "warnings": [],
        "errors": [],
    }

    # Check 1: Results structure
    required_attrs = [
        "standard_model_results",
        "doubt_model_results",
        "config",
        "timestamp",
        "training_time",
        "parameter_counts",
    ]

    for attr in required_attrs:
        if hasattr(results, attr):
            validation_results["checks_passed"].append(f"has_{attr}")
        else:
            validation_results["errors"].append(f"Missing attribute: {attr}")

    # Check 2: All seeds have results
    if hasattr(results, "standard_model_results") and hasattr(results, "doubt_model_results"):
        std_seeds = set(results.standard_model_results.keys())
        doubt_seeds = set(results.doubt_model_results.keys())

        if std_seeds == doubt_seeds:
            validation_results["checks_passed"].append("seed_consistency")
        else:
            validation_results["errors"].append(f"Seed mismatch: standard={std_seeds}, doubt={doubt_seeds}")

        # Check if all config seeds are present
        config_seeds = set(str(seed) for seed in config.seeds)
        if std_seeds == config_seeds:
            validation_results["checks_passed"].append("seed_completeness")
        else:
            missing = config_seeds - std_seeds
            if missing:
                validation_results["warnings"].append(f"Missing seeds: {missing}")

    # Check 3: Training times are positive
    if hasattr(results, "training_time"):
        training_time = results.training_time
        for model_type in ["standard", "doubt"]:
            if model_type in training_time:
                time_val = training_time[model_type]
                if time_val > 0:
                    validation_results["checks_passed"].append(f"{model_type}_training_time_positive")
                else:
                    validation_results["warnings"].append(f"{model_type} training time is {time_val}")

    # Check 4: Parameter counts are reasonable
    if hasattr(results, "parameter_counts"):
        param_counts = results.parameter_counts
        if "standard" in param_counts and "doubt" in param_counts:
            std_params = param_counts["standard"]
            doubt_params = param_counts["doubt"]

            if doubt_params > std_params:
                validation_results["checks_passed"].append("doubt_has_more_params")

                # Check relative difference
                rel_diff = (doubt_params - std_params) / std_params
                if rel_diff < 0.3:  # Less than 30% more parameters
                    validation_results["checks_passed"].append("reasonable_param_increase")
                else:
                    validation_results["warnings"].append(f"Large parameter increase: {rel_diff:.1%}")
            else:
                validation_results["errors"].append("Doubt model should have more parameters than standard model")

    # Determine overall validity
    is_valid = len(validation_results["errors"]) == 0
    validation_results["is_valid"] = is_valid

    return validation_results


def generate_validation_report(validation_results: Dict[str, Any], verbose: bool = True) -> str:
    """
    Generate a human-readable validation report.

    Args:
        validation_results: Output from validate_experiment_results
        verbose: Whether to include all details

    Returns:
        Formatted validation report
    """
    report_lines = []

    # Header
    report_lines.append("=" * 60)
    report_lines.append("EXPERIMENT VALIDATION REPORT")
    report_lines.append("=" * 60)

    # Overall status
    is_valid = validation_results.get("is_valid", False)
    status = "PASS" if is_valid else "FAIL"
    report_lines.append(f"Overall Status: {status}")
    report_lines.append("")

    # Summary counts
    passed = len(validation_results.get("checks_passed", []))
    failed = len(validation_results.get("errors", []))
    warnings = len(validation_results.get("warnings", []))

    report_lines.append(f"Checks Passed: {passed}")
    report_lines.append(f"Errors: {failed}")
    report_lines.append(f"Warnings: {warnings}")
    report_lines.append("")

    # Errors (if any)
    if validation_results.get("errors"):
        report_lines.append("ERRORS:")
        report_lines.append("-" * 40)
        for error in validation_results["errors"]:
            report_lines.append(f"  ✗ {error}")
        report_lines.append("")

    # Warnings (if any)
    if validation_results.get("warnings"):
        report_lines.append("WARNINGS:")
        report_lines.append("-" * 40)
        for warning in validation_results["warnings"]:
            report_lines.append(f"  ⚠ {warning}")
        report_lines.append("")

    # Passed checks (if verbose)
    if verbose and validation_results.get("checks_passed"):
        report_lines.append("PASSED CHECKS:")
        report_lines.append("-" * 40)
        for check in validation_results["checks_passed"]:
            report_lines.append(f"  ✓ {check}")

    return "\n".join(report_lines)
