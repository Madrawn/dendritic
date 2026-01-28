"""
Utility for generating scheduler variant configurations for pretraining experiments.
"""

import itertools
from copy import deepcopy
import dataclasses
from typing import Union, get_origin, get_args, get_type_hints

from pytest import param
from .PretrainingConfig import PretrainingConfig, CohortSchedulerConfig


def generate_scheduler_variants(
    base_config: PretrainingConfig, param_grid: dict[str, list]
) -> list[PretrainingConfig]:
    """
    Generate a list of PretrainingConfig variants by sweeping over parameters.

    Parameters
    ----------
    base_config : PretrainingConfig
        The base configuration to copy for each variant.
    param_grid : dict
        Mapping of field names to lists of values to sweep over.
        Field names can be dot‑separated to target nested attributes.
        Example: {"dropout": [0.0, 0.3], "cohort_scheduler.min_mult": [0.2, 0.3]}

    Returns
    -------
    list[PretrainingConfig]
        List of configuration objects, each with distinct parameter values.
    """

    keys, values = zip(*param_grid.items())
    variants = []
    params = {}
    for combo in itertools.product(*values):
        cfg = deepcopy(base_config)
        for key, val in zip(keys, combo):
            if val is None:
                continue
            assert isinstance(key, str)
            # set_nested_attr(cfg, key, val)
            cfg_field = key.split(".", 1)[0]
            sub_field = key.split(".", 1)[1] if "." in key else None
            if sub_field:
                cfg.set_deep(cfg_field, sub_field, val)

            else:
                setattr(cfg, cfg_field, val)
            params[key] = val

        cfg.param_grid = deepcopy(params)
        variants.append(cfg)

    return variants


def variant_identifier(param_grid: dict[str, object] | None) -> str:
    """Generate a concise identifier based on the param grid.

    Parameters
    ----------
    param_grid : dict[str, object] | None
        Parameter grid mapping parameter names to values.

    Returns
    -------
    str
        Identifier string like "dropout:0.0-layer_type:standard".
        Returns "baseline" if param_grid is None or empty.
    """
    if not param_grid:
        return "baseline"
    return "-".join(f"{k}:{v}" for k, v in param_grid.items())
