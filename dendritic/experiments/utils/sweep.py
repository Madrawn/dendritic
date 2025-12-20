"""
Utility for generating scheduler variant configurations for pretraining experiments.
"""

import itertools
from copy import deepcopy
import dataclasses
from typing import Union, get_origin, get_args, get_type_hints
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

    for combo in itertools.product(*values):
        cfg = deepcopy(base_config)
        for key, val in zip(keys, combo):
            assert isinstance(key, str)
            # set_nested_attr(cfg, key, val)
            cfg_field = key.split(".", 1)[0]
            sub_field = key.split(".", 1)[1] if "." in key else None
            if sub_field:
                cfg.set_deep(cfg_field, sub_field, val)
            else:
                setattr(cfg, cfg_field, val)
        variants.append(cfg)

    return variants
