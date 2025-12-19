"""
Utility for generating scheduler variant configurations for pretraining experiments.
"""

import itertools
from copy import deepcopy
from .PretrainingConfig import PretrainingConfig, CohortSchedulerConfig

def generate_scheduler_variants(base_config: PretrainingConfig, param_grid: dict) -> list[PretrainingConfig]:
    """
    Generate a list of PretrainingConfig variants with different CohortSchedulerConfig parameters.

    Parameters
    ----------
    base_config : PretrainingConfig
        The base configuration to copy for each variant.
    param_grid : dict
        Mapping of CohortSchedulerConfig field names to lists of values to sweep over.
        Example: {"min_mult": [0.4, 0.5], "max_mult": [0.9, 1.0]}

    Returns
    -------
    list[PretrainingConfig]
        List of configuration objects, each with a distinct CohortSchedulerConfig.
    """
    # Extract keys and corresponding value lists
    keys, values = zip(*param_grid.items())
    variants: list[PretrainingConfig] = []

    # Iterate over the Cartesian product of all parameter values
    for combo in itertools.product(*values):
        cfg = deepcopy(base_config)
        scheduler_cfg = CohortSchedulerConfig(**dict(zip(keys, combo)))
        cfg.cohort_scheduler = scheduler_cfg
        variants.append(cfg)

    return variants