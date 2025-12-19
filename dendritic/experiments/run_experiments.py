# dendritic/experiments/run_experiments.py
"""
Main script to run both .

Usage:
    python -m dendritic.experiments.run_experiments --experiment pretraining
    python -m dendritic.experiments.run_experiments --experiment finetuning
    python -m dendritic.experiments.run_experiments --experiment both
"""

# Standard library imports
import argparse
import logging
from math import exp
import multiprocessing
import os
from datetime import datetime
from pathlib import Path
import random
import numpy as np

# Third‑party imports
import torch
from torch.utils.data import DataLoader
from transformers.models.gpt2 import GPT2Tokenizer

# Project‑specific imports
from dendritic.experiments.utils.PretrainingConfig import PretrainingConfig
from dendritic.experiments.utils.ExperimentResults import ExperimentResults
from dendritic.experiments.utils.experiment_finetuning import (
    FinetuningConfig,
    FinetuningExperimentResults,
    run_finetuning_experiment,
)

# Environment configuration
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"  # Fix pygame spam

# Logger instance
logger = logging.getLogger(__name__)


def set_random_seed(seed: int) -> None:
    """Set seeds for reproducibility across random, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(log_level: int | None = None) -> logging.Logger:
    """Configure logging."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = logging.getLogger(__name__)
    # Remove any existing handlers to avoid duplicate logs
    logger.handlers.clear()
    logger.propagate = False
    level = log_level if log_level is not None else logging.INFO
    logger.setLevel(level)

    # Stream handler
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)

    # File handler
    file_handler = logging.FileHandler(Path(f"experiment_{timestamp}.log"))
    file_handler.setLevel(level)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    stream_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


def debug_dataset_integrity(
    dataset, tokenizer: GPT2Tokenizer, logger: logging.Logger
) -> None:
    """
    Inspect a single batch from the dataset and log debugging information.
    This replicates the previous print‑based debugging block but uses the
    provided logger at DEBUG level.
    """
    # Create a temporary loader to grab one batch
    debug_loader = DataLoader(dataset, batch_size=4)  # type: ignore
    batch = next(iter(debug_loader))

    inp = batch["input_ids"]
    lbl = batch["labels"]
    pad_id = tokenizer.pad_token_id

    # 1. Check Ratio of Padding
    total_tokens = inp.numel()
    pad_tokens = (inp == pad_id).sum().item()
    logger.debug(f"Batch Shape: {inp.shape}")
    logger.debug(f"Pad Token ID: {pad_id}")
    logger.debug(
        f"Padding Ratio: {pad_tokens / total_tokens:.2%} ({pad_tokens}/{total_tokens} tokens)"
    )

    # 2. Check Masking (Crucial for correct PPL)
    masked_tokens = (lbl == -100).sum().item()
    logger.debug(f"Masked Label Ratio (-100): {masked_tokens / total_tokens:.2%}")

    # 3. Visual Inspection
    logger.debug("-" * 20 + " Sample 0 (Decoded) " + "-" * 20)
    logger.debug("".join(tokenizer.decode(inp[0])))
    logger.debug("-" * 20 + " Sample 0 (Raw IDs) " + "-" * 20)
    logger.debug(inp[0].tolist())
    logger.debug("-" * 20 + " Sample 0 (Labels) " + "-" * 20)
    logger.debug(lbl[0].tolist())
    logger.debug("!" * 40 + "\n")


def load_pretraining_data(
    tokenizer: GPT2Tokenizer,
    config: PretrainingConfig,
    max_length: int = 256,
    num_workers: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Load and preprocess the WikiText-103 dataset for pretraining.

    Parameters
    ----------
    tokenizer : GPT2Tokenizer
        Tokenizer to use for text tokenization.
    config : PretrainingConfig
        Configuration object containing training parameters.
    max_length : int, optional
        Maximum sequence length for tokenization. Defaults to 256.
    num_workers : int | None, optional
        Number of worker processes for dataset mapping. If ``None`` the value
        defaults to half the CPU cores (minimum 1).

    Returns
    -------
    tuple[DataLoader, DataLoader]
        Training and evaluation DataLoaders.
    """

    from datasets import load_dataset, Dataset

    # Compute required samples: steps * batch_size * epochs (safety margin +10%)
    num_train_samples = (
        int(config.training_steps * config.batch_size * 1 * 1.1) * 5
    )  # compensate for packing
    num_eval_samples = int(num_train_samples * 0.1)

    logger.info(
        f"Loading {num_train_samples:,} train + {num_eval_samples:,} eval samples "
        f"(target: {config.training_steps * config.batch_size:,} batches)"
    )
    logger.info(f"Loading WikiText-103 (Top {num_train_samples} samples)...")

    # We load the full split first, then filter/select
    # This ensures we don't count empty lines as 'samples'
    try:
        full_train = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
        full_eval = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    except Exception as e:
        logger.error(f"Failed to load WikiText-103 dataset: {e}")
        raise

    assert isinstance(full_train, Dataset)
    assert isinstance(full_eval, Dataset)
    # Filter empty lines
    train_filtered = full_train.filter(lambda x: len(x["text"]) > 10)
    eval_filtered = full_eval.filter(lambda x: len(x["text"]) > 10)

    # SAFE SELECTION: Take the minimum of (what we want, what we have)
    # This prevents the IndexError

    # For Train: Ensure we have enough, or warn the user
    actual_train_len = len(train_filtered)
    if actual_train_len < num_train_samples:
        logger.warning(
            f"WARNING: Requested {num_train_samples} train samples, but only {actual_train_len} available."
        )
        logger.info("Training will run for fewer steps or cycle data.")

    train_dataset = train_filtered.select(
        range(min(num_train_samples, actual_train_len))
    )

    # For Eval: Just take whatever is available up to the target
    eval_dataset = eval_filtered.select(
        range(min(num_eval_samples, len(eval_filtered)))
    )

    # def tokenize_function(examples):
    #     tokenized = tokenizer(
    #         examples["text"],
    #         truncation=True,
    #         max_length=max_length,
    #         padding="max_length",
    #         return_tensors="pt",
    #     )
    #     tokenized["labels"] = tokenized["input_ids"].clone()
    #     return tokenized
    # 1. Tokenize with appended newline
    def tokenize_fn(examples):
        # Append \n so we don't merge "End of sentence." and "Start of next" into one word.
        # GPT-2 tokenizer handles \n (usually maps to token ID 198)
        texts = [t + "\n" for t in examples["text"]]
        return tokenizer(texts)

    # Number of parallel processes for dataset mapping – will be overridden by CLI if provided
    # Determine number of worker processes for dataset mapping.
    # If the caller provides a value (e.g., via CLI), use it; otherwise fall back to half the CPU cores (minimum 1).
    num_cores = (
        num_workers
        if num_workers is not None
        else (multiprocessing.cpu_count() // 2 or 1)
    )

    # 1. Tokenize without padding first
    train_ds = train_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=train_dataset.column_names,
        num_proc=num_cores,
    )
    eval_ds = eval_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=eval_dataset.column_names,
        num_proc=num_cores,
    )

    # 2. Group into blocks of max_length
    def group_texts(examples):
        # Concatenate all texts
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        # We drop the small remainder at the end
        total_length = (total_length // max_length) * max_length

        # Split into chunks of max_length
        result = {
            k: [t[i : i + max_length] for i in range(0, total_length, max_length)]
            for k, t in concatenated_examples.items()
        }
        # For causal LM, labels are a copy of input_ids
        result["labels"] = result["input_ids"].copy()
        return result

    train_dataset = train_ds.map(
        group_texts,
        batched=True,
        remove_columns=train_ds.column_names,
        num_proc=num_cores,
    )
    eval_dataset = eval_ds.map(
        group_texts,
        batched=True,
        remove_columns=eval_ds.column_names,
        num_proc=num_cores,
    )

    train_dataset.set_format("torch")
    eval_dataset.set_format("torch")
    # ========================== DEBUGGING BLOCK START ============================
    # (Removed inline definition; using module-level helper)
    if logger.isEnabledFor(logging.DEBUG):
        debug_dataset_integrity(train_dataset, tokenizer, logger)

    # Create a temporary loader to grab one batch
    # (moved into debug_dataset_integrity)

    # (moved into debug_dataset_integrity)

    # 1. Check Ratio of Padding
    # (moved into debug_dataset_integrity)

    # 2. Check Masking (Crucial for correct PPL)
    # In PyTorch CrossEntropyLoss, -100 is ignored. If this is 0%, your PPL is fake.
    # (moved into debug_dataset_integrity)

    # 3. Visual Inspection
    # (moved into debug_dataset_integrity)
    # (moved into debug_dataset_integrity)
    # ========================== DEBUGGING BLOCK END ============================
    train_dataloader = DataLoader(
        train_dataset,  # type: ignore
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_cores,  # Use background processes to load data
        pin_memory=True,  # Speeds up CPU-to-GPU transfer
        persistent_workers=True,  # Keeps workers alive, avoids re-initialization overhead
    )
    eval_dataloader = DataLoader(
        eval_dataset,  # type: ignore
        batch_size=config.batch_size,
        shuffle=False,
        # num_workers=num_cores,  # Use background processes to load data
        # pin_memory=True,  # Speeds up CPU-to-GPU transfer
        # persistent_workers=True,  # Keeps workers alive, avoids re-initialization overhead
    )

    return train_dataloader, eval_dataloader

# Helper to create a human‑readable identifier for a config variant
def _variant_identifier(cfg: PretrainingConfig) -> str:
    """Generate a concise identifier based on the cohort scheduler configuration."""
    if cfg.cohort_scheduler is None:
        return "no_scheduler"
    cs = cfg.cohort_scheduler
    return f"min{cs.min_mult}_max{cs.max_mult}"


def load_finetuning_data(
    tokenizer: GPT2Tokenizer,
    max_length: int = 256,
    batch_size: int = 4,
    num_workers: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Load finetuning data using either the PythonAlpacaHandler or a fallback dataset.

    Parameters
    ----------
    tokenizer : GPT2Tokenizer
        Tokenizer to use for text tokenization.
    max_length : int, optional
        Maximum sequence length for tokenization. Defaults to 256.
    batch_size : int, optional
        Batch size for the DataLoaders. Defaults to 4.
    num_workers : int | None, optional
        Number of worker processes for DataLoader. If ``None`` defaults to 0.

    Returns
    -------
    tuple[DataLoader, DataLoader]
        Training and evaluation DataLoaders.
    """
    # Try to use your handler, fall back to a simple alternative
    try:
        from dendritic.dataset_handlers.PythonAlpacaHandler import PythonAlpacaHandler

        handler = PythonAlpacaHandler(tokenizer, max_length=max_length)
        prepared = handler.prepare_data(test_size=0.1)

        train_dataloader = DataLoader(
            prepared["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers if num_workers is not None else 0,
        )
        eval_dataloader = DataLoader(
            prepared["eval"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers if num_workers is not None else 0,
        )
    except ImportError:
        # Fallback: use a code dataset from HuggingFace
        from datasets import load_dataset, Dataset

        dataset = load_dataset(
            "codeparrot/codeparrot-clean-valid", split="train[:10000]"
        )
        assert isinstance(dataset, Dataset)

        def tokenize_fn(examples):
            tokenized = tokenizer(
                examples["content"],
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
            tokenized["labels"] = tokenized.input_ids.copy()
            # Mask padding
            tokenized["labels"] = [
                [-100 if tok == tokenizer.pad_token_id else tok for tok in labels]
                for labels in tokenized.labels
            ]
            return tokenized

        dataset = dataset.map(
            tokenize_fn, batched=True, remove_columns=dataset.column_names
        )
        dataset.set_format("torch")

        split = dataset.train_test_split(test_size=0.1)

        train_dataloader = DataLoader(
            split["train"],  # type: ignore
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers if num_workers is not None else 0,
        )
        eval_dataloader = DataLoader(
            split["test"],  # type: ignore
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers if num_workers is not None else 0,
        )

    return train_dataloader, eval_dataloader


def run_pretraining_experiment(
    device: str,
    num_workers: int | None = None,
    scheduler_variants: list[PretrainingConfig] | None = None,
) -> dict[str, ExperimentResults]:
    """Run the pretraining experiment across configured seeds.

    Parameters
    ----------
    device : str
        Device identifier (e.g., ``'cuda'`` or ``'cpu'``).
    num_workers : int | None, optional
        Number of worker processes for data loading. If ``None`` the default
        is derived from the CLI argument.

    Returns
    -------
    ExperimentResults
        Results of the pretraining experiment.
    """
    from dendritic.experiments.utils.experiment_pretraining import (PretrainingExperiment, ModelVariant)

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    training_steps_count = 120000
    # Base configuration (used when no variants are supplied)
    base_config = PretrainingConfig(
        training_steps=training_steps_count,
        batch_size=16,
        eval_interval=min(max(training_steps_count // 20, 1), 500),
        seeds=[42],  # Use fewer seeds for faster testing,
        scheduler_type="plateau",
        plateau_threshold=0.0001,
    )
    # Determine which configs to run
    variants = scheduler_variants or [base_config]

    results_by_variant: dict[str, ExperimentResults] = {}

    for cfg in variants:
        experiment = PretrainingExperiment(config=cfg)
        logger.info(f"Pretraining config: {cfg}")

        train_dl, eval_dl = load_pretraining_data(
            tokenizer,
            cfg,
            max_length=cfg.max_seq_len,
            num_workers=num_workers,
        )

        # Create fresh models for each seed
        baseline_model, dendritic_model, stack_model, baseline_wave_model = (
            experiment.create_models()
        )

        # Define model variants (baseline and baseline_wave)
        model_variants = [
            ModelVariant(
                name="baseline",
                model=baseline_model,
                results=[],
                optimizer=torch.optim.AdamW(
                    baseline_model.parameters(),
                    lr=experiment.config.learning_rate,  # type: ignore
                    weight_decay=experiment.config.weight_decay,  # type: ignore
                    betas=(0.9, 0.95),
                ),
            ),
            ModelVariant(
                name="baseline_wave",
                model=baseline_wave_model,
                results=[],
                optimizer=torch.optim.AdamW(
                    baseline_wave_model.parameters(),
                    lr=experiment.config.learning_rate,  # type: ignore
                    weight_decay=experiment.config.weight_decay,  # type: ignore
                    betas=(0.9, 0.95),
                ),
            ),
        ]

        try:
            # Ensure deterministic behavior for each seed
            results: ExperimentResults | None = None
            for seed in experiment.config.seeds:
                set_random_seed(seed)
                results = experiment.run(
                    train_dl, eval_dl, model_variants=model_variants, device=device
                )
        finally:
            # Ensure GPU memory is freed even on error
            torch.cuda.empty_cache()
        if results is None:
            raise RuntimeError("Pretraining experiment failed without returning results")

        # After running all seeds for this variant, store the result
        variant_id = _variant_identifier(cfg)
        results_by_variant[variant_id] = results

    return results_by_variant


def run_finetuning_experiment_wrapper(
    device: str, num_workers: int | None = None
) -> FinetuningExperimentResults:
    """Run the finetuning experiment across configured seeds.

    Parameters
    ----------
    device : str
        Device identifier (e.g., ``'cuda'`` or ``'cpu'``).
    num_workers : int | None, optional
        Number of worker processes for data loading. If ``None`` the default
        is derived from the CLI argument.

    Returns
    -------
    FinetuningExperimentResults
        Results of the finetuning experiment.
    """

    # Ensure deterministic behavior
    set_random_seed(42)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    config = FinetuningConfig(
        training_steps=3000, batch_size=4, eval_interval=250, seeds=[42, 71, 123]
    )

    train_dl, eval_dl = load_finetuning_data(
        tokenizer,
        max_length=config.max_length,
        batch_size=config.batch_size,
        num_workers=num_workers,
    )

    from dendritic.experiments.utils.experiment_finetuning import (
        run_finetuning_experiment as run_finetuning_exp,
    )

    results: FinetuningExperimentResults | None = None
    try:
        results = run_finetuning_exp(train_dl, eval_dl, config, device)
    finally:
        torch.cuda.empty_cache()
    if results is None:
        raise RuntimeError("Finetuning experiment failed without returning results")
    return results


def main() -> None:
    """Entry point for the experiment runner.

    Parses command‑line arguments, configures logging, and dispatches
    the requested experiments (pretraining, finetuning, or both).
    """
    parser = argparse.ArgumentParser(description="Run dendritic layer experiments")
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["pretraining", "finetuning", "both"],
        default="both",
        help="Which experiment to run",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=max(multiprocessing.cpu_count() // 2, 1),
        help="Number of worker processes for data loading (default: half of CPU cores, at least 1)",
    )

    args = parser.parse_args()
    # Map textual log level to logging constant; default to INFO if not provided
    log_level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    logger = setup_logging(log_level=log_level_map.get(args.log_level))

    logger.info(f"Running experiment(s): {args.experiment}")
    logger.info(f"Device: {args.device}")

    if args.experiment in ["pretraining", "both"]:
        logger.info("\n" + "=" * 70)
        logger.info("RUNNING PRETRAINING EXPERIMENT")
        logger.info("=" * 70)
        run_pretraining_experiment(args.device, args.num_workers)

    if args.experiment in ["finetuning", "both"]:
        logger.info("\n" + "=" * 70)
        logger.info("RUNNING FINETUNING EXPERIMENT")
        logger.info("=" * 70)
        run_finetuning_experiment_wrapper(args.device, args.num_workers)

    logger.info("\nAll experiments complete!")


if __name__ == "__main__":
    main()
