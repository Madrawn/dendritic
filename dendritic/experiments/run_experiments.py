# dendritic/experiments/run_experiments.py
"""
Main script to run both .

Usage:
    python -m dendritic.experiments.run_experiments --experiment pretraining
    python -m dendritic.experiments.run_experiments --experiment finetuning
    python -m dendritic.experiments.run_experiments --experiment both
"""

import argparse
import logging
import multiprocessing
from datetime import datetime
import os
from dendritic.experiments.utils.PretrainingConfig import PretrainingConfig
from experiment_finetuning import FinetuningConfig

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"  # Fix pygame spam
from numpy import isin
import torch
from torch.utils.data import DataLoader
from transformers.models.gpt2 import GPT2Tokenizer


def setup_logging() -> logging.Logger:
    """Configure logging."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"experiment_{timestamp}.log"),
        ],
    )
    return logging.getLogger(__name__)


def load_pretraining_data(
    tokenizer: GPT2Tokenizer,
    config: PretrainingConfig,
    max_length: int = 256,
):
    """
    Load data for pretraining experiment.

    You can use datasets like:
    - TinyStories (small, fast)
    - OpenWebText subset
    - WikiText
    """
    from datasets import load_dataset, Dataset

    # Compute required samples: steps * batch_size * epochs (safety margin +10%)
    num_train_samples = (
        int(config.training_steps * config.batch_size * 1 * 1.1) * 5
    )  # compensate for packing
    num_eval_samples = int(num_train_samples * 0.1)

    print(
        f"Loading {num_train_samples:,} train + {num_eval_samples:,} eval samples "
        f"(target: {config.training_steps * config.batch_size:,} batches)"
    )
    print(f"Loading WikiText-103 (Top {num_train_samples} samples)...")

    # We load the full split first, then filter/select
    # This ensures we don't count empty lines as 'samples'
    full_train = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    full_eval = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")

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
        print(
            f"WARNING: Requested {num_train_samples} train samples, but only {actual_train_len} available."
        )
        print("Training will run for fewer steps or cycle data.")

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

    num_cores = multiprocessing.cpu_count() // 2 or 1

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
    print("\n" + "!" * 40)
    print("DEBUG: INSPECTING DATASET INTEGRITY")

    # Create a temporary loader to grab one batch
    debug_loader = DataLoader(train_dataset, batch_size=4)  # type: ignore
    batch = next(iter(debug_loader))

    inp = batch["input_ids"]
    lbl = batch["labels"]
    pad_id = tokenizer.pad_token_id

    # 1. Check Ratio of Padding
    total_tokens = inp.numel()
    pad_tokens = (inp == pad_id).sum().item()
    print(f"Batch Shape: {inp.shape}")
    print(f"Pad Token ID: {pad_id}")
    print(
        f"Padding Ratio: {pad_tokens / total_tokens:.2%} ({pad_tokens}/{total_tokens} tokens)"
    )

    # 2. Check Masking (Crucial for correct PPL)
    # In PyTorch CrossEntropyLoss, -100 is ignored. If this is 0%, your PPL is fake.
    masked_tokens = (lbl == -100).sum().item()
    print(f"Masked Label Ratio (-100): {masked_tokens / total_tokens:.2%}")

    # 3. Visual Inspection
    print("-" * 20 + " Sample 0 (Decoded) " + "-" * 20)
    print("".join(tokenizer.decode(inp[0])))
    print("-" * 20 + " Sample 0 (Raw IDs) " + "-" * 20)
    print(inp[0].tolist())
    print("-" * 20 + " Sample 0 (Labels) " + "-" * 20)
    print(lbl[0].tolist())
    print("!" * 40 + "\n")
    # ========================== DEBUGGING BLOCK END ============================
    train_dataloader = DataLoader(
        train_dataset,  # type: ignore
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,  # Use background processes to load data
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


def load_finetuning_data(
    tokenizer: GPT2Tokenizer, max_length: int = 256, batch_size: int = 4
):
    """
    Load data for finetuning experiment.

    Uses your PythonAlpacaHandler or similar.
    """
    # Try to use your handler, fall back to a simple alternative
    try:
        from dendritic.dataset_handlers.PythonAlpacaHandler import PythonAlpacaHandler

        handler = PythonAlpacaHandler(tokenizer, max_length=max_length)
        prepared = handler.prepare_data(test_size=0.1)

        train_dataloader = DataLoader(
            prepared["train"], batch_size=batch_size, shuffle=True
        )
        eval_dataloader = DataLoader(
            prepared["eval"], batch_size=batch_size, shuffle=False
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
            split["train"], batch_size=batch_size, shuffle=True  # type: ignore
        )
        eval_dataloader = DataLoader(
            split["test"], batch_size=batch_size, shuffle=False  # type: ignore
        )

    return train_dataloader, eval_dataloader


def run_pretraining_experiment(device: str):
    """Run pretraining comparison experiment."""
    from dendritic.experiments.utils.experiment_pretraining import (
        run_pretraining_experiment as run_exp,
    )

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    training_steps_count = 120000
    config = PretrainingConfig(
        training_steps=training_steps_count,
        batch_size=16,
        eval_interval=min(max(training_steps_count // 20, 1), 500),
        seeds=[42],  # Use fewer seeds for faster testing
        scheduler_type="plateau",
    )

    train_dl, eval_dl = load_pretraining_data(
        tokenizer,
        config,
        max_length=config.max_seq_len,
    )

    results = run_exp(train_dl, eval_dl, config, device)
    return results


def run_finetuning_experiment(device: str):
    """Run finetuning comparison experiment."""

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    config = FinetuningConfig(
        training_steps=3000, batch_size=4, eval_interval=250, seeds=[42, 71, 123]
    )

    train_dl, eval_dl = load_finetuning_data(
        tokenizer, max_length=config.max_length, batch_size=config.batch_size
    )

    results = run_finetuning_experiment(device)
    return results


def main():
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

    args = parser.parse_args()
    logger = setup_logging()

    logger.info(f"Running experiment(s): {args.experiment}")
    logger.info(f"Device: {args.device}")

    if args.experiment in ["pretraining", "both"]:
        logger.info("\n" + "=" * 70)
        logger.info("RUNNING PRETRAINING EXPERIMENT")
        logger.info("=" * 70)
        run_pretraining_experiment(args.device)

    if args.experiment in ["finetuning", "both"]:
        logger.info("\n" + "=" * 70)
        logger.info("RUNNING FINETUNING EXPERIMENT")
        logger.info("=" * 70)
        run_finetuning_experiment(args.device)

    logger.info("\nAll experiments complete!")


if __name__ == "__main__":
    main()
