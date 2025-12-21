"""
TextCorpusHandler - abstract base class for text‑only corpora (WikiText, OpenWebMath, etc.).
Implements common tokenization for causal language modeling and provides default loading
via Hugging Face datasets library.
"""

from abc import ABC
from typing import Any, Optional
from datasets import (
    Dataset,
    IterableDataset,
    DatasetDict,
    IterableDatasetDict,
    load_dataset,
)
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding

from dendritic.dataset_handlers.BaseDatasetHandler import BaseDatasetHandler


class TextCorpusHandler(BaseDatasetHandler, ABC):
    """Abstract handler for text corpora with a single text column.

    Subclasses must define `dataset_name` and `text_column` (or override `load_default_data`).
    Provides default tokenization suitable for causal LM pretraining.
    """

    # These should be set by concrete subclasses
    dataset_name: str = ""
    text_column: str = "text"

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 256,
        dataset_name: str = "",
        text_column: str = "text",
    ):
        super().__init__(tokenizer, max_length, text_column=text_column)
        if dataset_name:
            self.dataset_name = dataset_name
        # Ensure dataset_name is set
        if not self.dataset_name:
            raise ValueError(
                "dataset_name must be provided either via class attribute or constructor"
            )

    def _get_separator_token_ids(self, group_separator):
        """
        Convert group_separator config to list of token IDs.

        Parameters
        ----------
        group_separator : Literal['EOS_token', 'EOS_BOS_tokens'] | str
            Separator specification.

        Returns
        -------
        list[int]
            Token IDs to insert between documents (empty list for no separator).
        """
        if group_separator == "EOS_token":
            eos = self.tokenizer.eos_token_id
            if eos is None:
                raise ValueError(
                    "Tokenizer does not have an eos_token_id. "
                    "Please provide a custom separator string."
                )
            return [eos]
        elif group_separator == "EOS_BOS_tokens":
            eos = self.tokenizer.eos_token_id
            bos = self.tokenizer.bos_token_id
            if eos is None or bos is None:
                raise ValueError(
                    "Tokenizer missing eos_token_id or bos_token_id. "
                    "Please provide a custom separator string."
                )
            return [eos, bos]
        else:
            # Treat as literal string
            return self.tokenizer.encode(
                group_separator,
                add_special_tokens=False,
            )

    def load_default_data(
        self,
        max_samples: int,
        split: str = "train",
        test_size: float = 0.1,
        seed: int = 42,
        streaming: bool = True,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Load the dataset from Hugging Face, split into train/test in memory.

        Always loads with streaming (default) and converts to a regular Dataset
        after limiting to `max_samples`. Then splits by ratio using `train_test_split`.

        Parameters
        ----------
        max_samples : int
            Maximum number of samples to load (mandatory). This ensures we never
            accidentally download huge datasets.
        split : str
            Which split to load (if dataset has multiple). Default 'train'.
        test_size : float
            Fraction of loaded data to reserve for evaluation.
        seed : int
            Random seed for splitting.
        streaming : bool
            Whether to load as an IterableDataset (streaming mode). Default True.
        **kwargs : dict
            Additional arguments passed to `datasets.load_dataset`.

        Returns
        -------
        dict[str, Any]
            {'train': Dataset, 'test': Dataset}
        """
        assert (
            streaming
        ), "This method requires streaming=True to avoid large downloads. Don't you dare run it with streaming=False!"
        # 1. Load dataset (streaming or not)
        ds = load_dataset(
            self.dataset_name,
            split=split,
            streaming=streaming,
            **kwargs,
        )

        # 2. If we got a DatasetDict / IterableDatasetDict, extract the split
        if isinstance(ds, (DatasetDict, IterableDatasetDict)):
            if split not in ds:
                raise ValueError(
                    f"Split '{split}' not found in dataset {self.dataset_name}"
                )
            ds = ds[split]

        # 3. Convert IterableDataset to regular Dataset and limit to max_samples
        if isinstance(ds, IterableDataset):
            # Streaming dataset: take the first max_samples and convert to list
            ds_head = ds.take(max_samples)
            samples = list(ds_head)
            if not samples:
                raise ValueError(
                    f"No samples retrieved from dataset {self.dataset_name}"
                )
            ds = Dataset.from_list(samples)
        else:
            # Non‑streaming Dataset: select up to max_samples
            if len(ds) > max_samples:
                ds = ds.select(range(max_samples))

        # 4. Split into train/test
        if test_size == 0.0:
            # No test split requested, return empty test dataset
            return {"train": ds, "test": Dataset.from_list([])}
        split_result = ds.train_test_split(test_size=test_size, seed=seed)

        return {"train": split_result["train"], "test": split_result["test"]}

    def tokenize_function(self, examples: dict[str, Any]) -> dict[str, Any]:
        """
        Tokenize a batch of text examples for causal language modeling.

        Applies truncation and padding to `self.max_length`. The labels are
        a copy of input_ids, with padding tokens masked (set to -100).

        Parameters
        ----------
        examples : dict[str, Any]
            Batch with at least the key `self.text_column` containing a list of strings.

        Returns
        -------
        dict[str, Any]
            Tokenized batch with keys 'input_ids', 'attention_mask', 'labels'.
        """
        texts = examples[self.text_column]
        tokenized: BatchEncoding = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors=None,  # return lists
        )
        # Convert to dict to avoid BatchEncoding issues
        result = tokenized.data
        # For causal LM, labels are a copy of input_ids
        input_ids = result["input_ids"]
        # Ensure input_ids is list of lists
        if isinstance(input_ids[0], int):
            # single sequence (should not happen with batched=True)
            input_ids = [input_ids]
        labels = [seq.copy() for seq in input_ids]
        # Mask padding tokens in labels
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is not None:
            for seq in labels:
                for i, token in enumerate(seq):
                    if token == pad_token_id:
                        seq[i] = -100
        result["labels"] = labels
        return result

    def tokenize_for_pretraining(
        self, examples: dict[str, Any], append_newline: bool = True
    ) -> dict[str, Any]:
        """
        Tokenize a batch of text examples for pretraining (no padding, no truncation).

        This is used before block concatenation. Optionally appends a newline token
        to separate documents/sentences.

        Parameters
        ----------
        examples : dict[str, Any]
            Batch with at least the key `self.text_column` containing a list of strings.
        append_newline : bool
            Whether to append a newline token to each text.

        Returns
        -------
        dict[str, Any]
            Tokenized batch with keys 'input_ids' (list of token IDs).
        """
        texts = examples[self.text_column]
        if append_newline:
            texts = [t + "\n" for t in texts]
        tokenized: BatchEncoding = self.tokenizer(
            texts,
            truncation=False,
            padding=False,
            return_tensors=None,
        )
        return tokenized.data

    def prepare_pretraining_dataloaders(
        self,
        config: Any,  # PretrainingConfig
        num_workers: Optional[int] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Prepare data for pretraining with block concatenation (grouping).

        This method loads the dataset, tokenizes, groups consecutive tokens into
        blocks of `config.max_seq_len`, and returns DataLoaders ready for training.

        Uses streaming with deterministic hash-based train/test split and stops
        when enough blocks are collected for the required training steps.

        Parameters
        ----------
        config : PretrainingConfig
            Configuration object containing `max_seq_len`, `batch_size`, `training_steps`.
        num_workers : int, optional
            Number of worker processes for dataset mapping.
        **kwargs : dict
            Additional arguments passed to `load_default_data`.

        Returns
        -------
        dict[str, Any]
            {'train': DataLoader, 'eval': DataLoader}
        """
        from torch.utils.data import DataLoader
        from datasets import Dataset
        import hashlib
        import torch

        # Calculate required blocks
        train_blocks_needed = config.training_steps * config.batch_size
        test_blocks_needed = int(train_blocks_needed * config.eval_split_ratio)

        # Estimate max_samples: assume each raw sample yields at least 1 block
        # We'll load a bit more to account for short samples
        safety_factor = 2
        estimated_blocks_per_sample = 1  # conservative
        max_samples = int(
            (train_blocks_needed + test_blocks_needed)
            * safety_factor
            / estimated_blocks_per_sample
        )

        # Load raw data as streaming IterableDataset
        load_kwargs = kwargs.copy()
        load_kwargs.setdefault("test_size", 0.0)  # We'll handle split ourselves
        load_kwargs.setdefault("streaming", True)
        raw = self.load_default_data(max_samples=max_samples, **load_kwargs)
        # raw["train"] is the entire dataset (since test_size=0)
        raw_stream = raw["train"]

        # Determine separator token IDs based on config
        separator_token_ids = self._get_separator_token_ids(config.group_separator)

        # Tokenization function (no padding, no truncation) with separator
        def tokenize_no_pad(examples):
            texts = examples[self.text_column]
            tokenized = self.tokenizer(
                texts,
                truncation=False,
                padding=False,
                return_tensors=None,
            )
            input_ids = tokenized.data["input_ids"]
            # Append separator tokens to each sequence
            if separator_token_ids:
                new_input_ids = []
                for seq in input_ids:
                    new_seq = seq + separator_token_ids
                    new_input_ids.append(new_seq)
                tokenized["input_ids"] = new_input_ids
            return tokenized.data

        # Apply tokenization in streaming fashion
        tokenized_stream = raw_stream.map(
            tokenize_no_pad,
            batched=True,
            remove_columns=raw_stream.column_names,
        )

        # Grouping function (same as before but yields blocks one by one)
        def group_texts_generator(tokenized_iter, max_seq_len):
            """Yield individual blocks from tokenized stream."""
            buffer = []
            buffer_len = 0
            for example in tokenized_iter:
                # example is a dict with 'input_ids' list
                tokens = example["input_ids"]
                buffer.extend(tokens)
                buffer_len += len(tokens)
                # While we have enough for at least one block
                while buffer_len >= max_seq_len:
                    block = buffer[:max_seq_len]
                    buffer = buffer[max_seq_len:]
                    buffer_len -= max_seq_len
                    yield {"input_ids": block, "labels": block.copy()}
            # Discard remainder (incomplete block)

        # Deterministic split using seeded random for reproducible train/test split
        import random

        rng = random.Random(42)

        def split_generator(block_iter, test_split_ratio):
            for block in block_iter:
                # Use random.random() which is deterministic with fixed seed
                is_test = rng.random() < test_split_ratio
                if is_test:
                    yield "test", block
                else:
                    yield "train", block

        # Collect blocks until we have enough
        train_blocks = []
        test_blocks = []
        block_gen = group_texts_generator(iter(tokenized_stream), config.max_seq_len)
        split_gen = split_generator(block_gen, config.eval_split_ratio)

        for split, block in split_gen:
            if split == "train" and len(train_blocks) < train_blocks_needed:
                train_blocks.append(block)
            elif split == "test" and len(test_blocks) < test_blocks_needed:
                test_blocks.append(block)

            if (
                len(train_blocks) >= train_blocks_needed
                and len(test_blocks) >= test_blocks_needed
            ):
                break

        # Convert to Hugging Face Datasets
        train_dataset = Dataset.from_dict(
            {
                "input_ids": [block["input_ids"] for block in train_blocks],
                "labels": [block["labels"] for block in train_blocks],
            }
        )
        test_dataset = Dataset.from_dict(
            {
                "input_ids": [block["input_ids"] for block in test_blocks],
                "labels": [block["labels"] for block in test_blocks],
            }
        )
        train_dataset.set_format("torch")
        test_dataset.set_format("torch")

        # Create DataLoaders
        train_dl = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=num_workers if num_workers is not None else 0,
            pin_memory=True,
            persistent_workers=num_workers is not None and num_workers > 0,
            drop_last=True,
        )
        eval_dl = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=num_workers if num_workers is not None else 0,
            pin_memory=True,
            persistent_workers=False,
            drop_last=True,
        )

        return {"train": train_dl, "eval": eval_dl}
