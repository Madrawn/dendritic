"""
TextCorpusHandler - abstract base class for text‑only corpora (WikiText, OpenWebMath, etc.).
Implements common tokenization for causal language modeling and provides default loading
via Hugging Face datasets library.
"""

from abc import ABC
from calendar import c
from typing import Any
from datasets import (
    Dataset,
    IterableDataset,
    DatasetDict,
    IterableDatasetDict,
    load_dataset,
)
from tqdm import tqdm
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.tokenization_utils_base import BatchEncoding

from dendritic.dataset_handlers.BaseDatasetHandler import BaseDatasetHandler
from functools import partial
from torch.utils.data import DataLoader
from datasets import IterableDataset, Dataset

# --- 1. Define Helper Functions at Top Level ---
# These must be picklable, so they cannot be nested inside the class method.


def tokenize_no_pad_func(examples, tokenizer, text_column, separator_ids):
    """Tokenize without padding, adding separators."""
    texts = examples[text_column]
    tokenized = tokenizer(texts, truncation=False, padding=False, return_tensors=None)
    input_ids = tokenized["input_ids"]

    if separator_ids:
        # Optimizing: list comprehension is faster
        input_ids = [seq + separator_ids for seq in input_ids]

    return {"input_ids": input_ids}


def tokenize_padded_func(examples, tokenizer, text_column, max_seq_len):
    """Tokenize with padding and masking."""
    texts = examples[text_column]
    tokenized = tokenizer(
        texts,
        truncation=True,
        max_length=max_seq_len,
        padding="max_length",
        return_tensors=None,
    )
    input_ids = tokenized["input_ids"]

    # Handle implicit single-input wrapping
    if isinstance(input_ids[0], int):
        input_ids = [input_ids]

    labels = [seq.copy() for seq in input_ids]

    # Mask padding tokens in labels
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is not None:
        for seq in labels:
            for i, token in enumerate(seq):
                if token == pad_token_id:
                    seq[i] = -100

    return {"input_ids": input_ids, "labels": labels}


def group_texts_func(examples, max_seq_len):
    """
    Standard HF 'packing' logic.
    Concatenates a batch of examples and splits them into fixed chunks.
    """
    # Only concatenate columns that are lists of lists (token IDs)
    # This prevents errors if 'text' or other metadata columns are still present
    concatenated = {
        k: sum(examples[k], [])
        for k in examples.keys()
        if isinstance(examples[k][0], list)
    }

    # Total length of the concatenated sequences
    total_length = len(next(iter(concatenated.values())))

    # We drop the small remainder
    if total_length >= max_seq_len:
        total_length = (total_length // max_seq_len) * max_seq_len

    # Split into chunks of max_seq_len
    result = {
        k: [t[i : i + max_seq_len] for i in range(0, total_length, max_seq_len)]
        for k, t in concatenated.items()
    }

    # Standard LLM pretraining usually predicts the next token,
    # so we often create labels here if they don't exist
    if "input_ids" in result:
        result["labels"] = result["input_ids"].copy()

    return result


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
        max_samples: int = 10,
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
        self, config: Any, num_workers: int = 0, kwargs=None
    ):
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
        if kwargs is None:
            kwargs = {}

        # Calculate requirements
        train_blocks_needed = config.training_steps * config.batch_size
        test_blocks_needed = min(
            int(train_blocks_needed * config.eval_split_ratio),
            config.batch_size * config.eval_batches,
        )

        # Load raw stream
        load_kwargs = kwargs.copy()
        load_kwargs.setdefault("streaming", True)

        # Note: We remove the 'max_samples' restriction on the loader if possible,
        # or keep it high enough. streaming=True lazily loads anyway.
        raw = self.load_default_data(**load_kwargs)
        raw_stream: IterableDataset = raw["train"]  # This is an IterableDataset

        # --- Pipeline Construction ---

        if config.grouped:
            separator_token_ids = self._get_separator_token_ids(config.group_separator)

            # 1. Tokenize
            # Use partial to pass 'self' arguments into the top-level function
            tokenized_stream = raw_stream.map(
                partial(
                    tokenize_no_pad_func,
                    tokenizer=self.tokenizer,
                    text_column=self.text_column,
                    separator_ids=separator_token_ids,
                ),
                batched=True,
                remove_columns=raw_stream.column_names,
            )

            # 2. Group (Pack)
            # This replaces your custom generator.
            # batched=True allows access to multiple samples to pack them efficiently.
            processed_stream = tokenized_stream.map(
                partial(group_texts_func, max_seq_len=config.max_seq_len), batched=True
            )

        else:
            # Ungrouped pipeline
            processed_stream = raw_stream.map(
                partial(
                    tokenize_padded_func,
                    tokenizer=self.tokenizer,
                    text_column=self.text_column,
                    max_seq_len=config.max_seq_len,
                ),
                batched=True,
                remove_columns=raw_stream.column_names,
            )

        # --- Splitting Train / Test ---

        # Important: processed_stream is still an IterableDataset.
        # It is picklable because we used top-level functions in .map()

        # 1. Materialize Test Set
        # We take the first N items. .take() returns an IterableDataset,
        # so we iterate it to force loading into memory.
        print(f"Collecting {test_blocks_needed} validation blocks...")
        test_iter = iter(processed_stream.take(test_blocks_needed))
        test_blocks = list(tqdm(test_iter, total=test_blocks_needed))

        test_dataset = Dataset.from_dict(
            {
                "input_ids": [b["input_ids"] for b in test_blocks],
                "labels": [b["labels"] for b in test_blocks],
            }
        ).with_format("torch")
        # test_dataset = processed_stream.take(int(test_blocks_needed)).with_format(
        # "torch"
        # )

        # 2. Create Train Set using .skip()
        # This creates a NEW IterableDataset that fast-forwards the underlying stream.
        # This is safe for multiprocessing.
        train_dataset = processed_stream.skip(int(test_blocks_needed))
        train_dataset = train_dataset.with_format("torch")

        # --- DataLoaders ---

        train_dl = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            num_workers=2,  # Now safe to use > 0
            prefetch_factor=2,
            pin_memory=False,
            persistent_workers=False,
            drop_last=True,
        )

        eval_dl = DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            num_workers=1,
            pin_memory=False,
            persistent_workers=False,
            drop_last=True,
        )

        return {"train": train_dl, "eval": eval_dl}
