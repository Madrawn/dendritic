"""
TextCorpusHandler - abstract base class for text‑only corpora (WikiText, OpenWebMath, etc.).
Implements common tokenization for causal language modeling and provides default loading
via Hugging Face datasets library.
"""

from abc import ABC
from typing import Any, Dict, Optional, Union, List
from datasets import Dataset, IterableDataset, DatasetDict, IterableDatasetDict, load_dataset
from httpx import stream
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
            raise ValueError("dataset_name must be provided either via class attribute or constructor")
    
    def load_default_data(
        self,
        max_samples: int,
        split: str = "train",
        test_size: float = 0.1,
        seed: int = 42,
        streaming: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
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
        Dict[str, Any]
            {'train': Dataset, 'test': Dataset}
        """
        assert streaming, "This method requires streaming=True to avoid large downloads. Don't you dare run it with streaming=False!"
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
                raise ValueError(f"Split '{split}' not found in dataset {self.dataset_name}")
            ds = ds[split]
        
        # 3. Convert IterableDataset to regular Dataset and limit to max_samples
        if isinstance(ds, IterableDataset):
            # Streaming dataset: take the first max_samples and convert to list
            ds_head = ds.take(max_samples)
            samples = list(ds_head)
            if not samples:
                raise ValueError(f"No samples retrieved from dataset {self.dataset_name}")
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
    
    def tokenize_function(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tokenize a batch of text examples for causal language modeling.
        
        Applies truncation and padding to `self.max_length`. The labels are
        a copy of input_ids, with padding tokens masked (set to -100).
        
        Parameters
        ----------
        examples : Dict[str, Any]
            Batch with at least the key `self.text_column` containing a list of strings.
            
        Returns
        -------
        Dict[str, Any]
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
    
    def tokenize_for_pretraining(self, examples: Dict[str, Any], append_newline: bool = True) -> Dict[str, Any]:
        """
        Tokenize a batch of text examples for pretraining (no padding, no truncation).
        
        This is used before block concatenation. Optionally appends a newline token
        to separate documents/sentences.
        
        Parameters
        ----------
        examples : Dict[str, Any]
            Batch with at least the key `self.text_column` containing a list of strings.
        append_newline : bool
            Whether to append a newline token to each text.
            
        Returns
        -------
        Dict[str, Any]
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
    
    def prepare_pretraining_data(
        self,
        config: Any,  # PretrainingConfig
        num_workers: Optional[int] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Prepare data for pretraining with block concatenation (grouping).
        
        This method loads the dataset, tokenizes, groups consecutive tokens into
        blocks of `config.max_seq_len`, and returns DataLoaders ready for training.
        
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
        Dict[str, Any]
            {'train': DataLoader, 'eval': DataLoader}
        """
        from torch.utils.data import DataLoader
        import multiprocessing
        
        # 1. Load raw splits
        raw = self.load_default_data(**kwargs)
        train_raw = raw["train"]
        eval_raw = raw["test"]
        
        # 2. Tokenize (without padding) using the tokenize_function but without max_length padding
        # We'll define a temporary tokenization that does not pad.
        def tokenize_no_pad(examples):
            texts = examples[self.text_column]
            tokenized = self.tokenizer(
                texts,
                truncation=False,
                padding=False,
                return_tensors=None,
            )
            return tokenized.data
        
        # Determine number of processes for mapping
        num_proc = num_workers if num_workers is not None else (multiprocessing.cpu_count() // 2 or 1)
        
        # Apply tokenization
        train_tokenized = train_raw.map(
            tokenize_no_pad,
            batched=True,
            remove_columns=train_raw.column_names,
            num_proc=num_proc if not isinstance(train_raw, IterableDataset) else None,
        )
        eval_tokenized = eval_raw.map(
            tokenize_no_pad,
            batched=True,
            remove_columns=eval_raw.column_names,
            num_proc=num_proc if not isinstance(eval_raw, IterableDataset) else None,
        )
        
        # 3. Group into blocks of max_seq_len
        def group_texts(examples):
            # Concatenate all texts
            concatenated = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated[list(examples.keys())[0]])
            # Drop the small remainder
            total_length = (total_length // config.max_seq_len) * config.max_seq_len
            # Split into chunks
            result = {
                k: [t[i:i + config.max_seq_len] for i in range(0, total_length, config.max_seq_len)]
                for k, t in concatenated.items()
            }
            # Labels are copy of input_ids for causal LM
            result["labels"] = result["input_ids"].copy()
            return result
        
        train_grouped = train_tokenized.map(
            group_texts,
            batched=True,
            remove_columns=train_tokenized.column_names,
            num_proc=num_proc if not isinstance(train_tokenized, IterableDataset) else None,
        )
        eval_grouped = eval_tokenized.map(
            group_texts,
            batched=True,
            remove_columns=eval_tokenized.column_names,
            num_proc=num_proc if not isinstance(eval_tokenized, IterableDataset) else None,
        )
        
        # 4. Convert to torch format
        train_grouped.set_format("torch")
        eval_grouped.set_format("torch")
        
        # 5. Create DataLoaders
        train_dl = DataLoader(
            train_grouped,
            batch_size=config.batch_size,
            shuffle=not isinstance(train_grouped, IterableDataset),
            num_workers=num_workers if num_workers is not None else 0,
            pin_memory=True,
            persistent_workers=num_workers is not None and num_workers > 0,
        )
        eval_dl = DataLoader(
            eval_grouped,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=num_workers if num_workers is not None else 0,
            pin_memory=True,
            persistent_workers=False,
        )
        
        return {"train": train_dl, "eval": eval_dl}