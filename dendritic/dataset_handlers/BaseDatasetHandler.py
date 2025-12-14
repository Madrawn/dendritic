from datasets import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer


import os
from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseDatasetHandler(ABC):
    """Abstract base class for dataset handlers."""

    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_data(self, **kwargs) -> Dict[str, Any]:
        """Load and return the dataset splits.
        If 'data_files' is provided, load from specified file(s).
        Otherwise, call the abstract method `load_default_data`.
        """
        if "data_files" in kwargs:
            # Extract data_files and remove from kwargs to avoid duplication
            data_files = kwargs.pop("data_files")
            return self.load_from_file(data_files, **kwargs)
        else:
            return self.load_default_data(**kwargs)

    def load_from_file(self, data_files, **kwargs) -> Dict[str, Any]:
        """Load dataset from file(s)."""
        from datasets import load_dataset
        import os

        # Determine file format based on extension
        if isinstance(data_files, str):
            ext = os.path.splitext(data_files)[1].lower()
        elif isinstance(data_files, (list, tuple)) and data_files:
            ext = os.path.splitext(data_files[0])[1].lower()
        else:
            ext = ".json"  # default to JSON

        if ext == ".csv":
            dataset = load_dataset("csv", data_files=data_files)
        elif ext == ".txt":
            dataset = load_dataset("text", data_files=data_files)
        else:
            dataset = load_dataset("json", data_files=data_files)

        # Return the dataset without splitting
        return {
            "train": dataset["train"],
            "test": Dataset.from_dict({}),  # empty test set
        }

    @abstractmethod
    def load_default_data(self, **kwargs) -> Dict[str, Any]:
        """Load the default dataset when no data_files are provided."""
        pass

    @abstractmethod
    def tokenize_function(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenize batch of examples with proper masking."""
        pass

    def prepare_data(self, **kwargs) -> Dict[str, Any]:
        """Load and prepare the dataset, including tokenization."""
        ds = self.load_data(**kwargs)

        # Tokenize datasets
        train_dataset = ds["train"].map(
            self.tokenize_function,
            batched=True,
            batch_size=100,
            remove_columns=ds["train"].column_names,
        )

        # If test set exists, tokenize it
        if len(ds["test"]) > 0:
            eval_dataset = ds["test"].map(
                self.tokenize_function,
                batched=True,
                batch_size=100,
                remove_columns=ds["test"].column_names,
            )
        else:
            eval_dataset = Dataset.from_dict({})

        # Set format for PyTorch
        train_dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels"]
        )
        if len(eval_dataset) > 0:
            eval_dataset.set_format(
                type="torch", columns=["input_ids", "attention_mask", "labels"]
            )

        return {"train": train_dataset, "eval": eval_dataset}