from datasets import Dataset
from transformers.tokenization_utils import PreTrainedTokenizer


import os
from abc import ABC, abstractmethod
from typing import Any


class BaseDatasetHandler(ABC):
    """Abstract base class for dataset handlers."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 256,
        text_column: str = "text",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_column = text_column
        self.dataset_name = ""  # Should be set by subclasses
        self.predefined_splits = None  # Should be set by subclasses
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_data(self, **kwargs) -> dict[str, Any]:
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

    def load_from_file(self, data_files, **kwargs) -> dict[str, Any]:
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
    def load_default_data(self, *args, **kwargs) -> dict[str, Any]:
        """Load the default dataset when no data_files are provided."""
        pass

    @abstractmethod
    def tokenize_function(self, examples: dict[str, Any]) -> dict[str, Any]:
        """Tokenize batch of examples with proper masking."""
        pass

    def prepare_pretraining_dataloaders(
        self, config: Any, num_workers: int = 0
    ) -> dict[str, Any]:
        """Prepare dataloaders for pretraining."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement pretraining dataloader preparation. "
            "Use a dataset handler that inherits from TextCorpusHandler."
        )

    def tokenize_for_pretraining(
        self, examples: dict[str, Any], append_newline: bool = True
    ) -> dict[str, Any]:
        """
        Tokenize a batch of text examples for pretraining (no padding, no truncation).

        This is used before block concatenation. Optionally appends a newline token
        to separate documents/sentences.

        The base implementation raises NotImplementedError; subclasses that support
        pretraining should override this method.

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
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support pretraining tokenization. "
            "Use a dataset handler that inherits from TextCorpusHandler."
        )

    def prepare_data(self, **kwargs) -> dict[str, Any]:
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
