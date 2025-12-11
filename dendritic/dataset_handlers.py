from abc import ABC, abstractmethod
from datasets import Dataset, DatasetDict
from transformers.tokenization_utils import PreTrainedTokenizer
from typing import Dict, Any, List
import torch
import numpy as np

class BaseDatasetHandler(ABC):
    """Abstract base class for dataset handlers."""
    
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = 256):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenizer.pad_token = self.tokenizer.eos_token

    @abstractmethod
    def load_data(self, **kwargs) -> Dict[str, Any]:
        """Load and return the dataset splits."""
        pass

    @abstractmethod
    def tokenize_function(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenize batch of examples with proper masking."""
        pass

    def prepare_data(self, **kwargs) -> Dict[str, Any]:
        """Load and prepare the dataset, including tokenization."""
        ds = self.load_data(**kwargs)
        
        # Tokenize datasets
        train_dataset = ds['train'].map(
            self.tokenize_function,
            batched=True,
            batch_size=100,
            remove_columns=ds['train'].column_names
        )
        
        eval_dataset = ds['test'].map(
            self.tokenize_function,
            batched=True,
            batch_size=100,
            remove_columns=ds['test'].column_names
        )
        
        # Set format for PyTorch
        train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        
        return {
            'train': train_dataset,
            'eval': eval_dataset
        }


class PythonAlpacaHandler(BaseDatasetHandler):
    """Handler for the Python Code Instructions dataset (Alpaca style)."""
    
    def load_data(self, split: str = 'train', test_size: float = 0.1, **kwargs) -> Dict[str, Any]:
        """
        Load the Alpaca-style Python code instructions dataset and split into train/test.
        Returns:
            Dict[str, Any]: {'train': Dataset, 'test': Dataset}
        """
        from datasets import load_dataset

        dataset = load_dataset('iamtarun/python_code_instructions_18k_alpaca', split=split)
        assert isinstance(dataset, Dataset), "Loaded dataset is not a Dataset instance"
        dataset = dataset.train_test_split(test_size=test_size, seed=42)
        return {
            'train': dataset['train'],
            'test': dataset['test']
        }

    def tokenize_function(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tokenize with loss masking: only compute loss on output tokens, not prompt tokens.
        Args:
            examples (Dict[str, Any]): Batch with 'prompt' and 'output' fields.
        Returns:
            Dict[str, Any]: Tokenized batch with input_ids, attention_mask, labels.
        """
        try:
            prompts = examples['prompt']
            outputs = examples['output']
        except KeyError as e:
            raise TypeError(f"Missing required field in examples: {e}")

        if not isinstance(prompts, list) or not isinstance(outputs, list):
            raise TypeError("Both 'prompt' and 'output' must be lists.")

        truncated_prompts = []
        for prompt in prompts:
            if not isinstance(prompt, str):
                raise TypeError("Prompt must be a string.")
            idx = prompt.find('### Output:')
            if idx == -1:
                truncated_prompts.append(prompt)
            else:
                end_idx = idx + len('### Output:')
                truncated_prompts.append(prompt[:end_idx])

        # Tokenize prompts and outputs separately
        prompt_tokens = self.tokenizer(truncated_prompts, truncation=False, padding=False)
        output_texts = []
        for output in outputs:
            if output is None:
                raise TypeError("Output cannot be None.")
            if not isinstance(output, str):
                raise TypeError("Output must be a string.")
            output_texts.append(output + getattr(self.tokenizer, "eos_token", ""))

        output_tokens = self.tokenizer(output_texts, truncation=False, padding=False)

        input_ids = []
        labels = []
        max_length = self.max_length
        pad_token_id = getattr(self.tokenizer, "pad_token_id", 0)

        for prompt_ids, output_ids in zip(prompt_tokens['input_ids'], output_tokens['input_ids']):
            full_ids = prompt_ids + output_ids
            if len(full_ids) > max_length:
                full_ids = full_ids[:max_length]

            prompt_len = len(prompt_ids)
            full_labels = [-100] * prompt_len + output_ids
            if len(full_labels) > max_length:
                full_labels = full_labels[:max_length]

            padding_length = max_length - len(full_ids)
            full_ids = full_ids + [pad_token_id] * padding_length
            full_labels = full_labels + [-100] * padding_length

            input_ids.append(full_ids)
            labels.append(full_labels)

        attention_mask = [
            [1 if id != pad_token_id else 0 for id in ids]
            for ids in input_ids
        ]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


class TinyStoriesHandler(BaseDatasetHandler):
    """Handler for the TinyStories dataset (stub implementation)."""
    
    def load_data(self, **kwargs) -> Dict[str, Any]:
        """Load TinyStories dataset."""
        raise NotImplementedError("TinyStoriesHandler not yet implemented")
    
    def tokenize_function(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenize TinyStories examples."""
        raise NotImplementedError("TinyStoriesHandler not yet implemented")


class WikiTextHandler(BaseDatasetHandler):
    """Handler for the WikiText dataset (stub implementation)."""
    
    def load_data(self, **kwargs) -> Dict[str, Any]:
        """Load WikiText dataset."""
        raise NotImplementedError("WikiTextHandler not yet implemented")
    
    def tokenize_function(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenize WikiText examples."""
        raise NotImplementedError("WikiTextHandler not yet implemented")


class GitHubCodeHandler(BaseDatasetHandler):
    """Handler for GitHub code dataset (stub implementation)."""
    
    def load_data(self, languages: List[str] = ['Python'], **kwargs) -> Dict[str, Any]:
        """Load GitHub code dataset for specified languages."""
        raise NotImplementedError("GitHubCodeHandler not yet implemented")
    
    def tokenize_function(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """Tokenize GitHub code examples."""
        raise NotImplementedError("GitHubCodeHandler not yet implemented")


# Export standard interface
__all__ = ['BaseDatasetHandler', 'PythonAlpacaHandler', 'GitHubCodeHandler', 
           'TinyStoriesHandler', 'WikiTextHandler']
