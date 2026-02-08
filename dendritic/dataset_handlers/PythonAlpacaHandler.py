from dendritic.dataset_handlers.InstructionHandler import InstructionHandler


from datasets import DatasetDict, IterableDataset, IterableDatasetDict


from typing import Any, Dict


class PythonAlpacaHandler(InstructionHandler):
    """Handler for the Python Code Instructions dataset (Alpaca style)."""

    def load_default_data(self, split: str = "train", test_size: float = 0.1, **kwargs) -> Dict[str, Any]:
        """
        Load the Alpaca-style Python code instructions dataset and split into train/test.
        Returns:
            Dict[str, Any]: {'train': Dataset, 'test': Dataset}
        """
        from datasets import load_dataset

        dataset = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split=split)
        assert not isinstance(dataset, (DatasetDict, IterableDataset, IterableDatasetDict)), (
            "Loaded dataset is not a Dataset instance"
        )
        dataset = dataset.train_test_split(test_size=test_size, seed=42)
        return {"train": dataset["train"], "test": dataset["test"]}

    def tokenize_function(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tokenize with loss masking: only compute loss on output tokens, not prompt tokens.
        Args:
            examples (Dict[str, Any]): Batch with 'prompt' and 'output' fields.
        Returns:
            Dict[str, Any]: Tokenized batch with input_ids, attention_mask, labels.
        """
        # Apply custom prompt truncation for PythonAlpaca dataset
        prompts = examples["prompt"]
        outputs = examples["output"]

        truncated_prompts = []
        for prompt in prompts:
            if not isinstance(prompt, str):
                raise TypeError("Prompt must be a string.")
            idx = prompt.find("### Output:")
            if idx == -1:
                truncated_prompts.append(prompt)
            else:
                end_idx = idx + len("### Output:")
                truncated_prompts.append(prompt[:end_idx])

        # Create modified examples with truncated prompts
        modified_examples = {"prompt": truncated_prompts, "output": outputs}

        # Call parent tokenize_function with modified examples
        return super().tokenize_function(modified_examples)
