from dendritic.dataset_handlers.BaseDatasetHandler import BaseDatasetHandler


from datasets import DatasetDict, IterableDataset, IterableDatasetDict


from typing import Any, Dict


class PythonAlpacaHandler(BaseDatasetHandler):
    """Handler for the Python Code Instructions dataset (Alpaca style)."""

    def load_default_data(
        self, split: str = "train", test_size: float = 0.1, **kwargs
    ) -> Dict[str, Any]:
        """
        Load the Alpaca-style Python code instructions dataset and split into train/test.
        Returns:
            Dict[str, Any]: {'train': Dataset, 'test': Dataset}
        """
        from datasets import load_dataset

        dataset = load_dataset(
            "iamtarun/python_code_instructions_18k_alpaca", split=split
        )
        assert not isinstance(
            dataset, (DatasetDict, IterableDataset, IterableDatasetDict)
        ), "Loaded dataset is not a Dataset instance"
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
        try:
            prompts = examples["prompt"]
            outputs = examples["output"]
        except KeyError as e:
            raise TypeError(f"Missing required field in examples: {e}")

        if not isinstance(prompts, list) or not isinstance(outputs, list):
            raise TypeError("Both 'prompt' and 'output' must be lists.")

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

        # Tokenize prompts and outputs separately
        prompt_tokens = self.tokenizer(
            truncated_prompts, truncation=False, padding=False, add_special_tokens=False
        )
        output_texts = []
        for output in outputs:
            if output is None:
                raise TypeError("Output cannot be None.")
            if not isinstance(output, str):
                raise TypeError("Output must be a string.")
            eos_token = getattr(self.tokenizer, "eos_token", "") or ""
            output_texts.append(output + eos_token)
        output_tokens = self.tokenizer(
            output_texts, truncation=False, padding=False, add_special_tokens=False
        )

        input_ids = []
        labels = []
        max_length = self.max_length
        pad_token_id = getattr(self.tokenizer, "pad_token_id", 0) or 0

        for prompt_ids, output_ids in zip(
            prompt_tokens.input_ids, output_tokens.input_ids
        ):
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
            [1 if id != pad_token_id else 0 for id in ids] for ids in input_ids
        ]
        for i, ids in enumerate(input_ids):
            invalid = [x for x in ids if x is None or x < 0 or x > 100000]
            if invalid:
                print(f"WARNING: Invalid token IDs at index {i}: {invalid[:5]}...")
                print(f"  pad_token_id: {pad_token_id}")
                print(f"  prompt_len: {len(prompt_tokens.input_ids[i])}")
                print(f"  output_len: {len(output_tokens.input_ids[i])}")
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
