from dendritic.dataset_handlers.factory import get_handler
from transformers.models.gpt2 import GPT2Tokenizer
import argparse
import sys


def run(dataset_name: str, max_samples: int):
    """Load dataset, tokenize samples, compute length statistics."""
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Get handler
    handler = get_handler(dataset_name, tokenizer)

    # Load raw data (no tokenization)
    # Use test_size=0.0 to get all samples in train split
    raw_data = handler.load_default_data(
        max_samples=max_samples,
        split="train",
        test_size=0.0,
        streaming=True,
    )
    train_dataset = raw_data["train"]

    lengths = []
    for i, example in enumerate(train_dataset):
        # Tokenize without truncation/padding
        tokenized = handler.tokenize_for_pretraining(
            {handler.text_column: [example[handler.text_column]]},
            append_newline=False,
        )
        # tokenized["input_ids"] is a list of lists
        seq_len = len(tokenized["input_ids"][0])
        lengths.append(seq_len)
        if i % 100 == 0:
            print(f"Processed {i+1} samples...", file=sys.stderr)

        if len(lengths) >= max_samples:
            break

    if not lengths:
        print("No samples found.", file=sys.stderr)
        return

    avg_len = sum(lengths) / len(lengths)
    min_len = min(lengths)
    max_len = max(lengths)

    print(f"Dataset: {dataset_name}")
    print(f"Number of samples: {len(lengths)}")
    print(f"Average token length: {avg_len:.2f}")
    print(f"Minimum token length: {min_len}")
    print(f"Maximum token length: {max_len}")
    print(f"Length distribution: {lengths[:10]}... (first 10)")


def main():
    parser = argparse.ArgumentParser(
        description="Compute token length statistics for a dataset."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="openwebmath",
        help="Dataset identifier (e.g., 'openwebmath', 'wikitext', 'python_alpaca')",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1000,
        help="Maximum number of samples to analyze",
    )
    args = parser.parse_args()

    run(args.dataset, args.max_samples)


if __name__ == "__main__":
    main()