import torch
import numpy as np

from torch.utils.data import DataLoader

from experiments.models.MiniGPT import MiniGPT


def evaluate(
    model: MiniGPT, dataloader: DataLoader, max_batches: int | None, device: str
) -> float:
    """Evaluate model and return mean loss."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if max_batches and i >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, labels=labels)

            # Count non-masked tokens
            non_masked = (labels != -100).sum().item()
            total_loss += outputs["loss"].item() * non_masked
            total_tokens += non_masked

    return total_loss / total_tokens if total_tokens > 0 else float("nan")
