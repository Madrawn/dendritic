#!/usr/bin/env python3
"""
Validate that confidence model evaluation now uses full sequence loss.
"""

import sys

sys.path.insert(0, ".")

import torch
import numpy as np
import logging


from dendritic.experiments.confidence.config import ConfidenceExperimentConfig
from dendritic.experiments.confidence.data_loader import prepare_confidence_data
from dendritic.experiments.confidence.ConfidenceTrainingStrategy import (
    ConfidenceTrainingStrategy,
)
from dendritic.experiments.confidence.StandardTrainingStrategy import (
    StandardTrainingStrategy,
)
from dendritic.experiments.models.MiniGPT import MiniGPT, ConfidenceAwareGPT
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)


def main():
    # Use a tiny config for quick testing
    config = ConfidenceExperimentConfig(
        dataset="tinystories",
        max_seq_len=16,
        batch_size=4,
        training_steps=10,
        eval_interval=5,
        eval_batches=2,
        vocab_size=50257,
        embed_dim=64,
        num_heads=2,
        num_layers=2,
        dropout=0.0,
        confidence_alpha=1.0,
        lookahead_steps=2,
        confidence_init_bias=2.0,
        results_dir="debug_results",
        save_interval=100,
        baseline_hidden_dim=256,
        dendritic_hidden_dim=256,
        dendritic_stack_hidden_dim=256,
    )

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare data
    dataloaders = prepare_confidence_data(config, tokenizer, num_workers=0)
    eval_loader = dataloaders["eval"]

    # Create models
    standard_model = MiniGPT(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        max_seq_len=config.max_seq_len,
        hidden_dim=config.baseline_hidden_dim,
        mlp_type="standard",
        poly_rank=config.poly_rank,
        poly_degree=config.poly_degree,
        dropout=config.dropout,
    )

    confidence_model = ConfidenceAwareGPT(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        max_seq_len=config.max_seq_len,
        hidden_dim=config.baseline_hidden_dim,
        mlp_type="standard",
        poly_rank=config.poly_rank,
        poly_degree=config.poly_degree,
        dropout=config.dropout,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    standard_model.to(device)
    confidence_model.to(device)

    # Create strategies
    standard_strategy = StandardTrainingStrategy(config)
    confidence_strategy = ConfidenceTrainingStrategy(config)

    # Evaluate a single batch with both strategies
    batch = next(iter(eval_loader))
    print(
        f"Batch shapes: tokens_t {batch[0].shape}, t+1 {batch[1].shape}, t+2 {batch[2].shape}"
    )

    # Standard evaluation
    standard_model.eval()
    with torch.no_grad():
        standard_result = standard_strategy.evaluation_step(
            standard_model, batch, device
        )
    std_loss = standard_result["loss"].item()
    print(f"Standard loss: {std_loss}")
    print(f"Standard perplexity: {np.exp(std_loss)}")

    # Confidence evaluation (new)
    confidence_model.eval()
    with torch.no_grad():
        confidence_result = confidence_strategy.evaluation_step(
            confidence_model, batch, device
        )
    conf_loss = confidence_result["loss"].item()
    print(f"Confidence loss: {conf_loss}")
    print(f"Confidence perplexity: {np.exp(conf_loss)}")

    # Compare
    print(
        f"Loss ratio (confidence/standard): {conf_loss / std_loss if std_loss != 0 else 'inf'}"
    )

    # Also compute loss using direct forward (full sequence) for sanity
    tokens_t, tokens_t_plus_1, _ = batch
    input_ids = tokens_t.to(device)
    seq_labels = torch.cat([tokens_t[:, 1:], tokens_t_plus_1.unsqueeze(1)], dim=1).to(
        device
    )
    with torch.no_grad():
        outputs = confidence_model(
            input_ids, labels=seq_labels, confidence_scalars=None
        )
        direct_loss = outputs["loss"].item()
    print(f"Direct forward loss: {direct_loss}")

    # Check if tokens_t_plus_1 are padding tokens
    pad_token_id = tokenizer.pad_token_id
    print(f"Pad token ID: {pad_token_id}")
    t_plus_1 = batch[1]
    padding_ratio = (t_plus_1 == pad_token_id).float().mean().item()
    print(f"Padding ratio in tokens_t_plus_1: {padding_ratio}")

    # Also check tokens_t padding
    padding_ratio_t = (batch[0] == pad_token_id).float().mean().item()
    print(f"Padding ratio in tokens_t: {padding_ratio_t}")

    # If padding ratio is high, suggest adjusting dataset handling
    if padding_ratio > 0.5:
        print(
            "WARNING: tokens_t_plus_1 are mostly padding, which may cause artificially low loss."
        )

    # Finally, run a few batches to see average loss
    max_batches = 5
    total_std_loss = 0.0
    total_conf_loss = 0.0
    count = 0
    for i, batch in enumerate(eval_loader):
        if i >= max_batches:
            break
        with torch.no_grad():
            std_res = standard_strategy.evaluation_step(standard_model, batch, device)
            conf_res = confidence_strategy.evaluation_step(
                confidence_model, batch, device
            )
        total_std_loss += std_res["loss"].item()
        total_conf_loss += conf_res["loss"].item()
        count += 1
    if count > 0:
        avg_std = total_std_loss / count
        avg_conf = total_conf_loss / count
        print(f"Average over {count} batches:")
        print(f"  Standard loss: {avg_std}, perplexity: {np.exp(avg_std)}")
        print(f"  Confidence loss: {avg_conf}, perplexity: {np.exp(avg_conf)}")


if __name__ == "__main__":
    main()
