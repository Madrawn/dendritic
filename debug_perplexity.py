#!/usr/bin/env python3
"""
Debug script to investigate perplexity calculation for confidence model.
"""
import sys

sys.path.insert(0, ".")

import torch
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

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


def main():
    # Use a tiny config for quick testing
    config = ConfidenceExperimentConfig(
        dataset="tinystories",
        max_seq_len=16,
        batch_size=4,
        training_steps=10,
        eval_interval=5,
        eval_batches=2,
        vocab_size=50257,  # GPT-2 vocab size
        embed_dim=64,
        num_heads=2,
        num_layers=2,
        hidden_dim=128,
        poly_rank=4,
        poly_degree=2,
        dropout=0.0,
        confidence_alpha=1.0,
        lookahead_steps=2,
        confidence_init_bias=2.0,
        results_dir="debug_results",
        save_interval=100,
    )

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare data
    dataloaders = prepare_confidence_data(config, tokenizer, num_workers=0)
    train_loader = dataloaders["train"]
    eval_loader = dataloaders["eval"]

    # Create models
    standard_model = MiniGPT(
        vocab_size=config.vocab_size,
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        max_seq_len=config.max_seq_len,
        hidden_dim=config.hidden_dim,
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
        hidden_dim=config.hidden_dim,
        mlp_type="standard",
        poly_rank=config.poly_rank,
        poly_degree=config.poly_degree,
        dropout=config.dropout,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    standard_model.to(device)
    confidence_model.to(device)

    # Create strategies
    standard_strategy = StandardTrainingStrategy(config)
    confidence_strategy = ConfidenceTrainingStrategy(config)

    # Evaluate a single batch with both strategies
    batch = next(iter(eval_loader))
    print(f"Batch shapes: {[t.shape for t in batch]}")

    # Standard evaluation
    standard_model.eval()
    with torch.no_grad():
        standard_result = standard_strategy.evaluation_step(
            standard_model, batch, device
        )
    print(f"Standard loss: {standard_result['loss'].item()}")

    # Confidence evaluation
    confidence_model.eval()
    with torch.no_grad():
        confidence_result = confidence_strategy.evaluation_step(
            confidence_model, batch, device
        )
    print(
        f"Confidence loss_lm: {confidence_result.get('loss_lm', confidence_result.get('loss'))}"
    )

    # Compute perplexity manually
    eval_loss_standard = standard_result["loss"].item()
    eval_loss_confidence = confidence_result.get(
        "loss_lm", confidence_result.get("loss")
    ).item()
    print(f"Standard perplexity: {np.exp(eval_loss_standard)}")
    print(f"Confidence perplexity: {np.exp(eval_loss_confidence)}")

    # Also compute loss per token for standard model (should be similar)
    # Let's also inspect token distribution
    tokens_t, tokens_t_plus_1, tokens_t_plus_2 = batch
    print(f"tokens_t shape: {tokens_t.shape}")
    print(f"tokens_t_plus_1 shape: {tokens_t_plus_1.shape}")
    print(f"Sample tokens_t[0]: {tokens_t[0]}")
    print(f"Sample tokens_t_plus_1[0]: {tokens_t_plus_1[0]}")
    print(f"Sample tokens_t_plus_2[0]: {tokens_t_plus_2[0]}")

    # Check if tokens are repeated (data leakage)
    # Compute uniqueness
    unique_tokens = torch.unique(tokens_t)
    print(f"Unique tokens in tokens_t: {len(unique_tokens)}")

    # Compute loss using model forward directly
    with torch.no_grad():
        # Standard model forward with labels
        input_ids = tokens_t.to(device)
        seq_labels = torch.cat(
            [tokens_t[:, 1:], tokens_t_plus_1.unsqueeze(1)], dim=1
        ).to(device)
        outputs = standard_model(input_ids, labels=seq_labels)
        print(f"Standard model loss (direct): {outputs['loss'].item()}")

        # Confidence model forward with two_pass_training_step (for evaluation)
        # Use zero prev_conf
        batch_size, seq_len = tokens_t.shape
        prev_conf = torch.zeros(
            (batch_size, seq_len, 1), device=device, dtype=torch.float32
        )
        result = ConfidenceAwareGPT.two_pass_training_step(
            confidence_model,
            prev_conf,
            tokens_t.to(device),
            tokens_t_plus_1.to(device),
            tokens_t_plus_2.to(device),
            alpha=config.confidence_alpha,
        )
        print(f"Confidence loss_lm (direct): {result['loss_lm'].item()}")
        print(f"Confidence total_loss: {result['total_loss'].item()}")
        print(f"Confidence loss_confidence: {result['loss_confidence'].item()}")

        # Check logits
        logits_t = result.get("pred_conf_t", None)
        if logits_t is not None:
            print(f"pred_conf_t shape: {logits_t.shape}")

    # Also compute perplexity using the UnifiedTrainer's _evaluate_with_iterator
    # We'll simulate a few batches
    from dendritic.experiments.confidence.UnifiedTrainer import UnifiedTrainer

    trainer = UnifiedTrainer(config, model_type="confidence", device=device)
    # monkey-patch strategy
    trainer.strategy = confidence_strategy
    eval_loss = trainer._evaluate_with_iterator(
        confidence_model, eval_loader, max_batches=2, device=device
    )
    print(f"UnifiedTrainer eval_loss (confidence): {eval_loss}")
    print(f"Perplexity from eval_loss: {np.exp(eval_loss)}")

    # For standard model
    trainer_std = UnifiedTrainer(config, model_type="standard", device=device)
    trainer_std.strategy = standard_strategy
    eval_loss_std = trainer_std._evaluate_with_iterator(
        standard_model, eval_loader, max_batches=2, device=device
    )
    print(f"UnifiedTrainer eval_loss (standard): {eval_loss_std}")
    print(f"Perplexity from eval_loss: {np.exp(eval_loss_std)}")


if __name__ == "__main__":
    main()
