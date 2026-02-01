from typing import Literal
from .MetaAwareBlock import DirectMetaAwareBlock as MetaAwareBlock
from .BaselineMLP import BaselineMLP
from .DendriticPretrainingMLP import DendriticPretrainingMLP
from .DendriticStackPretrainingMLP import DendriticStackPretrainingMLP
from .TransformerBlock import TransformerBlock
import torch.nn.functional as F


import torch
import torch.nn as nn


class MiniGPT(nn.Module):
    """
    Minimal GPT for pretraining .

    Supports baseline, dendritic, and dendritic_stack MLP variants.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        max_seq_len: int,
        hidden_dim: int,
        mlp_type: Literal[
            "standard", "dendritic"
        ] = "standard",  # "standard", "dendritic", or "dendritic_stack"
        poly_rank: int = 16,
        poly_degree: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # Embeddings
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)
        self.drop = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            if mlp_type == "standard":
                mlp = BaselineMLP(embed_dim, hidden_dim, dropout)
            elif mlp_type == "dendritic":
                mlp = DendriticPretrainingMLP(
                    embed_dim=embed_dim,
                    hidden_dim=hidden_dim,
                    poly_rank=poly_rank,
                    poly_degree=poly_degree,
                    dropout=dropout,
                )
            else:
                raise ValueError(f"Unknown mlp_type: {mlp_type}")

            block = self.create_transformer_block(embed_dim, num_heads, dropout, mlp)
            self.blocks.append(block)

        # Output
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.tok_emb.weight

        # Causal mask (registered as buffer)
        # Create causal mask and register as buffer to ensure proper device placement
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool(),
        )

        self._init_weights()

    def create_transformer_block(self, embed_dim, num_heads, dropout, mlp) -> nn.Module:
        return TransformerBlock(embed_dim, num_heads, mlp, dropout)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, input_ids: torch.Tensor, labels: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        B, T = input_ids.shape

        # Embeddings
        x, mask = self.embed_input_sequence(input_ids, T)

        # Transformer blocks
        x, _ = self.forward_through_blocks(x, mask)

        output = self.compute_logits_and_loss(labels, x)

        return output

    def compute_logits_and_loss(
        self, labels: torch.Tensor | None, x: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        x = self.ln_f(x)
        logits = self.head(x)

        output = {"logits": logits}

        # Compute loss if labels provided
        if labels is not None:
            loss = self.calculate_loss(labels, logits)
            output["loss"] = loss
        return output

    def calculate_loss(self, labels, logits) -> torch.Tensor:
        # logits[..., :-1, :] predicts labels[..., 1:]

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        return loss

    def forward_through_blocks(
        self, x: torch.Tensor, mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        all_layer_outputs = []

        for block in self.blocks:
            x = block(x, attn_mask=mask)
            all_layer_outputs.append(x.detach())
        return x, all_layer_outputs

    def embed_input_sequence(
        self, input_ids: torch.Tensor, T: int
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        tok_emb = self.tok_emb(input_ids)
        pos: torch.Tensor = torch.arange(T, device=input_ids.device).unsqueeze(0)
        pos_emb: torch.Tensor = self.pos_emb(pos)
        x: torch.Tensor = self.drop(tok_emb + pos_emb)
        # Causal mask for this sequence length
        mask = self.causal_mask[:T, :T]  # type: ignore DO NOT TOUCH!
        return x, mask


class ConfidenceAwareGPT(MiniGPT):
    """
    GPT model that uses MetaAwareBlock transformer blocks,
    incorporating confidence scalars into computations, and predicts
    its own future loss.
    """

    def __init__(self, *args, **kwargs):
        # Initialize the base MiniGPT
        super().__init__(*args, **kwargs)

        # ADDED: The Head to predict the loss/confidence
        # Projects hidden_state [B, T, dim] -> [B, T, 1]
        self.confidence_predictor = nn.Linear(self.embed_dim * len(self.blocks), 1)

        # Initialize bias to a positive value (e.g. 2.0 ~ perplexity 7.4)
        # Prevents initial "zero loss" predictions which cause instability
        nn.init.constant_(self.confidence_predictor.bias, 2.0)

    def create_transformer_block(
        self, embed_dim, num_heads, dropout, mlp
    ) -> MetaAwareBlock:
        # Use the custom block with AdaptiveLayer
        return MetaAwareBlock(embed_dim, num_heads, mlp, dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        confidence_scalars: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        B, T = input_ids.shape

        # ADDED: Handle default confidence
        if confidence_scalars is None:
            # Default to 0.0 (neutral confidence)
            confidence_scalars = torch.zeros(
                (B, T, 1),
                device=input_ids.device,
                dtype=self.tok_emb.weight.dtype,
            )

        # Embeddings
        x, mask = self.embed_input_sequence(input_ids, T)

        # Transformer blocks (Passing confidence)
        x, all_outputs = self.forward_through_blocks(x, mask, confidence_scalars)

        # Final Normalization
        x = self.ln_f(x)

        # 1. Standard Token Logits
        logits = self.head(x)

        # 2. ADDED: Predict Future Loss
        # Output shape: [B, T, 1] -> squeeze to [B, T] for easier loss calc later
        current_confidence_pred = self.confidence_predictor(
            torch.cat(all_outputs, dim=-1)
        ).squeeze(-1)

        output = {"logits": logits, "confidence_pred": current_confidence_pred}

        # Compute standard LM loss if labels provided
        if labels is not None:
            loss = self.calculate_loss(labels, logits)
            output["loss"] = loss
            # Note: We do NOT compute the confidence loss here.
            # That requires the 2-pass lookahead.

        return output

    def forward_through_blocks(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None,
        confidence_scalars: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        all_layer_outputs = []

        # Override to ensure blocks get the confidence scalar
        for block in self.blocks:
            x = block(x, confidence_scalar=confidence_scalars, attn_mask=mask)
            all_layer_outputs.append(x.detach())
        return x, all_layer_outputs

    @staticmethod
    def two_pass_training_step(
        model: "ConfidenceAwareGPT", tokens_t, tokens_t_plus_1, alpha=1.0
    ):
        """
        tokens_t: [B, SeqLen] - input sequence
        tokens_t_plus_1: [B] - next token after the sequence
        """
        B, SeqLen = tokens_t.shape

        # ===== PASS 1: Baseline (no confidence) =====
        with torch.no_grad():
            outputs_1 = model(tokens_t, confidence_scalars=None)
            logits_1 = outputs_1["logits"]  # [B, SeqLen, vocab]
            conf_pred_1 = outputs_1["confidence_pred"]  # [B, SeqLen]

            # Create labels: for position i, the label is token at position i+1
            labels = torch.cat(
                [
                    tokens_t[:, 1:],  # [B, SeqLen-1]
                    tokens_t_plus_1.unsqueeze(1),  # [B, 1]
                ],
                dim=1,
            )  # [B, SeqLen]

            # Future losses at all positions
            # logits at positions 0..SeqLen-1 predict labels at positions 0..SeqLen-1
            future_losses_baseline = F.cross_entropy(
                logits_1.reshape(-1, logits_1.size(-1)),  # [B*SeqLen, vocab]
                labels.reshape(-1),  # [B*SeqLen]
                reduction="none",
            ).view(
                B, SeqLen
            )  # [B, SeqLen]

            # Prepare confidence input for pass 2 (shift right by 1)
            conf_input = torch.cat(
                [
                    torch.zeros(
                        B, 1, 1, device=tokens_t.device, dtype=conf_pred_1.dtype
                    ),
                    conf_pred_1[:, :-1].unsqueeze(-1),
                ],
                dim=1,
            )  # [B, SeqLen, 1]

        # ===== PASS 2: With confidence =====
        outputs_2 = model(tokens_t, confidence_scalars=conf_input)
        logits_2 = outputs_2["logits"]  # [B, SeqLen, vocab]
        conf_pred_2 = outputs_2["confidence_pred"]  # [B, SeqLen]

        # Token prediction loss - USE SHIFTED LOGITS AND LABELS
        # Logits at positions 0..SeqLen-2 predict labels at positions 0..SeqLen-2
        # (We drop the last logit position since it doesn't have a "next" token to predict reliably in pass 2)
        loss_lm = F.cross_entropy(
            logits_2[:, :-1].reshape(-1, logits_2.size(-1)),  # [B*(SeqLen-1), vocab]
            labels[:, :-1].reshape(-1),  # [B*(SeqLen-1)]
            ignore_index=-100,
        )

        # Confidence loss: positions 0..SeqLen-2 predict their future difficulty
        # (We drop the last confidence position since we don't have its "future" to measure)
        loss_confidence = F.mse_loss(
            conf_pred_2[:, :-1],  # [B, SeqLen-1]
            future_losses_baseline[:, :-1].detach(),  # [B, SeqLen-1]
        )

        total_loss = loss_lm + alpha * loss_confidence

        return {
            "total_loss": total_loss,
            "loss_lm": loss_lm,
            "loss_confidence": loss_confidence,
            "pred_conf_t": conf_pred_2[:, -2],  # Second-to-last position for logging
        }
