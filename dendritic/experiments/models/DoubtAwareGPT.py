from dendritic.experiments.models.MetaAwareBlock import AdaptiveMetaAwareBlock
from dendritic.experiments.models.BaseGPT import BaseGPT
from dendritic.experiments.utils.loss_utils import compute_doubt_loss, compute_sequence_language_modeling_loss


import torch
from torch import nn


class DoubtAwareGPT(BaseGPT):
    """
    GPT model that uses MetaAwareBlock transformer blocks,
    incorporating doubt scalars into computations, and predicts
    its own future loss.
    """

    def __init__(self, *args, **kwargs):
        # Initialize the base MiniGPT
        super().__init__(*args, **kwargs)
        self.take_meta = min(kwargs.get("take_meta", 3), self.num_layers)
        if not isinstance(self.take_meta, int) or self.take_meta < 1:
            raise ValueError("take_meta must be a positive integer")
        # ADDED: The Head to predict the loss/doubt
        # Projects hidden_state [B, T, dim] -> [B, T, 1]
        self.loss_predictor = nn.Linear(self.embed_dim * self.take_meta, 1)

    @staticmethod
    def create_transformer_block(embed_dim, num_heads, dropout, mlp) -> AdaptiveMetaAwareBlock:
        # Use the custom block with AdaptiveLayer
        return AdaptiveMetaAwareBlock(embed_dim, num_heads, mlp, dropout)

    def forward(  # type: ignore[override]
        self,
        input_ids: torch.Tensor,
        doubt_scalars: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass of the DoubtAwareGPT model.

        Args:
            input_ids: Tensor of shape [batch_size, seq_len] with token indices
            doubt_scalars: Optional tensor of shape [batch_size, seq_len, 1]
                with doubt values. If None, defaults to zeros.

        Returns:
            Dictionary with:
                - "logits": Tensor of shape [batch_size, seq_len, vocab_size]
                - "loss_prediction": Tensor of shape [batch_size, seq_len]
        """
        B, T = input_ids.shape

        # Handle default doubt
        if doubt_scalars is None:
            # Default to 0.0 (neutral doubt)
            doubt_scalars = torch.zeros(
                (B, T, 1),
                device=input_ids.device,
                dtype=self.tok_emb.weight.dtype,
            )

        # Embeddings
        x, mask = self.embed_input_sequence(input_ids, T)

        # Transformer blocks (Passing doubt)
        x, all_outputs = self.forward_through_blocks(x, mask, doubt_scalars)

        # Final Normalization
        x = self.ln_f(x)

        # 1. Standard Token Logits
        logits = self.head(x)

        # 2. Predict Future Loss
        # Output shape: [B, T, 1] -> squeeze to [B, T] for easier loss calc later
        loss_prediction = self.loss_predictor(torch.cat(all_outputs, dim=-1)).squeeze(-1)

        return {"logits": logits, "loss_prediction": loss_prediction}

    def forward_through_blocks(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None,
        doubt_scalars: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        all_layer_outputs = []

        # Override to ensure blocks get the doubt scalar
        for block in self.blocks:
            x = block(x, doubt_scalar=doubt_scalars, attn_mask=mask)
            all_layer_outputs.append(x.detach())
        return x, all_layer_outputs[-self.take_meta :]  # For doubt predictor

    @staticmethod
    def two_pass_training_step(model: "DoubtAwareGPT", tokens_t, tokens_t_plus_1, alpha=1.0):
        """
        tokens_t: [B, SeqLen] - input sequence
        tokens_t_plus_1: [B] - next token after the sequence
        """
        B, SeqLen = tokens_t.shape

        # ===== PASS 1: Baseline (no doubt) =====
        with torch.no_grad():
            outputs_1 = model(tokens_t, doubt_scalars=None)
            logits_1 = outputs_1["logits"]  # [B, SeqLen, vocab]
            loss_pred_1 = outputs_1["loss_prediction"]  # [B, SeqLen]

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
            future_losses_baseline = compute_sequence_language_modeling_loss(
                logits_1,
                labels,
                reduction="none",
            )  # [B, SeqLen]

            # Prepare loss prediction input for pass 2 (shift right by 1)
            loss_pred_input = torch.cat(
                [
                    torch.zeros(B, 1, 1, device=tokens_t.device, dtype=loss_pred_1.dtype),
                    loss_pred_1[:, :-1].unsqueeze(-1),
                ],
                dim=1,
            )  # [B, SeqLen, 1]

        # ===== PASS 2: With loss prediction =====
        outputs_2 = model(tokens_t, doubt_scalars=loss_pred_input)
        logits_2 = outputs_2["logits"]  # [B, SeqLen, vocab]
        loss_pred_2 = outputs_2["loss_prediction"]  # [B, SeqLen]

        # Token prediction loss
        # Logits at positions 0..SeqLen-2 predict labels at positions 0..SeqLen-2
        # (We drop the last logit position since it doesn't have a "next" token to predict reliably in pass 2)
        loss_lm = compute_sequence_language_modeling_loss(
            logits_2[:, :-1],  # Only use up to SeqLen-1
            labels[:, :-1],  # Only use up to SeqLen-1
            reduction="mean",
        )

        # Loss prediction loss: positions 0..SeqLen-2 predict their future difficulty
        # (We drop the last loss prediction position since we don't have its "future" to measure)
        loss_doubt = compute_doubt_loss(
            loss_pred_2[:, :-1],  # [B, SeqLen-1]
            future_losses_baseline[:, :-1],  # [B, SeqLen-1]
        )

        total_loss = loss_lm + alpha * loss_doubt

        return {
            "total_loss": total_loss,
            "loss_lm": loss_lm,
            "loss_doubt": loss_doubt,
            "pred_loss_t": loss_pred_2[:, -2],  # Second-to-last position for logging
        }
