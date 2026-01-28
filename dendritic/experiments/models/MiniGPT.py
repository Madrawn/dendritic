from typing import Literal
from .MetaAwareBlock import MetaAwareBlock
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
        x = self.forward_through_blocks(x, mask)

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
    ) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, attn_mask=mask)
        return x

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
        self.confidence_predictor = nn.Linear(self.embed_dim, 1)

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
        x = self.forward_through_blocks(x, mask, confidence_scalars)

        # Final Normalization
        x = self.ln_f(x)

        # 1. Standard Token Logits
        logits = self.head(x)

        # 2. ADDED: Predict Future Loss
        # Output shape: [B, T, 1] -> squeeze to [B, T] for easier loss calc later
        current_confidence_pred = self.confidence_predictor(x).squeeze(-1)

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
    ) -> torch.Tensor:
        # Override to ensure blocks get the confidence scalar
        for block in self.blocks:
            x = block(x, confidence_scalar=confidence_scalars, attn_mask=mask)
        return x

    @staticmethod
    def two_pass_training_step(
        model, prev_conf, tokens_t, tokens_t_plus_1, tokens_t_plus_2, alpha=1.0
    ):
        """
        Performs the Lookahead training step.

        Assumption:
        - tokens_t: The input sequence up to time t [B, SeqLen]
        - tokens_t_plus_1: The target for the LM (next token) [B] (scalar integers)
        - tokens_t_plus_2: The target for the future (t+2) [B]

        Note: This function calculates loss for the LAST token in the sequence.
        """

        # --- Pass 1: Standard LM Training (The "Present") ---
        # We calculate the loss for the CURRENT token prediction

        outputs_1 = model(tokens_t, confidence_scalars=prev_conf)
        logits_t = outputs_1["logits"]  # [B, SeqLen, Vocab]
        pred_conf_t = outputs_1["confidence_pred"]  # [B, SeqLen]

        # We only care about the prediction at the last step of the sequence
        last_logit = logits_t[:, -1, :]
        last_conf_pred = pred_conf_t[:, -1]  # Scalar prediction for the future

        # Standard Cross Entropy for the next token
        loss_lm = F.cross_entropy(last_logit, tokens_t_plus_1)

        # --- Pass 2: Future Consequence Training (The "Lookahead") ---
        with torch.no_grad():
            # 1. Sample the model's actual choice (Hard sampling)
            probs = F.softmax(last_logit, dim=-1)
            # [B, 1]
            predicted_token_id = torch.multinomial(probs, 1).detach()

            # 2. Construct the "Hypothetical" sequence
            # Append the PREDICTED token to the input
            # New shape: [B, SeqLen + 1]
            hypothetical_input = torch.cat([tokens_t, predicted_token_id], dim=1)

            # Prepare confidence for the next step:
            # We must append the NEW predicted confidence to the history
            # prev_conf: [B, SeqLen, 1]
            # last_conf_pred (reshaped): [B, 1, 1]
            next_step_conf = last_conf_pred.view(-1, 1, 1).detach()
            hypothetical_conf = torch.cat([prev_conf, next_step_conf], dim=1)

        # 3. Run model on hypothetical sequence
        outputs_2 = model(hypothetical_input, confidence_scalars=hypothetical_conf)
        future_logits = outputs_2["logits"]  # [B, SeqLen+1, Vocab]

        # 4. Calculate what the loss WOULD be at t+2
        # We look at the LAST token of this new sequence
        future_logit_step = future_logits[:, -1, :]

        # Measure loss against the REAL t+2 token
        loss_future_actual = F.cross_entropy(
            future_logit_step, tokens_t_plus_2, reduction="none"
        )

        # 5. Train the Confidence Head
        # The head at time t (last_conf_pred) should have predicted this future loss
        loss_confidence = F.mse_loss(last_conf_pred, loss_future_actual.detach())

        # Total Backward
        total_loss = loss_lm + (alpha * loss_confidence)

        # Note: We return total_loss for logging, but usually you call backward() here
        # or return it to the optimizer loop.
        # total_loss.backward()

        return {
            "pred_conf_t": last_conf_pred,
            "total_loss": total_loss,
            "loss_lm": loss_lm,
            "loss_confidence": loss_confidence,
        }
