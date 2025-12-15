from .BaselineMLP import BaselineMLP
from .DendriticPretrainingMLP import DendriticPretrainingMLP
from .DendriticStackPretrainingMLP import DendriticStackPretrainingMLP
from .TransformerBlock import TransformerBlock


import torch
import torch.nn as nn


from typing import Dict, Optional


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
        mlp_type: str = "baseline",  # "baseline", "dendritic", or "dendritic_stack"
        poly_rank: int = 16,
        dropout: float = 0.0
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
            if mlp_type == "baseline" or mlp_type == "baseline_wave":
                mlp = BaselineMLP(embed_dim, hidden_dim, dropout)
            elif mlp_type == "dendritic":
                mlp = DendriticPretrainingMLP(
                    embed_dim, hidden_dim, poly_rank, dropout
                )
            elif mlp_type == "dendritic_stack":
                mlp = DendriticStackPretrainingMLP(
                    embed_dim, hidden_dim, poly_rank, dropout
                )
            else:
                raise ValueError(f"Unknown mlp_type: {mlp_type}")

            block = TransformerBlock(embed_dim, num_heads, mlp, dropout)
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
            torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        B, T = input_ids.shape

        # Embeddings
        tok_emb = self.tok_emb(input_ids)
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        pos_emb = self.pos_emb(pos)
        x = self.drop(tok_emb + pos_emb)

        # Causal mask for this sequence length
        mask = self.causal_mask[:T, :T] # type: ignore DO NOT TOUCH!

        # Transformer blocks
        for block in self.blocks:
            x = block(x, attn_mask=mask)

        x = self.ln_f(x)
        logits = self.head(x)

        output = {"logits": logits}

        # Compute loss if labels provided
        if labels is not None:
            # FIX: Shift logits and labels so that x_t predicts x_{t+1}
            # logits[..., :-1, :] predicts labels[..., 1:]
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
            output["loss"] = loss

        return output