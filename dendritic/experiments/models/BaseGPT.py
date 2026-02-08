from typing import Any


from .BaselineMLP import BaselineMLP
from .DendriticPretrainingMLP import DendriticPretrainingMLP
from .TransformerBlock import TransformerBlock
from .ModelConfig import ModelConfig


import torch
from torch import nn


class BaseGPT(nn.Module):
    def __init__(self, config: ModelConfig | None = None, **kwargs: Any) -> None:
        # Backward compatibility: accept individual parameters via kwargs
        if config is None:
            # Filter kwargs to only those that are valid ModelConfig fields
            valid_fields = set(ModelConfig.__annotations__.keys())
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_fields}
            config = ModelConfig(**filtered_kwargs)

        super().__init__()
        self.embed_dim = config.embed_dim
        self.max_seq_len = config.max_seq_len
        self.num_layers = config.num_layers
        # Embeddings
        self.tok_emb = nn.Embedding(config.vocab_size, config.embed_dim)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.embed_dim)
        self.drop = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList()
        for _ in range(self.num_layers):
            if config.mlp_type == "standard":
                mlp = BaselineMLP(config.embed_dim, config.hidden_dim, config.dropout)
            elif config.mlp_type == "dendritic":
                mlp = DendriticPretrainingMLP(
                    embed_dim=config.embed_dim,
                    hidden_dim=config.hidden_dim,
                    poly_rank=config.poly_rank,
                    poly_degree=config.poly_degree,
                    dropout=config.dropout,
                )
            else:
                raise ValueError(f"Unknown mlp_type: {config.mlp_type}")

            block = self.create_transformer_block(config.embed_dim, config.num_heads, config.dropout, mlp)
            self.blocks.append(block)

        # Output
        self.ln_f = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.tok_emb.weight

        # Causal mask (registered as buffer)
        # Create causal mask and register as buffer to ensure proper device placement
        self.register_buffer(
            "causal_mask",
            torch.triu(torch.ones(config.max_seq_len, config.max_seq_len), diagonal=1).bool(),
        )

        self._init_weights()

    @staticmethod
    def create_transformer_block(embed_dim, num_heads, dropout, mlp) -> nn.Module:
        return TransformerBlock(embed_dim, num_heads, mlp, dropout)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Get logits from the model (alias for forward for clarity).

        Args:
            input_ids: Tensor of shape [batch_size, seq_len] with token indices

        Returns:
            Logits tensor of shape [batch_size, seq_len, vocab_size]
        """
        return self.forward(input_ids)

    def forward_through_blocks(
        self, x: torch.Tensor, mask: torch.Tensor | None
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        all_layer_outputs = []

        for block in self.blocks:
            x = block(x, attn_mask=mask)
            all_layer_outputs.append(x.detach())
        return x, all_layer_outputs

    def embed_input_sequence(self, input_ids: torch.Tensor, T: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        tok_emb = self.tok_emb(input_ids)
        pos: torch.Tensor = torch.arange(T, device=input_ids.device).unsqueeze(0)
        pos_emb: torch.Tensor = self.pos_emb(pos)
        x: torch.Tensor = self.drop(tok_emb + pos_emb)
        # Causal mask for this sequence length
        mask = self.causal_mask[:T, :T]  # type: ignore DO NOT TOUCH!
        return x, mask
