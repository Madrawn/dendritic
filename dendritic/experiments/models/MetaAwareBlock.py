import torch
import torch.nn as nn


class AdaptiveMetaAwareBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_module: nn.Module,
        dropout: float = 0.0,
    ):
        super().__init__()
        # Replace standard LayerNorm with your Adaptive Layer
        self.ln1 = AdaptiveLayer(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        self.ln2 = AdaptiveLayer(embed_dim)
        self.mlp = mlp_module
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, doubt_scalar, attn_mask: torch.Tensor | None = None):
        # doubt_scalar shape: [batch_size, seq_len, 1]

        # 1. Attention Sub-layer
        # The doubt modulates the norm BEFORE attention
        norm_x = self.ln1(x, doubt_scalar)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x, attn_mask=attn_mask)
        x = x + self.dropout(attn_out)

        # 2. MLP Sub-layer
        # The doubt modulates the norm BEFORE the MLP
        norm_x = self.ln2(x, doubt_scalar)
        x = x + self.dropout(self.mlp(norm_x))

        return x


class AdaptiveLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Optimization: elementwise_affine=False
        # We don't need LayerNorm's internal gamma/beta because
        # our AdaLN scale/shift will handle that job.
        # Reduces params slightly and avoids fighting between two shifts.
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)

        self.doubt_project = nn.Linear(1, 2 * dim)

        # CRITICAL: Zero initialization
        # This ensures the model starts as a standard Transformer
        nn.init.zeros_(self.doubt_project.weight)
        nn.init.zeros_(self.doubt_project.bias)

    def forward(self, x, doubt_scalar):
        normalized_x = self.norm(x)
        scale, shift = self.doubt_project(doubt_scalar).chunk(2, dim=-1)
        return normalized_x * (1 + scale) + shift
