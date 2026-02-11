import torch
import torch.nn as nn

from dendritic.layers import norm


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_module: nn.Module,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # Replace MultiheadAttention with manual Q, K, V projections
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.mlp = mlp_module
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, C = x.shape

        # Self-attention with QK normalization
        x_norm = norm(x)

        # Compute Q, K, V
        qkv = self.qkv_proj(x_norm)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each is (B, num_heads, T, head_dim)

        q = norm(q)
        k = norm(k)

        # Scaled dot-product attention
        if attn_mask is not None:
            # Custom mask (for instruction tuning, padding, etc.)
            attn_out = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                is_causal=False,  # Mask already encodes causality
                dropout_p=self.dropout.p if self.training else 0.0,
            )
        else:
            # Default causal (for standard LM training)
            attn_out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, is_causal=True, dropout_p=self.dropout.p if self.training else 0.0
            )

        # Reshape and project back
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(B, T, C)
        attn_out = self.out_proj(attn_out)

        x = x + self.dropout(attn_out)

        # MLP with residual
        x = x + self.dropout(self.mlp(norm(x)))
        return x
