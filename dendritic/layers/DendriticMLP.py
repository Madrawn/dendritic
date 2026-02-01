from typing import Optional
from dendritic.layers.DendriticLayer import DendriticLayer


import torch
import torch.nn as nn


class DendriticMLP(nn.Module):
    """
    MLP block with dendritic computation on the input projection.

    Drop-in replacement for transformer MLP:
        Standard:  x → Linear → Act → Linear → out
        Dendritic: x → DendriticLayer → Act → Linear → out

    The dendritic layer captures quadratic interactions in the input
    before the nonlinear activation expands to hidden_dim.

    Args:
        embed_dim: Input/output embedding dimension
        hidden_dim: Hidden layer dimension (typically 4x embed_dim)
        poly_rank: Rank for quadratic interactions
        activation: Activation function (default: GELU)
        dropout: Dropout probability (default: 0.0)
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        poly_rank: Optional[int] = None,
        activation: Optional[nn.Module] = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        if poly_rank is None:
            poly_rank = max(8, embed_dim // 16)

        self.fc1 = DendriticLayer(embed_dim, hidden_dim, poly_rank=poly_rank)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.act = activation if activation is not None else nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
