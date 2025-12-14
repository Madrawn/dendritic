from typing import Optional
from layers.DendriticLayer import DendriticLayer


import torch
import torch.nn as nn


class DendriticResidualBlock(nn.Module):
    """
    Residual block with dendritic computation. Input/output dims must match.

    output = x + DendriticMLP(x)
    """

    def __init__(
        self,
        dim: int,
        poly_rank: int = 16,
        expansion: float = 2.0,
        activation: Optional[nn.Module] = None,
        dropout: float = 0.0,
    ):
        super().__init__()

        hidden = int(dim * expansion)

        self.layer1 = DendriticLayer(dim, hidden, poly_rank=poly_rank)
        self.act = activation if activation is not None else nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer2 = nn.Linear(hidden, dim)  # Output projection can be linear

        # Layer scale (helps training deep networks)
        self.scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.layer1(x)
        h = self.act(h)
        h = self.dropout(h)
        h = self.layer2(h)
        return x + self.scale * h