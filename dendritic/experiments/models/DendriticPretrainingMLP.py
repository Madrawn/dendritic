import torch
import torch.nn as nn


class DendriticPretrainingMLP(nn.Module):
    """
    Dendritic MLP for pretraining comparison.

    Uses DendriticLayer for fc1 (input projection) where quadratic
    interactions are most valuable.
    """
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        poly_rank: int = 16,
        dropout: float = 0.0
    ):
        super().__init__()

        # Import here to avoid circular imports
        from dendritic.layers.DendriticLayer import DendriticLayer

        self.fc1 = DendriticLayer(
            embed_dim,
            hidden_dim,
            poly_rank=poly_rank,
            init_scale=0.1
        )
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x