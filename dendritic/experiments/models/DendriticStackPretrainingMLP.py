import torch
import torch.nn as nn


class DendriticStackPretrainingMLP(nn.Module):
    """
    Dendritic Stack MLP for pretraining comparison.

    Uses DendriticStack as the main transformation layer.
    """

    def __init__(
        self, embed_dim: int, hidden_dim: int, poly_rank: int = 16, dropout: float = 0.0
    ):
        super().__init__()

        # Import here to avoid circular imports
        from dendritic.layers.DendriticStack import DendriticStack

        self.stack = DendriticStack(
            input_dim=embed_dim,
            output_dim=hidden_dim,
            poly_rank=poly_rank,
        )
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stack(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
