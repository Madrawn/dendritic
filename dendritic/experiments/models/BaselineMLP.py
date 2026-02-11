import torch
import torch.nn as nn


class BaselineMLP(nn.Module):
    """Standard MLP block for transformer."""

    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class SwiGLUMLP(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        # SwiGLU needs 2x hidden_dim because it splits into two branches
        self.fc1 = nn.Linear(embed_dim, hidden_dim * 2)  # expand to 2x
        self.fc2 = nn.Linear(hidden_dim, embed_dim)  # project back
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, v = self.fc1(x).chunk(2, dim=-1)  # split into two branches
        # u: "value" branch, v: "gate" branch
        x = u * torch.nn.functional.silu(v)  # gated activation
        x = self.dropout(x)
        x = self.fc2(x)
        return x
