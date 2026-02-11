from dataclasses import dataclass
from typing import Literal


@dataclass
class ModelConfig:
    """Configuration for GPT model architecture and MLP.

    This dataclass encapsulates all parameters needed to construct a BaseGPT model,
    reducing the number of arguments in the constructor from 10 to 1.

    Attributes:
        vocab_size: Size of the vocabulary
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        max_seq_len: Maximum sequence length
        hidden_dim: Hidden dimension of the MLP
        mlp_type: Type of MLP ("standard" or "dendritic")
        poly_rank: Polynomial rank for dendritic MLP (ignored for standard)
        poly_degree: Polynomial degree for dendritic MLP (ignored for standard)
        dropout: Dropout probability
    """

    vocab_size: int
    embed_dim: int
    num_heads: int
    num_layers: int
    max_seq_len: int
    hidden_dim: int
    mlp_type: Literal["standard", "dendritic"] = "standard"
    poly_rank: int = 16
    poly_degree: int = 3
    dropout: float = 0.0
    doubt_vector_dim: int = 1  # Dimension of the doubt vector (1 = scalar, >1 = vectorized)
