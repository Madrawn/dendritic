from typing import Optional
from dendritic.layers.DendriticLayer import DendriticLayer


import torch
import torch.nn as nn
from gguf import Literal


class DendriticStack(nn.Module):
    """
    Efficient stack with bottleneck architecture.

    For degree-4 interactions, we don't need huge hidden dims.
    Use: input → bottleneck → output

    All parameters (poly_rank, diag_rank, independent_inputs, init_scale, bias, etc.)
    are passed identically to both internal DendriticLayer instances.
    This ensures consistent behavior and simplifies usage.

    Args:
        input_dim: Input feature dimension
        output_dim: Output feature dimension
        poly_rank: Rank for quadratic interactions (used for both layers)
        bottleneck_dim: Bottleneck hidden dimension (default: min(input_dim, output_dim)//2, at least 2x poly_rank)
        activation: Activation function between layers (default: GELU)
        independent_inputs, diag_rank, init_scale, bias, dropout: Passed to both DendriticLayer layers
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        poly_rank: int = 16,
        bottleneck_dim: Optional[int] = None,
        activation: Optional[nn.Module] = None,
        independent_inputs: bool = False,
        diag_rank: Optional[int] | Literal['auto'] = "auto",
        init_scale: float = 0.1,
        bias: bool = True,
        dropout: float = 0.0,
        # NEW ARGUMENT
        preserve_linear_path: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # ... (Existing logic for bottleneck_dim calculation) ...
        if bottleneck_dim is None:
            bottleneck_dim = poly_rank * 2

        # The Non-Linear Stack
        self.layer1 = DendriticLayer(
            input_dim, bottleneck_dim, poly_rank=poly_rank,
            independent_inputs=independent_inputs, diag_rank=diag_rank,
            init_scale=init_scale, bias=True # Bias needed for internal stack
        )
        self.act = activation if activation is not None else nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer2 = DendriticLayer(
            bottleneck_dim, output_dim, poly_rank=poly_rank,
            independent_inputs=independent_inputs, diag_rank=diag_rank,
            init_scale=init_scale, bias=bias
        )

        # NEW: The "Identity" Path
        # This will hold the original pre-trained weights
        self.preserve_linear_path = preserve_linear_path
        if self.preserve_linear_path:
            self.base_linear = nn.Linear(input_dim, output_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Compute the Polynomial/Non-linear Stack
        stack_out = self.layer1(x)
        stack_out = self.act(stack_out)
        stack_out = self.dropout(stack_out)
        stack_out = self.layer2(stack_out)

        # 2. Add the Base Linear Path (Original Weights)
        if self.preserve_linear_path:
            return self.base_linear(x) + stack_out

        return stack_out
    

try:
    from torch.serialization import add_safe_globals
    from .DendriticStack import DendriticStack # Adjust import path as needed
    add_safe_globals([DendriticStack])
except ImportError:
    # Fallback for older PyTorch versions that don't have add_safe_globals
    pass
