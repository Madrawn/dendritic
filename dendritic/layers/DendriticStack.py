from typing import Optional
from dendritic.layers.DendriticLayer import DendriticLayer


import torch
import torch.nn as nn
from gguf import Literal
import torch.nn.functional as F


class DendriticStack(nn.Module):
    """
    Clean degree-k polynomial, drop-in Linear replacement.
    Activation comes from the surrounding architecture.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        poly_rank: int = 16,
        poly_degree: int = 3,
        independent_inputs: bool = False,
        diag_rank: Optional[int] | Literal['auto'] = "auto",  # Changed default to flexible
        init_scale: float = 0.1,
        bias: bool = True,

    ):
        super().__init__()
        # Linear pathway (preserved for initialization from pretrained)
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        
        # Degree-k pathway via k-way product
        self.projections = nn.ParameterList([
            nn.Parameter(torch.empty(poly_rank, input_dim))
            for _ in range(poly_degree)
        ])
        self.poly_out = nn.Parameter(torch.empty(output_dim, poly_rank))
        self.scale = nn.Parameter(torch.tensor(init_scale))
                # Optional diagonal        
                # # Logic for auto-configuring the diagonal rank
        if diag_rank is None or diag_rank == "auto":
            if independent_inputs:
                self.diag_rank = poly_rank
            else:
                self.diag_rank = max(4, poly_rank // 4)
        else:
            self.diag_rank = diag_rank

        if isinstance(self.diag_rank, int) and self.diag_rank > 0:
            diag_rank = max(4, poly_rank // 4)
            self.w_diag_in = nn.Parameter(torch.empty(diag_rank, input_dim))
            self.w_diag_out = nn.Parameter(torch.empty(output_dim, diag_rank))
            self.diag_scale = nn.Parameter(torch.tensor(init_scale))
            self.use_diagonal = True
        else:
            self.w_diag_in = None
            self.w_diag_out = None
            self.diag_scale = None
        self._reset_parameters()
    
    def _reset_parameters(self):
        self.linear.reset_parameters()
        for w in self.projections:
            nn.init.orthogonal_(w, gain=0.1)
        nn.init.orthogonal_(self.poly_out, gain=0.1)
        if self.use_diagonal:
            nn.init.orthogonal_(self.w_diag_in, gain=0.1)
            nn.init.orthogonal_(self.w_diag_out, gain=0.1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        
        # k-way product → degree-k polynomial
        poly = F.linear(x, self.projections[0])
        for w in self.projections[1:]:
            poly = poly * F.linear(x, w)
        
        return out + self.scale * F.linear(poly, self.poly_out)    

try:
    from torch.serialization import add_safe_globals
    from .DendriticStack import DendriticStack # Adjust import path as needed
    add_safe_globals([DendriticStack])
except ImportError:
    # Fallback for older PyTorch versions that don't have add_safe_globals
    pass
