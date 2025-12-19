from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class DendriticStack(nn.Module):
    """
    General degree-k polynomial layer with optional diagonal squared terms.
    
    Computes:
        output = Wx + b + scale · W_out @ (∏_{i=1}^{k} (W_i x))
    
    where k = poly_degree. This represents a degree-k polynomial with rank r = poly_rank.
    The k-way product formulation (W_i ≠ W_j) provides richer interactions than symmetric
    alternatives and can approximate any degree-k polynomial of rank r.
    
    Additionally, an optional diagonal pathway captures squared terms:
        diag_out = W_diag_out @ (W_diag_in @ x)²
    
    This diagonal pathway is useful when inputs are independent (e.g., tabular features).
    
    Parameter count:
        Linear: input_dim * output_dim + (output_dim if bias else 0)
        Projections: poly_degree * poly_rank * input_dim
        Poly_out: output_dim * poly_rank
        Scale: 1
        Diagonal (if effective_diag_rank > 0): effective_diag_rank * input_dim + output_dim * effective_diag_rank + 1
        where effective_diag_rank is determined by independent_inputs and diag_rank.
    
    Args:
        input_dim: Input feature dimension
        output_dim: Output feature dimension
        poly_rank: Rank of polynomial (number of interaction terms)
        poly_degree: Degree of polynomial (k), default 3
        init_scale: Initial scale for polynomial pathway (default: 0.1)
        bias: Include bias in linear pathway (default: True)
        independent_inputs: Control 'auto' behavior for diagonal terms.
            - False (Default): Assumes inputs are distributed/entangled (e.g., embeddings,
              images, deep hidden states). Sets effective_diag_rank = max(4, poly_rank // 4) to save params.
            - True: Assumes inputs are disentangled/independent (e.g., tabular features,
              physical variables, network input layer). Sets effective_diag_rank = poly_rank
              to maximize capacity for individual x_i² terms.
        diag_rank: Rank for squared terms. Can be an integer, None, or 'auto'.
            If None or 'auto', the rank is determined by independent_inputs as above.
            If 0, the diagonal pathway is disabled.
    
    Note:
        DendriticLayer is a subclass with poly_degree=2, providing a quadratic layer.
    """

    @staticmethod
    def _compute_diag_rank(diag_rank, independent_inputs, poly_rank):
        """
        Compute effective diagonal rank given the arguments.
        """
        if diag_rank is None or diag_rank == "auto":
            return poly_rank if independent_inputs else max(4, poly_rank // 4)
        return diag_rank

    @classmethod
    def parameter_count(
        cls,
        input_dim: int,
        output_dim: int,
        poly_rank: int = 16,
        poly_degree: int = 3,
        independent_inputs: bool = False,
        diag_rank: int | Literal['auto'] = "auto",
        init_scale: float = 0.1,
        bias: bool = True,
        include_linear: bool = True,
    ) -> int:
        """
        Calculate total number of parameters for a DendriticStack with these dimensions.
        """
        # Compute effective diagonal rank
        effective_diag_rank = cls._compute_diag_rank(diag_rank, independent_inputs, poly_rank)
        
        # Linear pathway (optional)
        linear_params = 0
        if include_linear:
            linear_params = input_dim * output_dim + (output_dim if bias else 0)
        
        # Degree-k polynomial pathway
        projections_params = poly_degree * (poly_rank * input_dim)
        poly_out_params = output_dim * poly_rank
        scale_params = 1
        
        # Diagonal pathway
        diag_params = 0
        if isinstance(effective_diag_rank, int) and effective_diag_rank > 0:
            diag_params = effective_diag_rank * input_dim + output_dim * effective_diag_rank + 1
        
        return linear_params + projections_params + poly_out_params + scale_params + diag_params

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        poly_rank: int = 16,
        poly_degree: int = 3,
        independent_inputs: bool = False,
        diag_rank: int | Literal['auto'] = "auto",  # Changed default to flexible
        init_scale: float = 0.1,
        bias: bool = True,
    ):
        super().__init__()
        # Store dimensions for introspection
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.poly_rank = poly_rank
        self.poly_degree = poly_degree
        self.independent_inputs = independent_inputs
        self.init_scale = init_scale
        self.bias = bias
        
        # Validate diag_rank
        if diag_rank is not None and diag_rank != "auto" and diag_rank < 0:
            raise ValueError(f"diag_rank must be non-negative, got {diag_rank}")
        
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
        # Logic for auto-configuring the diagonal rank
        if diag_rank is None or diag_rank == "auto":
            if independent_inputs:
                self.diag_rank = poly_rank
            else:
                self.diag_rank = max(4, poly_rank // 4)
        else:
            self.diag_rank = diag_rank

        if isinstance(self.diag_rank, int) and self.diag_rank > 0:
            effective_diag_rank = self.diag_rank
            self.w_diag_in = nn.Parameter(torch.empty(effective_diag_rank, input_dim))
            self.w_diag_out = nn.Parameter(torch.empty(output_dim, effective_diag_rank))
            self.diag_scale = nn.Parameter(torch.tensor(init_scale))
            self.use_diagonal = True
        else:
            self.w_diag_in = None
            self.w_diag_out = None
            self.diag_scale = None
            self.use_diagonal = False
        self._reset_parameters()
    
    def _reset_parameters(self):
        self.linear.reset_parameters()
        for w in self.projections:
            nn.init.orthogonal_(w, gain=0.1)
        nn.init.orthogonal_(self.poly_out, gain=0.1)
        if self.use_diagonal:
            assert self.w_diag_in is not None and self.w_diag_out is not None
            nn.init.orthogonal_(self.w_diag_in, gain=0.1)
            nn.init.orthogonal_(self.w_diag_out, gain=0.1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        
        # k-way product → degree-k polynomial
        poly = F.linear(x, self.projections[0])
        for w in self.projections[1:]:
            poly = poly * F.linear(x, w)
        
        out = out + self.scale * F.linear(poly, self.poly_out)
        
        # Diagonal pathway
        if self.use_diagonal:
            assert self.w_diag_in is not None and self.w_diag_out is not None and self.diag_scale is not None
            h_diag = F.linear(x, self.w_diag_in)  # [batch, diag_rank]
            diag = F.linear(h_diag * h_diag, self.w_diag_out)  # [batch, output_dim]
            out = out + self.diag_scale * diag
        
        return out

try:
    from torch.serialization import add_safe_globals
    add_safe_globals([DendriticStack])
except ImportError:
    # Fallback for older PyTorch versions that don't have add_safe_globals
    pass
