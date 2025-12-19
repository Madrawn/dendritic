from typing import Literal
import torch
import torch.nn as nn
import torch.nn.functional as F

from .DendriticStack import DendriticStack


class DendriticLayer(DendriticStack):
    """
    Optimized dendritic layer with efficient diagonal pathway.

    Instead of full W_diag @ (x²), use low-rank:
        diag_out = W_diag_out @ (W_diag_in @ x)²

    This captures the most important squared terms with far fewer params.

    Dendritic layer with quadratic cross-term interactions.

    Computes:
        output = Wx + b + scale · W_out @ ((W₁x) ⊙ (W₂x))

    This can represent any rank-r quadratic form where r = poly_rank.
    The asymmetric formulation (W₁ ≠ W₂) provides better optimization
    dynamics than the symmetric (Px)² alternative.

    Parameter count: input_dim * output_dim + output_dim  (linear)
                   + 2 * poly_rank * input_dim            (W₁, W₂)
                   + poly_rank * output_dim               (W_out)
                   + 1                                    (scale)

    Args:
        input_dim: Input feature dimension
        output_dim: Output feature dimension
        poly_rank: Rank of quadratic form (number of interaction terms)
        init_scale: Initial scale for polynomial pathway (default: 0.1)
        bias: Include bias in linear pathway (default: True)
        independent_inputs: Control 'auto' behavior for diagonal terms.
            - False (Default): Assumes inputs are distributed/entangled (e.g., embeddings, 
              images, deep hidden states). Sets diag_rank ~ poly_rank/4 to save params.
            - True: Assumes inputs are disentangled/independent (e.g., tabular features, 
              physical variables, network input layer). Sets diag_rank = poly_rank 
              to maximize capacity for individual x_i² terms.

        diag_rank: Explicitly set rank for squared terms. If None, behavior is 
                   controlled by 'independent_inputs'.

    """
    @classmethod
    def parameter_count(
        cls,
        input_dim: int,
        output_dim: int,
        poly_rank: int = 16,
        independent_inputs: bool = False,
        diag_rank: int | Literal['auto'] = "auto",
        init_scale: float = 0.1,
        bias: bool = True,
        *args,
        **kwargs
    ) -> int:
        return DendriticStack.parameter_count(
            input_dim=input_dim,
            output_dim=output_dim,
            poly_rank=poly_rank,
            poly_degree=2,
            independent_inputs=independent_inputs,
            diag_rank=diag_rank,
            init_scale=init_scale,
            bias=bias,
            include_linear=True
        )
    
    def __init__(self, input_dim, output_dim, poly_rank=16, independent_inputs=False, diag_rank="auto", init_scale=0.1, bias=True, *args, **kwargs):
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            poly_rank=poly_rank,
            poly_degree=2,
            independent_inputs=independent_inputs,
            diag_rank=diag_rank,
            init_scale=init_scale,
            bias=bias
        )
        # Alias for compatibility
        self.w1 = self.projections[0]
        self.w2 = self.projections[1]
        # poly_out, scale already exist
        # diagonal references already exist (w_diag_in, w_diag_out, diag_scale)
    
    def extra_repr(self) -> str:
        return (
            f'{self.input_dim}, {self.output_dim}, '
            f'poly_rank={self.poly_rank}, diag_rank={self.diag_rank}'
        )

try:
    from torch.serialization import add_safe_globals
    add_safe_globals([DendriticLayer])
except ImportError:
    # Fallback for older PyTorch versions that don't have add_safe_globals
    pass
