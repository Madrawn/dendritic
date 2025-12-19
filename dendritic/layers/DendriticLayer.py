import torch
import torch.nn as nn
import torch.nn.functional as F
from gguf import Literal

class DendriticLayer(nn.Module):
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
        independent_inputs: bool = False,
        diag_rank: int | Literal['auto'] = "auto",
        init_scale: float = 0.1,
        bias: bool = True,
        *args,
        **kwargs
    ) -> int:
        """
        Calculate total number of parameters for a DendriticLayer with these dimensions.
        """
        # Compute effective diagonal rank
        effective_diag_rank = cls._compute_diag_rank(diag_rank, independent_inputs, poly_rank)
        
        # Linear pathway
        linear_params = input_dim * output_dim + (output_dim if bias else 0)
        
        # Cross-term pathway
        cross_params = 2 * poly_rank * input_dim  # W1, W2
        cross_params += output_dim * poly_rank     # poly_out
        cross_params += 1                          # scale
        
        # Diagonal pathway
        diag_params = 0
        if isinstance(effective_diag_rank, int) and effective_diag_rank > 0:
            diag_params = effective_diag_rank * input_dim    # W_diag_in
            diag_params += output_dim * effective_diag_rank  # W_diag_out
            diag_params += 1                                 # diag_scale
        
        return linear_params + cross_params + diag_params

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        poly_rank: int = 16,
        independent_inputs: bool = False,
        diag_rank: int | Literal['auto'] = "auto",  # Changed default to flexible
        init_scale: float = 0.1,
        bias: bool = True,
        *args,
        **kwargs
    ):
        if init_scale == 0.0:
            raise ValueError("init_scale must be non-zero to enable polynomial pathway.")
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.poly_rank = poly_rank

        # Logic for auto-configuring the diagonal rank
        if diag_rank is None or diag_rank == "auto":
            if independent_inputs:
                self.diag_rank = poly_rank
            else:
                self.diag_rank = max(4, poly_rank // 4)
        else:
            self.diag_rank = diag_rank

        # Linear pathway
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)

        # Cross-term pathway: (W₁x) ⊙ (W₂x) → W_out
        self.w1 = nn.Parameter(torch.empty(poly_rank, input_dim))
        self.w2 = nn.Parameter(torch.empty(poly_rank, input_dim))
        self.poly_out = nn.Parameter(torch.empty(output_dim, poly_rank))
        self.scale = nn.Parameter(torch.tensor(init_scale))

        # Low-rank diagonal pathway: (W_diag_in @ x)² → W_diag_out
        # This computes sum of squared projections
        if isinstance(self.diag_rank, int) and self.diag_rank > 0:
            self.w_diag_in = nn.Parameter(torch.empty(self.diag_rank, input_dim))
            self.w_diag_out = nn.Parameter(torch.empty(output_dim, self.diag_rank))
            self.diag_scale = nn.Parameter(torch.tensor(init_scale))
            self.use_diagonal = True
        else:
            self.use_diagonal = False

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.linear.reset_parameters()
        nn.init.orthogonal_(self.w1, gain=0.1)
        nn.init.orthogonal_(self.w2, gain=0.1)
        nn.init.orthogonal_(self.poly_out, gain=0.1)
        if self.use_diagonal:
            nn.init.orthogonal_(self.w_diag_in, gain=0.1)
            nn.init.orthogonal_(self.w_diag_out, gain=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Linear pathway
        out = self.linear(x)

        # Cross-term polynomial pathway
        h1 = F.linear(x, self.w1)  # [batch, poly_rank]
        h2 = F.linear(x, self.w2)  # [batch, poly_rank]
        poly = F.linear(h1 * h2, self.poly_out)  # [batch, output_dim]
        out = out + self.scale * poly

        # Low-rank diagonal pathway
        if self.use_diagonal:
            h_diag = F.linear(x, self.w_diag_in)  # [batch, diag_rank]
            diag = F.linear(h_diag * h_diag, self.w_diag_out)  # [batch, output_dim]
            out = out + self.diag_scale * diag

        return out

    def extra_repr(self) -> str:
        return (
            f'{self.input_dim}, {self.output_dim}, '
            f'poly_rank={self.poly_rank}, diag_rank={self.diag_rank}'
        )
    

try:
    from torch.serialization import add_safe_globals
    from .DendriticLayer import DendriticLayer # Adjust import path as needed
    add_safe_globals([DendriticLayer])
except ImportError:
    # Fallback for older PyTorch versions that don't have add_safe_globals
    pass
