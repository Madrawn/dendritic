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
        input_dim,
        output_dim,
        poly_rank=16,
        poly_degree=3,
        independent_inputs=False,
        diag_rank: int | Literal["auto"] = "auto",
        bias=True,
        include_linear=True,
    ) -> int:
        # effective diag
        if diag_rank is None or diag_rank == "auto":
            diag = poly_rank if independent_inputs else max(4, poly_rank // 4)
        else:
            diag = int(diag_rank)

        linear_params = (
            (input_dim * output_dim + (output_dim if bias else 0))
            if include_linear
            else 0
        )
        projections_params = poly_degree * (poly_rank * input_dim)
        projection_bias_params = poly_degree * poly_rank
        poly_out_params = output_dim * poly_rank
        diag_params = (diag * input_dim + output_dim * diag) if diag > 0 else 0
        gate_params = 1 + (1 if diag > 0 else 0)  # alpha + optional alpha_diag

        return (
            linear_params
            + projections_params
            + projection_bias_params
            + poly_out_params
            + diag_params
            + gate_params
        )

    @classmethod
    def from_linear(cls, linear: nn.Linear, **kwargs):
        layer = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            **kwargs,
        )
        with torch.no_grad():
            layer.linear.weight.copy_(linear.weight)
            if linear.bias is not None:
                layer.linear.bias.copy_(linear.bias)
        return layer

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        poly_rank: int = 16,
        poly_degree: int = 3,
        independent_inputs: bool = False,
        diag_rank: int | Literal["auto"] = "auto",  # Changed default to flexible
        bias: bool = True,
    ):
        super().__init__()
        # Store dimensions for introspection
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.poly_rank = poly_rank
        self.poly_degree = poly_degree
        self.independent_inputs = independent_inputs
        self.bias = bias
        self.alpha_grad_boost = 1e-3  # not a parameter
        self.alpha = nn.Parameter(torch.zeros(1))

        # Validate diag_rank
        if diag_rank is not None and diag_rank != "auto" and diag_rank < 0:
            raise ValueError(f"diag_rank must be non-negative, got {diag_rank}")

        # Linear pathway (preserved for initialization from pretrained)
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)

        # Degree-k pathway via k-way product

        self.projections = nn.Parameter(torch.empty(poly_degree, poly_rank, input_dim))
        self.proj_biases = nn.Parameter(torch.zeros(poly_degree, poly_rank))

        self.poly_out = nn.Parameter(torch.empty(output_dim, poly_rank))
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
            self.use_diagonal = True
        else:
            self.w_diag_in = None
            self.w_diag_out = None
            self.use_diagonal = False

        if self.use_diagonal:
            self.alpha_diag = nn.Parameter(torch.zeros(1))

        self._reset_parameters()

    def _reset_parameters(self):
        self.linear.reset_parameters()

        self.alpha.data.fill_(1e-3)
        if self.use_diagonal:
            self.alpha_diag.data.fill_(1e-3)

        self.proj_biases.data.normal_(0.0, 1e-3)

        proj_gain = 10 ** (
            -1.0 / self.poly_degree
        )  # keeps product var ≈ 1e-2 if inputs~N(0,1)
        for w in self.projections:
            nn.init.orthogonal_(w, gain=proj_gain)

        nn.init.orthogonal_(self.poly_out, gain=0.1)
        if self.use_diagonal:
            assert self.w_diag_in is not None and self.w_diag_out is not None
            nn.init.orthogonal_(self.w_diag_in, gain=0.1)
            nn.init.orthogonal_(self.w_diag_out, gain=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Linear path: preserves leading dims
        out = self.linear(x)

        # Degree-k product via vectorized projections
        # x: [..., D], projections: [K, R, D], biases: [K, R]
        logits = (
            torch.einsum("...d,krd->...kr", x, self.projections) + self.proj_biases
        )  # [..., K, R]
        poly = logits.prod(dim=-2)  # [..., R]
        poly_out = F.linear(poly, self.poly_out)  # [..., O]

        # Gated sum with gradient-boost trick
        alpha_eff = self.alpha + self.alpha_grad_boost
        out = out + self.alpha * poly_out
        out = out + (alpha_eff - self.alpha).detach() * poly_out

        if self.use_diagonal:
            assert self.w_diag_in is not None and self.w_diag_out is not None
            # Diagonal squared terms
            h = F.linear(x, self.w_diag_in)  # [..., R_d]
            diag = F.linear(h * h, self.w_diag_out)  # [..., O]
            alpha_diag_eff = self.alpha_diag + self.alpha_grad_boost
            out = out + self.alpha_diag * diag
            out = out + (alpha_diag_eff - self.alpha_diag).detach() * diag

        return out


try:
    from torch.serialization import add_safe_globals

    add_safe_globals([DendriticStack])
except ImportError:
    # Fallback for older PyTorch versions that don't have add_safe_globals
    pass
