import torch
import torch.nn as nn
import torch.nn.functional as F


class InstabilityGatedAttention(nn.Module):
    """
    Custom attention where positions are weighted by their distributional stability.
    High drift → negative contribution (suppression)
    Low drift → positive contribution (trust)
    """

    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)

        # The instability probe: predicts drift from hidden states
        # This is trained to predict distributional variance
        self.drift_probe = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
        )

        # Maps drift score to contribution sign/magnitude
        # Output interpretation: -1 (suppress), 0 (ignore), +1 (trust)
        self.drift_to_gate = nn.Sequential(
            nn.Linear(1, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Tanh(),  # Output in [-1, +1]
        )

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim**-0.5

    def forward(self, x, drift_scores=None, causal_mask=True):
        """
        Args:
            x: Input tensor (batch, seq_len, embed_dim)
            drift_scores: Precomputed instability scores (batch, seq_len).
                          If None, computed by drift_probe.
            causal_mask: Whether to apply causal masking
        """
        batch, seq_len, _ = x.shape

        # Project Q, K, V
        Q = (
            self.W_q(x)
            .view(batch, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        K = (
            self.W_k(x)
            .view(batch, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        V = (
            self.W_v(x)
            .view(batch, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Compute attention scores (before softmax)
        attn_logits = Q @ K.transpose(-2, -1) * self.scale  # (batch, heads, seq, seq)

        # Apply causal mask
        if causal_mask:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device), diagonal=1
            ).bool()
            attn_logits = attn_logits.masked_fill(mask, float("-inf"))

        # Standard softmax attention weights
        attn_weights = F.softmax(attn_logits, dim=-1)  # (batch, heads, seq, seq)

        # === THE KEY MODIFICATION ===
        # Compute or use provided drift scores
        if drift_scores is None:
            drift_scores = self.drift_probe(x).squeeze(-1)  # (batch, seq_len)

        # Convert drift to gate values in [-1, +1]
        gate = self.drift_to_gate(drift_scores.unsqueeze(-1)).squeeze(
            -1
        )  # (batch, seq_len)

        # Apply gate to attention weights
        # gate > 0: trust this position (weight preserved or amplified)
        # gate < 0: suppress this position (weight negated)
        # gate ≈ 0: ignore this position

        # Broadcast gate to match attention shape: (batch, 1, 1, seq_len)
        gate = gate.unsqueeze(1).unsqueeze(1)

        # CRITICAL: This allows negative attention!
        gated_weights = attn_weights * gate

        # Dropout on the signed weights
        gated_weights = self.dropout(gated_weights)

        # Apply to values (negative weights will subtract from residual)
        out = gated_weights @ V  # (batch, heads, seq, head_dim)
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)

        return self.W_o(out), {
            "attn_weights": attn_weights,
            "gated_weights": gated_weights,
            "drift_scores": drift_scores,
            "gate": gate.squeeze(),
        }


def convert_to_inhibitory_attention(model, layer_indices=[12, 13, 14]):
    """
    Replaces standard attention in specified layers with InstabilityGatedAttention.

    Args:
        model: A HuggingFace LlamaForCausalLM (or similar)
        layer_indices: Which layers to modify (middle layers recommended)
    """
    for idx in layer_indices:
        old_attn = model.model.layers[idx].self_attn

        # Create new inhibitory attention with same dimensions
        new_attn = InstabilityGatedAttention(
            embed_dim=old_attn.hidden_size,
            num_heads=old_attn.num_heads,
            dropout=(
                old_attn.attention_dropout
                if hasattr(old_attn, "attention_dropout")
                else 0.0
            ),
        )

        # Initialize with weights from original (for Q, K, V, O projections)
        with torch.no_grad():
            new_attn.W_q.weight.copy_(old_attn.q_proj.weight)
            new_attn.W_k.weight.copy_(old_attn.k_proj.weight)
            new_attn.W_v.weight.copy_(old_attn.v_proj.weight)
            new_attn.W_o.weight.copy_(old_attn.o_proj.weight)
            # Biases if they exist
            if old_attn.q_proj.bias is not None:
                new_attn.W_q.bias.copy_(old_attn.q_proj.bias)
                
                # ... etc

        # The drift probe is randomly initialized — needs training

        # Replace
        model.model.layers[idx].self_attn = new_attn

    return model
