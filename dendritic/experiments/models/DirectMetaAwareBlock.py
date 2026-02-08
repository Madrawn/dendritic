import torch.nn as nn


class DirectMetaAwareBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_module, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = mlp_module
        self.dropout = nn.Dropout(dropout)

        # Direct doubt pathway
        self.doubt_to_attn = nn.Linear(1, embed_dim)
        self.doubt_to_mlp = nn.Linear(1, embed_dim)

        # Initialize small but non-zero
        nn.init.normal_(self.doubt_to_attn.weight, std=0.02)
        nn.init.normal_(self.doubt_to_mlp.weight, std=0.02)

    def forward(self, x, doubt_scalar, attn_mask=None):
        # Attention with doubt
        doubt_attn = self.doubt_to_attn(doubt_scalar)
        norm_x = self.ln1(x)
        attn_out, _ = self.attn(norm_x + doubt_attn, norm_x, norm_x, attn_mask=attn_mask)
        x = x + self.dropout(attn_out)

        # MLP with doubt
        doubt_mlp = self.doubt_to_mlp(doubt_scalar)
        norm_x = self.ln2(x)
        x = x + self.dropout(self.mlp(norm_x + doubt_mlp))

        return x
