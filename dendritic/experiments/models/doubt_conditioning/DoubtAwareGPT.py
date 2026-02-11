from dendritic.experiments.models.doubt_conditioning.MetaAwareBlock import AdaptiveMetaAwareBlock
from dendritic.experiments.models.BaseGPT import BaseGPT
from dendritic.experiments.models.ModelConfig import ModelConfig


import torch
from torch import nn

from dendritic.layers import norm
import torch.nn.init as init


class DoubtAwareGPT(BaseGPT):
    """
    GPT model that uses MetaAwareBlock transformer blocks,
    incorporating doubt scalars into computations, and predicts
    its own future loss.
    """

    def __init__(self, config: ModelConfig | None = None, **kwargs):
        # Determine doubt_vector_dim: from kwargs, from config, or default 1
        if "doubt_vector_dim" in kwargs:
            self.doubt_vector_dim = kwargs.pop("doubt_vector_dim")
        elif config is not None and hasattr(config, "doubt_vector_dim"):
            self.doubt_vector_dim = config.doubt_vector_dim
        else:
            self.doubt_vector_dim = 1
        # Validate
        if not isinstance(self.doubt_vector_dim, int) or self.doubt_vector_dim < 1:
            raise ValueError("doubt_vector_dim must be a positive integer")
        # Initialize the base MiniGPT
        super().__init__(config=config, **kwargs)
        self.take_meta = min(kwargs.get("take_meta", self.num_layers // 2), self.num_layers)
        if not isinstance(self.take_meta, int) or self.take_meta < 1:
            raise ValueError("take_meta must be a positive integer")
        # ADDED: The Head to predict the loss/doubt
        # Projects hidden_state [B, T, embed_dim * take_meta] -> [B, T, doubt_vector_dim]
        self.loss_predictor = nn.Linear(self.embed_dim * self.take_meta, self.doubt_vector_dim)
        s = 3**0.5 * self.embed_dim**-0.5
        init.uniform_(self.loss_predictor.weight, -s, s)

    def create_transformer_block(self, embed_dim, num_heads, dropout, mlp) -> AdaptiveMetaAwareBlock:  # type: ignore[override]
        # Use the custom block with AdaptiveLayer, passing the instance's doubt_vector_dim
        return AdaptiveMetaAwareBlock(embed_dim, num_heads, mlp, dropout, self.doubt_vector_dim)

    def forward(  # type: ignore[override]
        self,
        input_ids: torch.Tensor,
        doubt_scalars: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass of the DoubtAwareGPT model.

        Args:
            input_ids: Tensor of shape [batch_size, seq_len] with token indices
            doubt_scalars: Optional tensor of shape [batch_size, seq_len, doubt_vector_dim]
                with doubt values. If None, defaults to zeros.

        Returns:
            Dictionary with:
                - "logits": Tensor of shape [batch_size, seq_len, vocab_size]
                - "loss_prediction": Tensor of shape [batch_size, seq_len, doubt_vector_dim]
        """
        B, T = input_ids.shape

        # Handle default doubt
        if doubt_scalars is None:
            # Default to 0.0 (neutral doubt)
            doubt_scalars = torch.zeros(
                (B, T, self.doubt_vector_dim),
                device=input_ids.device,
                dtype=self.tok_emb.weight.dtype,
            )

        # Embeddings
        x, mask = self.embed_input_sequence(input_ids, T)

        # Transformer blocks (Passing doubt)
        x, all_outputs = self.forward_through_blocks(x, mask, doubt_scalars)

        # Final Normalization
        x = norm(x)

        # 1. Standard Token Logits
        logits = self.head(x)

        # 2. Predict Future Loss
        # Output shape: [B, T, doubt_vector_dim]
        loss_prediction = self.loss_predictor(torch.cat(all_outputs, dim=-1))

        return {"logits": logits, "loss_prediction": loss_prediction}

    def forward_through_blocks(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None,
        doubt_scalars: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        all_layer_outputs = []

        # Override to ensure blocks get the doubt scalar
        for block in self.blocks:
            x = block(x, doubt_scalar=doubt_scalars, attn_mask=mask)
            all_layer_outputs.append(x.detach())
        return x, all_layer_outputs[-self.take_meta :]  # For doubt predictor
