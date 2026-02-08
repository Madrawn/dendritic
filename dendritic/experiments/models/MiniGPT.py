from dendritic.experiments.models.BaseGPT import BaseGPT


import torch


class MiniGPT(BaseGPT):
    """
    Minimal GPT for pretraining .

    Supports baseline, dendritic, and dendritic_stack MLP variants.
    """

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MiniGPT model.

        Args:
            input_ids: Tensor of shape [batch_size, seq_len] with token indices

        Returns:
            Logits tensor of shape [batch_size, seq_len, vocab_size]
        """
        B, T = input_ids.shape

        # Embeddings
        x, mask = self.embed_input_sequence(input_ids, T)

        # Transformer blocks
        x, _ = self.forward_through_blocks(x, mask)

        # Final normalization and projection
        x = self.ln_f(x)
        logits = self.head(x)

        return logits
