from dendritic.experiments.models.DoubtAwareGPT import DoubtAwareGPT
from dendritic.experiments.models.ModelConfig import ModelConfig


import torch
from torch import nn


class SelfConditionedGPT(nn.Module):
    def __init__(self, config: ModelConfig, bound_fn: str = "tanh", take_meta: int = 3):
        super().__init__()
        self.core = DoubtAwareGPT(config=config, take_meta=take_meta)
        self.max_seq_len = config.max_seq_len
        self.bound = self._get_bound_fn(bound_fn)

    def forward(self, input_ids: torch.Tensor, doubt_scalars: torch.Tensor | None = None) -> torch.Tensor:
        if doubt_scalars is not None:
            return self.core(input_ids, doubt_scalars=doubt_scalars)["logits"]

        # Pass 1: Generate conditioning signal
        pass1_out = self.core(input_ids, doubt_scalars=None)
        doubt_signal = self.bound(pass1_out["loss_prediction"]).unsqueeze(-1)

        # Pass 2: Conditioned prediction
        return self.core(input_ids, doubt_scalars=doubt_signal)["logits"]

    def forward_with_diagnostics(self, input_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """For analysis: returns both passes' outputs."""
        pass1_out = self.core(input_ids, doubt_scalars=None)
        doubt_signal = self.bound(pass1_out["loss_prediction"]).unsqueeze(-1)
        pass2_out = self.core(input_ids, doubt_scalars=doubt_signal)

        return {
            "logits": pass2_out["logits"],
            "doubt_signal": doubt_signal.squeeze(-1),
            "pass1_logits": pass1_out["logits"],
        }

    @staticmethod
    def _get_bound_fn(name: str):
        if name == "sigmoid":
            return nn.Sigmoid()
        elif name == "softsign":
            return nn.Softsign()
        elif name == "relu":
            return nn.ReLU()
        elif name == "tanh":
            return nn.Tanh()
        elif name == "none":
            return nn.Identity()
        else:
            raise ValueError(f"Unsupported bound function: {name}")
