import torch
import torch.nn as nn
import math

# def get_lr_multipliers(layer_size, device):
#     indices = torch.arange(layer_size, device=device)
#     # Same random value for all neurons this step, but different across steps
#     random_phase = torch.rand(1).item() * 2 * math.pi  # or hash(step) based
#     return 0.25 * torch.cos(random_phase + math.pi * (indices / layer_size)) + 0.75


# def apply_neuron_lr_scaling(model):
#     for module in model.modules():
#         if isinstance(module, nn.Linear):
#             mult = get_lr_multipliers(module.out_features, module.weight.device)
#             module.weight.grad *= mult.unsqueeze(1)  # (out, 1) broadcasts over in_features
#             if module.bias is not None:
#                 module.bias.grad *= mult

#         elif isinstance(module, nn.Conv2d):
#             mult = get_lr_multipliers(module.out_channels, module.weight.device)
#             module.weight.grad *= mult.view(-1, 1, 1, 1)  # (out, 1, 1, 1)
#             if module.bias is not None:
#                 module.bias.grad *= mult

# # Training loop
# for step, (x, y) in enumerate(dataloader):
#     optimizer.zero_grad()
#     loss = criterion(model(x), y)
#     loss.backward()

#     apply_neuron_lr_scaling(model)  # <-- insert here

#     optimizer.step()

import torch
import torch.nn as nn
import math


class CohortLRScheduler:
    def __init__(self, min_mult=0.5, max_mult=1.0, seed=42, device="cpu"):
        self.min_mult = min_mult
        self.max_mult = max_mult
        self.rng = torch.Generator(device=device).manual_seed(seed)
        self.step_count = 0
        self.current_phase = torch.tensor(0.0, device=device)
        self.permutations = {}
        self.phase = torch.tensor(2 * math.pi, device=device)

    def step(self):
        """Advance to next random phase"""
        self.current_phase = torch.rand(1, generator=self.rng, device=self.current_phase.device) * self.phase
        self.step_count += 1

    def get_multipliers(self, layer, layer_size, device):
        if layer_size not in self.permutations:
            perm = torch.randperm(layer_size, generator=self.rng, device=device)
            self.permutations[layer] = perm

        indices = self.permutations[layer].float()
        amplitude = (self.max_mult - self.min_mult) / 2
        center = (self.max_mult + self.min_mult) / 2
        return (
            amplitude * torch.cos(self.current_phase + math.pi * (indices / layer_size))
            + center
        )

    def apply_to_gradients(self, model):
        for module in model.modules():
            if isinstance(module, nn.Linear):
                if module.weight.grad is None:
                    continue
                mult = self.get_multipliers(
                    module, module.out_features, module.weight.device
                )
                module.weight.grad *= mult.unsqueeze(1)
                if module.bias is not None and module.bias.grad is not None:
                    module.bias.grad *= mult

            elif isinstance(module, nn.Conv2d):
                if module.weight.grad is None:
                    continue
                mult = self.get_multipliers(
                    module, module.out_channels, module.weight.device
                )
                module.weight.grad *= mult.view(-1, 1, 1, 1)
                if module.bias is not None and module.bias.grad is not None:
                    module.bias.grad *= mult


# Usage
# cohort_scheduler = CohortLRScheduler(min_mult=0.5, max_mult=1.0)

# for step, (x, y) in enumerate(dataloader):
#     optimizer.zero_grad()
#     loss = criterion(model(x), y)
#     loss.backward()

#     cohort_scheduler.apply_to_gradients(model)
#     cohort_scheduler.step()

#     optimizer.step()
