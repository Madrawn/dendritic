import torch
import torch.nn as nn
import math

class CohortLRScheduler:
    def __init__(self, min_mult=0.5, max_mult=1.0, sharpness=1.0, seed=42, device="cpu"):
        """
        Args:
            min_mult: Minimum multiplier
            max_mult: Maximum multiplier
            sharpness: Higher values make the "high LR" band narrower. 
                       1.0 is standard cosine. 2.0+ is peakier.
        """
        self.min_mult = min_mult
        self.max_mult = max_mult
        self.sharpness = sharpness
        self.rng = torch.Generator(device=device).manual_seed(seed)
        self.step_count = 0
        self.current_phase = torch.tensor(0.0, device=device)
        self.permutations = {}
        self.phase_range = torch.tensor(2 * math.pi, device=device)

    def step(self):
        self.current_phase = torch.rand(1, generator=self.rng, device=self.current_phase.device) * self.phase_range
        self.step_count += 1

    def get_multipliers(self, layer, layer_size, device):
        if layer not in self.permutations: # Fixed bug: checking 'layer' key instead of 'layer_size'
            perm = torch.randperm(layer_size, generator=self.rng, device=device)
            self.permutations[layer] = perm

        indices = self.permutations[layer].float()
        
        # 1. Standard cosine wave: range [-1, 1]
        # We use pi * (indices/size) to ensure we only cover one peak/slope per layer
        cos_val = torch.cos(self.current_phase + math.pi * (indices / layer_size))
        
        # 2. Map to [0, 1] range
        normalized_val = (cos_val + 1) / 2
        
        # 3. Apply sharpness (The "Power" step)
        # Raising a [0, 1] value to a power > 1 makes the peak narrower
        peaky_val = torch.pow(normalized_val, self.sharpness)
        
        # 4. Rescale to [min_mult, max_mult]
        return peaky_val * (self.max_mult - self.min_mult) + self.min_mult

    def apply_to_gradients(self, model):
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if module.weight.grad is None:
                    continue
                
                size = module.out_features if isinstance(module, nn.Linear) else module.out_channels
                mult = self.get_multipliers(module, size, module.weight.device)
                
                if isinstance(module, nn.Linear):
                    module.weight.grad *= mult.unsqueeze(1)
                else:
                    module.weight.grad *= mult.view(-1, 1, 1, 1)
                    
                if module.bias is not None and module.bias.grad is not None:
                    module.bias.grad *= mult