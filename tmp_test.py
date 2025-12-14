import torch
from safetensors.torch import load_file

from dendritic.enhancement import enhance_model_with_dendritic
from dendritic.layers.DendriticStack import DendriticStack
from diffusers import 

POLY_RANK = 8

path = "E:\\AI\\visual\\ComfyUI\\ComfyUI\\models\\diffusion_models\\z_image_turbo_bf16.safetensors"

# Load the state dict from the safetensors file
state_dict = load_file(path)

# Print layer names, shapes, and types
for name, tensor in state_dict.items():
    print(f"{name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")


model_dendritic = enhance_model_with_dendritic(
    model_dendritic,
    target_layers=["feed_forward", "attention"],
    poly_rank=POLY_RANK,
    freeze_linear=True,
    verbose=True,
    dendritic_cls=DendriticStack,
    dendritic_kwargs={"dropout": 0.1},
)
