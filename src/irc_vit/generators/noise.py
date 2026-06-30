from __future__ import annotations

import torch

from src.irc_vit.config import DEFAULT_IID_RGB_NOISE_DISTRIBUTION
from src.irc_vit.generators.base import validate_images
from src.irc_vit.utils import torch_generator


class IIDRGBNoise:
    name = "iid_rgb_noise"

    def __init__(self, distribution: str = DEFAULT_IID_RGB_NOISE_DISTRIBUTION):
        self.distribution = distribution

    def generate(self, batch_size: int, image_size: int, device, seed: int | None = None) -> torch.Tensor:
        device = torch.device(device)
        gen = torch_generator(device, seed)
        if self.distribution == "gaussian":
            images = torch.randn(batch_size, 3, image_size, image_size, device=device, generator=gen)
            images = images.mul(0.2).add(0.5)
        else:
            images = torch.rand(batch_size, 3, image_size, image_size, device=device, generator=gen)
        return validate_images(images.clamp(0, 1), batch_size, image_size)
