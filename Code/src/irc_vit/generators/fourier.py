from __future__ import annotations

import math

import torch

from src.irc_vit.config import (
    DEFAULT_FOURIER_COLOR_MIXING,
    DEFAULT_FOURIER_COMPONENTS,
    DEFAULT_FOURIER_MAX_FREQ,
    DEFAULT_FOURIER_MIN_FREQ,
)
from src.irc_vit.generators.base import validate_images
from src.irc_vit.utils import torch_generator


class FourierTexture:
    name = "fourier_texture"

    def __init__(
            self,
            components: int = DEFAULT_FOURIER_COMPONENTS,
            min_freq: float = DEFAULT_FOURIER_MIN_FREQ,
            max_freq: float = DEFAULT_FOURIER_MAX_FREQ,
            color_mixing: bool = DEFAULT_FOURIER_COLOR_MIXING,
    ):
        self.components = int(components)
        self.min_freq = float(min_freq)
        self.max_freq = float(max_freq)
        self.color_mixing = bool(color_mixing)

    def generate(self, batch_size: int, image_size: int, device, seed: int | None = None) -> torch.Tensor:
        device = torch.device(device)
        gen = torch_generator(device, seed)
        coords = torch.linspace(-1, 1, image_size, device=device)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        grid = torch.stack([xx, yy], dim=0)
        images = torch.zeros(batch_size, 3, image_size, image_size, device=device)
        for _ in range(self.components):
            freq = torch.empty(batch_size, 1, 1, 1, device=device).uniform_(
                self.min_freq, self.max_freq, generator=gen,
            )
            theta = torch.empty(batch_size, 1, 1, 1, device=device).uniform_(0, 2 * math.pi, generator=gen)
            phase = torch.empty(batch_size, 1, 1, 1, device=device).uniform_(0, 2 * math.pi, generator=gen)
            direction = torch.cat([theta.cos(), theta.sin()], dim=1)
            wave_arg = freq * (
                direction[:, :1] * grid[0].view(1, 1, image_size, image_size)
                + direction[:, 1:] * grid[1].view(1, 1, image_size, image_size)
            ) + phase
            amp = torch.randn(batch_size, 3, 1, 1, device=device, generator=gen) if self.color_mixing else 1.0
            images = images + amp * torch.sin(wave_arg)
        images = images / math.sqrt(self.components)
        minv = images.amin(dim=(1, 2, 3), keepdim=True)
        maxv = images.amax(dim=(1, 2, 3), keepdim=True)
        images = (images - minv) / (maxv - minv).clamp_min(1e-6)
        return validate_images(images, batch_size, image_size)
