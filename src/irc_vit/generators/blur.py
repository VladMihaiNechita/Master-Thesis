from __future__ import annotations

import torch
import torch.nn.functional as F

from src.irc_vit.config import DEFAULT_GAUSSIAN_BLUR_SIGMA
from src.irc_vit.generators.base import validate_images
from src.irc_vit.generators.noise import IIDRGBNoise


def gaussian_kernel1d(sigma: float, device: torch.device) -> torch.Tensor:
    radius = max(1, int(round(3 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


class GaussianBlurNoise:
    name = "gaussian_blur_noise"

    def __init__(self, sigma: float = DEFAULT_GAUSSIAN_BLUR_SIGMA):
        self.sigma = float(sigma)
        self.noise = IIDRGBNoise()

    def generate(self, batch_size: int, image_size: int, device, seed: int | None = None) -> torch.Tensor:
        device = torch.device(device)
        images = self.noise.generate(batch_size, image_size, device, seed)
        kernel = gaussian_kernel1d(self.sigma, device)
        pad = kernel.numel() // 2
        kx = kernel.view(1, 1, 1, -1).repeat(3, 1, 1, 1)
        ky = kernel.view(1, 1, -1, 1).repeat(3, 1, 1, 1)
        images = F.conv2d(F.pad(images, (pad, pad, 0, 0), mode="reflect"), kx, groups=3)
        images = F.conv2d(F.pad(images, (0, 0, pad, pad), mode="reflect"), ky, groups=3)
        return validate_images(images, batch_size, image_size)
