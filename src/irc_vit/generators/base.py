from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


class ImageGenerator(Protocol):
    name: str

    def generate(
            self,
            batch_size: int,
            image_size: int,
            device: torch.device | str,
            seed: int | None = None,
    ) -> torch.Tensor:
        """Return float images in [0, 1] with shape [B, 3, H, W]."""


@dataclass
class GeneratorConfig:
    name: str
    params: dict


def validate_images(images: torch.Tensor, batch_size: int, image_size: int) -> torch.Tensor:
    if images.shape != (batch_size, 3, image_size, image_size):
        raise ValueError(f"bad image shape {tuple(images.shape)}")
    if images.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError(f"bad image dtype {images.dtype}")
    return images.detach().float().clamp(0, 1)
