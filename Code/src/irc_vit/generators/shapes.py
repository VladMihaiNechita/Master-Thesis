from __future__ import annotations

import math

import torch

from src.irc_vit.config import DEFAULT_SIMPLE_SHAPES_MAX, DEFAULT_SIMPLE_SHAPES_MIN
from src.irc_vit.generators.base import validate_images
from src.irc_vit.utils import torch_generator


class SimpleShapes:
    name = "simple_shapes"

    def __init__(
            self,
            shape_count_min: int = DEFAULT_SIMPLE_SHAPES_MIN,
            shape_count_max: int = DEFAULT_SIMPLE_SHAPES_MAX,
            max_shapes: int | None = None,
    ):
        if max_shapes is not None:
            shape_count_max = max_shapes
        self.shape_count_min = int(shape_count_min)
        self.shape_count_max = int(shape_count_max)

    def generate(self, batch_size: int, image_size: int, device, seed: int | None = None) -> torch.Tensor:
        device = torch.device(device)
        gen = torch_generator(device, seed)
        bg = torch.rand(batch_size, 3, 1, 1, device=device, generator=gen)
        images = bg.expand(batch_size, 3, image_size, image_size).clone()
        coords = torch.linspace(0, image_size - 1, image_size, device=device)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        xx = xx.unsqueeze(0)
        yy = yy.unsqueeze(0)
        n_shapes = torch.randint(
            self.shape_count_min,
            self.shape_count_max + 1,
            (batch_size, 1, 1),
            generator=gen,
            device=device,
        )
        for shape_idx in range(self.shape_count_max):
            active = shape_idx < n_shapes
            color = torch.rand(batch_size, 3, 1, 1, device=device, generator=gen)
            kind = torch.randint(0, 5, (batch_size, 1, 1), generator=gen, device=device)
            cx = torch.empty((batch_size, 1, 1), device=device).uniform_(0.25 * image_size, 0.75 * image_size, generator=gen)
            cy = torch.empty((batch_size, 1, 1), device=device).uniform_(0.25 * image_size, 0.75 * image_size, generator=gen)
            scale = torch.empty((batch_size, 1, 1), device=device).uniform_(0.10 * image_size, 0.25 * image_size, generator=gen)
            angle = torch.empty((batch_size, 1, 1), device=device).uniform_(0, math.pi, generator=gen)

            circle = (xx - cx) ** 2 + (yy - cy) ** 2 <= scale ** 2
            square = ((xx - cx).abs() <= scale) & ((yy - cy).abs() <= scale)
            xr = (xx - cx) * angle.cos() + (yy - cy) * angle.sin()
            yr = -(xx - cx) * angle.sin() + (yy - cy) * angle.cos()
            rotated_rect = (xr.abs() <= 1.5 * scale) & (yr.abs() <= 0.45 * scale)
            triangle = (yy >= cy - scale) & (yy <= cy + scale)
            left = cx - (yy - (cy - scale)).clamp_min(0) / (2 * scale).clamp_min(1e-6) * scale
            right = cx + (yy - (cy - scale)).clamp_min(0) / (2 * scale).clamp_min(1e-6) * scale
            triangle &= (xx >= left) & (xx <= right)
            line_dist = ((xx - cx) * angle.sin() - (yy - cy) * angle.cos()).abs()
            line_along = ((xx - cx) * angle.cos() + (yy - cy) * angle.sin()).abs()
            line = (line_dist <= 1.5) & (line_along <= 1.6 * scale)

            mask = circle
            mask = torch.where(kind == 1, square, mask)
            mask = torch.where(kind == 2, rotated_rect, mask)
            mask = torch.where(kind == 3, triangle, mask)
            mask = torch.where(kind == 4, line, mask)
            mask &= active
            images = torch.where(mask.unsqueeze(1), color, images)
        return validate_images(images, batch_size, image_size)
