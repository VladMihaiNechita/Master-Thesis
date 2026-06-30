from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.irc_vit.config import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_PREVIEW_BATCH_SIZE,
    DEFAULT_PREVIEW_GRID_NROW,
    DEFAULT_PREVIEW_IRC_BUFFER_MULTIPLIER,
    DEFAULT_PREVIEW_IRC_MIN_BUFFER,
    DEFAULT_PREVIEW_OUTPUT_DIR,
    DEFAULT_RANDOM_CONV_DEPTH,
    DEFAULT_RANDOM_CONV_WIDTH,
    DEFAULT_SEED,
)
from src.irc_vit.generators import build_generator
from src.irc_vit.utils import save_grid, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save preview grids for the image generators")
    parser.add_argument("--out", default=DEFAULT_PREVIEW_OUTPUT_DIR)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_PREVIEW_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    out = Path(args.out)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    specs = [
        ("iid_rgb_noise", {"name": "iid_rgb_noise", "params": {}}),
        ("gaussian_blur_noise", {"name": "gaussian_blur_noise", "params": {}}),
        ("fourier_texture", {"name": "fourier_texture", "params": {}}),
        ("simple_shapes", {"name": "simple_shapes", "params": {}}),
    ]
    for k in (0, 1, 2, 4, 8):
        specs.append((
            f"irc_conv_k{k}",
            {
                "name": "irc_conv",
                "params": {
                    "k": k,
                    "mode": "random_generator_per_batch",
                    "width": DEFAULT_RANDOM_CONV_WIDTH,
                    "depth": DEFAULT_RANDOM_CONV_DEPTH,
                    "buffer_size": max(DEFAULT_PREVIEW_IRC_MIN_BUFFER, args.batch_size * DEFAULT_PREVIEW_IRC_BUFFER_MULTIPLIER),
                    "seed": args.seed,
                },
            },
        ))

    for name, cfg in specs:
        gen = build_generator(cfg)
        images = gen.generate(args.batch_size, args.image_size, device, seed=args.seed).cpu()
        path = out / f"{name}.png"
        save_grid(images, path, nrow=min(DEFAULT_PREVIEW_GRID_NROW, args.batch_size), title=name)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
