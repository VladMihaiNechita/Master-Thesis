from __future__ import annotations

import csv
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image, ImageDraw


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def torch_generator(device: torch.device | str, seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None
    gen = torch.Generator(device=str(device))
    gen.manual_seed(int(seed))
    return gen


def init_distributed_if_needed() -> tuple[int, int, int]:
    if "RANK" not in os.environ:
        return 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    return rank, world_size, local_rank


def is_main_process() -> bool:
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def append_csv(path: str | Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in fieldnames})


def make_grid(images: torch.Tensor, nrow: int = 8, padding: int = 2) -> Image.Image:
    images = images.detach().float().cpu().clamp(0, 1)
    if images.ndim != 4:
        raise ValueError("images must have shape [B, C, H, W]")
    b, c, h, w = images.shape
    if c == 1:
        images = images.repeat(1, 3, 1, 1)
    if c != 3:
        raise ValueError("only 1 or 3 channel images are supported")
    nrow = max(1, min(nrow, b))
    ncol = (b + nrow - 1) // nrow
    canvas = Image.new("RGB", (nrow * w + (nrow - 1) * padding, ncol * h + (ncol - 1) * padding), "white")
    for i in range(b):
        y = i // nrow
        x = i % nrow
        img = (images[i].permute(1, 2, 0).numpy() * 255).round().astype("uint8")
        canvas.paste(Image.fromarray(img), (x * (w + padding), y * (h + padding)))
    return canvas


def save_grid(images: torch.Tensor, path: str | Path, nrow: int = 8, title: str | None = None) -> None:
    grid = make_grid(images, nrow=nrow)
    if title:
        header = Image.new("RGB", (grid.width, grid.height + 24), "white")
        draw = ImageDraw.Draw(header)
        draw.text((4, 4), title, fill="black")
        header.paste(grid, (0, 24))
        grid = header
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(path)


def save_labeled_grid_rows(
        rows: list[tuple[str, torch.Tensor]],
        path: str | Path,
        nrow: int = 8,
        title: str | None = None,
        label_width: int = 150,
) -> None:
    if not rows:
        raise ValueError("rows must contain at least one labeled image tensor")

    grids = [(label, make_grid(images, nrow=nrow)) for label, images in rows]
    padding = 8
    title_height = 24 if title else 0
    width = label_width + max(grid.width for _, grid in grids)
    height = title_height + sum(grid.height for _, grid in grids) + padding * (len(grids) - 1)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    y = 0
    if title:
        draw.text((4, 4), title, fill="black")
        y += title_height

    for label, grid in grids:
        draw.text((4, y + 4), label, fill="black")
        canvas.paste(grid, (label_width, y))
        y += grid.height + padding

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
