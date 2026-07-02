from __future__ import annotations

import math
from functools import lru_cache
from typing import Callable

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


DEFAULT_SYNTHETIC_TRAIN_SIZE = 50_000
DEFAULT_SYNTHETIC_EVAL_SIZE = 10_000
SYNTHETIC_DATASET_VERSION = "info_txt_synthetic_v1"
SYNTHETIC_SPLIT_OFFSETS = {
    "train": 0,
    "val": 5_000_000_009,
    "validation": 5_000_000_009,
    "dev": 5_000_000_009,
    "test": 10_000_000_019,
    "eval": 10_000_000_019,
}

SHAPE_COLOR_OBJECT_CLASSES = (
    "circle",
    "square",
    "triangle",
    "ellipse",
    "star",
    "ring",
    "cross",
    "polygon",
)
LINE_FIELD_ORIENTATION_CLASSES = tuple(
    f"{orientation}_{frequency}"
    for orientation in ("horizontal", "vertical", "diagonal_45", "diagonal_135")
    for frequency in ("low", "medium", "high")
)
TEXTURE_MOSAIC_CLASSES = (
    "perlin_noise",
    "fourier_texture",
    "reaction_diffusion",
    "cellular_automata",
    "fractal_texture",
    "blurred_noise",
    "dot_texture",
    "wave_texture",
)
CHECKER_GRID_FIELD_CLASSES = tuple(
    f"{grid}_{cell}px"
    for grid in ("checkerboard", "square_grid", "hex_grid")
    for cell in (4, 8, 16, 32)
)


@lru_cache(maxsize=16)
def _coords(size: int) -> tuple[torch.Tensor, torch.Tensor]:
    values = torch.linspace(-1.0, 1.0, int(size), dtype=torch.float32)
    yy, xx = torch.meshgrid(values, values, indexing="ij")
    return xx, yy


def _generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(int(seed) & 0x7FFFFFFFFFFFFFFF)


def _rand(gen: torch.Generator, low: float = 0.0, high: float = 1.0) -> float:
    return low + (high - low) * float(torch.rand((), generator=gen).item())


def _randint(gen: torch.Generator, low: int, high: int) -> int:
    if high <= low:
        return low
    return int(torch.randint(low, high, (1,), generator=gen).item())


def _choice(gen: torch.Generator, values: tuple[int, ...] | list[int]) -> int:
    return int(values[_randint(gen, 0, len(values))])


def _rand_color(gen: torch.Generator, minimum: float = 0.08, maximum: float = 0.95) -> torch.Tensor:
    return torch.empty(3, 1, 1).uniform_(minimum, maximum, generator=gen)


def _normalize01(x: torch.Tensor) -> torch.Tensor:
    return (x - x.amin()).div((x.amax() - x.amin()).clamp_min(1e-6)).clamp(0.0, 1.0)


def _smooth_noise(size: int, cells: int, gen: torch.Generator, mode: str = "bicubic") -> torch.Tensor:
    cells = max(2, int(cells))
    noise = torch.rand(1, 1, cells, cells, generator=gen)
    out = F.interpolate(noise, size=(size, size), mode=mode, align_corners=False if mode != "nearest" else None)
    return _normalize01(out[0, 0])


def _soft_downsample(image: torch.Tensor, image_size: int) -> torch.Tensor:
    if image.shape[-1] == image_size:
        return image.clamp(0.0, 1.0)
    return F.interpolate(
        image.unsqueeze(0),
        size=(image_size, image_size),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).squeeze(0).clamp(0.0, 1.0)


def _downsample_mask(mask: torch.Tensor, image_size: int) -> torch.Tensor:
    if mask.shape[-1] == image_size:
        return mask.long()
    return F.interpolate(mask[None, None].float(), size=(image_size, image_size), mode="nearest")[0, 0].long()


def _balanced_size(requested: int, class_count: int) -> int:
    requested = int(requested)
    class_count = int(class_count)
    if requested < class_count:
        return requested
    return requested - requested % class_count


def _background(size: int, gen: torch.Generator) -> torch.Tensor:
    xx, yy = _coords(size)
    base = _rand_color(gen, 0.12, 0.75)
    tint = _rand_color(gen, 0.0, 0.35)
    angle = _rand(gen, 0.0, 2.0 * math.pi)
    grad = (xx * math.cos(angle) + yy * math.sin(angle) + 1.0) * 0.5
    waves = torch.sin((xx * _rand(gen, 1.0, 4.0) + yy * _rand(gen, 1.0, 4.0)) * math.pi + _rand(gen, 0.0, 2.0 * math.pi))
    low_noise = _smooth_noise(size, _randint(gen, 4, 10), gen)
    image = base + tint * grad.unsqueeze(0) + 0.08 * waves.unsqueeze(0) + 0.12 * (low_noise.unsqueeze(0) - 0.5)
    return image.clamp(0.0, 1.0)


def _regular_polygon_mask(x: torch.Tensor, y: torch.Tensor, sides: int) -> torch.Tensor:
    angle = torch.atan2(y, x)
    radius = torch.sqrt(x.square() + y.square())
    sector = 2.0 * math.pi / float(sides)
    local = torch.remainder(angle + math.pi, sector) - sector * 0.5
    boundary = math.cos(sector * 0.5) / torch.cos(local).clamp_min(1e-4)
    return radius <= boundary


def _shape_mask(shape_id: int, size: int, gen: torch.Generator, scale_range: tuple[float, float]) -> torch.Tensor:
    xx, yy = _coords(size)
    scale = _rand(gen, *scale_range)
    cx = _rand(gen, -0.45, 0.45)
    cy = _rand(gen, -0.45, 0.45)
    angle = _rand(gen, 0.0, 2.0 * math.pi)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    x = ((xx - cx) * cos_a + (yy - cy) * sin_a) / scale
    y = (-(xx - cx) * sin_a + (yy - cy) * cos_a) / scale
    radius = torch.sqrt(x.square() + y.square())

    if shape_id == 0:
        return radius <= 1.0
    if shape_id == 1:
        return torch.maximum(x.abs(), y.abs()) <= 1.0
    if shape_id == 2:
        return _regular_polygon_mask(x, y, 3)
    if shape_id == 3:
        return x.square() / 1.35 + y.square() / 0.55 <= 1.0
    if shape_id == 4:
        theta = torch.atan2(y, x)
        boundary = 0.62 + 0.30 * torch.cos(5.0 * theta)
        return radius <= boundary
    if shape_id == 5:
        return (radius <= 1.0) & (radius >= 0.52)
    if shape_id == 6:
        return ((x.abs() <= 0.28) & (y.abs() <= 1.0)) | ((x.abs() <= 1.0) & (y.abs() <= 0.28))
    sides = _choice(gen, [5, 6, 7])
    return _regular_polygon_mask(x, y, sides)


def _apply_mask(image: torch.Tensor, mask: torch.Tensor, color: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    mask3 = mask.unsqueeze(0).float()
    return image * (1.0 - mask3 * alpha) + color * mask3 * alpha


def _voronoi_regions(size: int, gen: torch.Generator, regions: int) -> torch.Tensor:
    xx, yy = _coords(size)
    centers = torch.empty(regions, 2).uniform_(-1.1, 1.1, generator=gen)
    dist = (xx.unsqueeze(0) - centers[:, 0, None, None]).square() + (yy.unsqueeze(0) - centers[:, 1, None, None]).square()
    return dist.argmin(dim=0)


def _dominant_region_mask(size: int, gen: torch.Generator) -> torch.Tensor:
    xx, yy = _coords(size)
    cx = _rand(gen, -0.15, 0.15)
    cy = _rand(gen, -0.15, 0.15)
    sx = _rand(gen, 0.75, 1.05)
    sy = _rand(gen, 0.55, 0.85)
    angle = _rand(gen, 0.0, math.pi)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    x = (xx - cx) * cos_a + (yy - cy) * sin_a
    y = -(xx - cx) * sin_a + (yy - cy) * cos_a
    return x.square() / (sx * sx) + y.square() / (sy * sy) <= 1.0


def _compose_regions(size: int, gen: torch.Generator, dominant_label: int, class_count: int) -> list[tuple[torch.Tensor, int]]:
    region_count = _randint(gen, 4, 8)
    labels = _voronoi_regions(size, gen, region_count)
    dominant = _dominant_region_mask(size, gen)
    other_masks = [(labels == region_id) & ~dominant for region_id in range(region_count)]
    if other_masks:
        largest_other = max(other_masks, key=lambda mask: int(mask.sum().item()))
        if int(dominant.sum().item()) <= int(largest_other.sum().item()):
            dominant = dominant | largest_other
    result = [(dominant, dominant_label)]
    distractor_labels = [class_id for class_id in range(class_count) if class_id != dominant_label]
    order = torch.randperm(len(distractor_labels), generator=gen).tolist()
    for region_id in range(region_count):
        mask = (labels == region_id) & ~dominant
        result.append((mask, distractor_labels[order[region_id % len(order)]]))
    return result


def _line_pattern(class_id: int, size: int, gen: torch.Generator) -> torch.Tensor:
    orientation = class_id // 3
    frequency = class_id % 3
    cycles = (5.0, 10.0, 18.0)[frequency]
    xx, yy = _coords(size)
    if orientation == 0:
        coord = yy
    elif orientation == 1:
        coord = xx
    elif orientation == 2:
        coord = (xx + yy) / math.sqrt(2.0)
    else:
        coord = (xx - yy) / math.sqrt(2.0)
    wobble = 0.08 * torch.sin((xx * _rand(gen, 1.0, 3.0) + yy * _rand(gen, 1.0, 3.0)) * math.pi)
    phase = _rand(gen, 0.0, 2.0 * math.pi)
    stripe = torch.cos((coord + wobble) * cycles * math.pi + phase)
    return (stripe > _rand(gen, -0.15, 0.25)).float()


def _fourier_texture(size: int, gen: torch.Generator, components: int = 6) -> torch.Tensor:
    xx, yy = _coords(size)
    out = torch.zeros(size, size)
    for _ in range(components):
        freq = _rand(gen, 1.5, 10.0)
        theta = _rand(gen, 0.0, 2.0 * math.pi)
        phase = _rand(gen, 0.0, 2.0 * math.pi)
        out += math.sin(_rand(gen, 0.0, 2.0 * math.pi)) * torch.sin(
            (xx * math.cos(theta) + yy * math.sin(theta)) * freq * math.pi + phase
        )
    return _normalize01(out)


def _dot_texture(size: int, gen: torch.Generator) -> torch.Tensor:
    xx, yy = _coords(size)
    out = torch.zeros(size, size)
    for _ in range(_randint(gen, 16, 36)):
        cx = _rand(gen, -1.0, 1.0)
        cy = _rand(gen, -1.0, 1.0)
        sigma = _rand(gen, 0.025, 0.08)
        out += torch.exp(-((xx - cx).square() + (yy - cy).square()) / (2.0 * sigma * sigma))
    return _normalize01(out)


def _texture_pattern(class_id: int, size: int, gen: torch.Generator) -> torch.Tensor:
    if class_id == 0:
        return _smooth_noise(size, 7, gen)
    if class_id == 1:
        return _fourier_texture(size, gen, components=8)
    if class_id == 2:
        a = _smooth_noise(size, 12, gen)
        b = _smooth_noise(size, 24, gen)
        return _normalize01(torch.sin((a - b) * 12.0))
    if class_id == 3:
        cells = torch.rand(1, 1, max(4, size // 7), max(4, size // 7), generator=gen)
        cells = (cells > 0.52).float()
        out = F.interpolate(cells, size=(size, size), mode="nearest")[0, 0]
        out = F.avg_pool2d(out[None, None], kernel_size=3, stride=1, padding=1)[0, 0]
        return _normalize01(out)
    if class_id == 4:
        out = torch.zeros(size, size)
        for cells, weight in ((4, 0.45), (8, 0.28), (16, 0.18), (32, 0.09)):
            out += weight * _smooth_noise(size, cells, gen)
        return _normalize01(out)
    if class_id == 5:
        return _smooth_noise(size, 18, gen)
    if class_id == 6:
        return _dot_texture(size, gen)
    xx, yy = _coords(size)
    radius = torch.sqrt(xx.square() + yy.square())
    waves = torch.sin((xx * _rand(gen, 2.0, 6.0) + radius * _rand(gen, 5.0, 14.0)) * math.pi + _rand(gen, 0.0, 2.0 * math.pi))
    return _normalize01(waves)


def _colorize(pattern: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    a = _rand_color(gen, 0.05, 0.95)
    b = _rand_color(gen, 0.05, 0.95)
    return (a * (1.0 - pattern.unsqueeze(0)) + b * pattern.unsqueeze(0)).clamp(0.0, 1.0)


def _checker_pattern(class_id: int, size: int, image_size: int, gen: torch.Generator) -> torch.Tensor:
    grid_type = class_id // 4
    cell_px = (4, 8, 16, 32)[class_id % 4]
    render_scale = size / float(image_size)
    cell = max(2.0, cell_px * image_size / 224.0 * render_scale)
    yy_pix, xx_pix = torch.meshgrid(torch.arange(size, dtype=torch.float32), torch.arange(size, dtype=torch.float32), indexing="ij")
    xx_pix = xx_pix - size * 0.5
    yy_pix = yy_pix - size * 0.5
    angle = _rand(gen, -0.45, 0.45)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    x = xx_pix * cos_a + yy_pix * sin_a + _rand(gen, 0.0, cell)
    y = -xx_pix * sin_a + yy_pix * cos_a + _rand(gen, 0.0, cell)
    if grid_type == 0:
        return ((torch.floor(x / cell) + torch.floor(y / cell)).remainder(2.0) > 0).float()
    if grid_type == 1:
        fx = torch.remainder(x / cell, 1.0)
        fy = torch.remainder(y / cell, 1.0)
        dist = torch.minimum(torch.minimum(fx, 1.0 - fx), torch.minimum(fy, 1.0 - fy))
        return (dist < 0.08).float()
    directions = (
        (1.0, 0.0),
        (0.5, math.sqrt(3.0) / 2.0),
        (0.5, -math.sqrt(3.0) / 2.0),
    )
    line = torch.zeros(size, size, dtype=torch.bool)
    for dx, dy in directions:
        coord = x * dx + y * dy
        frac = torch.remainder(coord / cell, 1.0)
        line |= torch.minimum(frac, 1.0 - frac) < 0.07
    return line.float()


def render_shape_color_objects(label: int, image_size: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    image = _background(image_size, gen)
    segmentation = torch.zeros(image_size, image_size, dtype=torch.long)
    for _ in range(_randint(gen, 0, 4)):
        distractor_label = _randint(gen, 0, len(SHAPE_COLOR_OBJECT_CLASSES))
        mask = _shape_mask(distractor_label, image_size, gen, (0.07, 0.17))
        image = _apply_mask(image, mask, _rand_color(gen), alpha=_rand(gen, 0.45, 0.8))
        segmentation[mask] = distractor_label + 1
    mask = _shape_mask(label, image_size, gen, (0.26, 0.48))
    image = _apply_mask(image, mask, _rand_color(gen, 0.12, 0.98), alpha=0.92)
    segmentation[mask] = label + 1
    if _rand(gen) < 0.65:
        occ_label = _randint(gen, 0, len(SHAPE_COLOR_OBJECT_CLASSES))
        occ_mask = _shape_mask(occ_label, image_size, gen, (0.10, 0.22))
        image = _apply_mask(image, occ_mask & mask, _rand_color(gen, 0.05, 0.9), alpha=_rand(gen, 0.35, 0.75))
        segmentation[occ_mask & mask] = occ_label + 1
    image += 0.025 * torch.randn(image.shape, generator=gen)
    return image.clamp(0.0, 1.0), segmentation


def render_line_field_orientation(label: int, image_size: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    render_size = max(image_size, image_size * 2)
    image = torch.zeros(3, render_size, render_size)
    segmentation = torch.zeros(render_size, render_size, dtype=torch.long)
    for mask, class_id in _compose_regions(render_size, gen, label, len(LINE_FIELD_ORIENTATION_CLASSES)):
        pattern = _line_pattern(class_id, render_size, gen)
        region = _colorize(pattern, gen)
        image = torch.where(mask.unsqueeze(0), region, image)
        segmentation[mask] = class_id
    image += 0.015 * torch.randn(image.shape, generator=gen)
    return _soft_downsample(image, image_size), _downsample_mask(segmentation, image_size)


def render_texture_mosaic(label: int, image_size: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    image = torch.zeros(3, image_size, image_size)
    segmentation = torch.zeros(image_size, image_size, dtype=torch.long)
    for mask, class_id in _compose_regions(image_size, gen, label, len(TEXTURE_MOSAIC_CLASSES)):
        pattern = _texture_pattern(class_id, image_size, gen)
        region = _colorize(pattern, gen)
        image = torch.where(mask.unsqueeze(0), region, image)
        segmentation[mask] = class_id
    image += 0.012 * torch.randn(image.shape, generator=gen)
    return image.clamp(0.0, 1.0), segmentation


def render_checker_grid_field(label: int, image_size: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    render_size = max(image_size, image_size * 2)
    image = torch.zeros(3, render_size, render_size)
    segmentation = torch.zeros(render_size, render_size, dtype=torch.long)
    for mask, class_id in _compose_regions(render_size, gen, label, len(CHECKER_GRID_FIELD_CLASSES)):
        pattern = _checker_pattern(class_id, render_size, image_size, gen)
        region = _colorize(pattern, gen)
        contrast = _rand(gen, 0.65, 1.0)
        region = 0.5 + (region - 0.5) * contrast
        image = torch.where(mask.unsqueeze(0), region, image)
        segmentation[mask] = class_id
    image += 0.012 * torch.randn(image.shape, generator=gen)
    return _soft_downsample(image, image_size), _downsample_mask(segmentation, image_size)


class SyntheticVisualDataset(Dataset):
    def __init__(
            self,
            name: str,
            class_names: tuple[str, ...],
            renderer: Callable[[int, int, torch.Generator], tuple[torch.Tensor, torch.Tensor]],
            image_size: int,
            size: int,
            seed: int = 0,
            split: str = "train",
    ):
        self.name = name
        self.classes = tuple(class_names)
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}
        self.renderer = renderer
        self.image_size = int(image_size)
        self.size = int(size)
        self.seed = int(seed)
        self.split = str(split)

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def __len__(self) -> int:
        return self.size

    def label_for_index(self, index: int) -> int:
        return int(index) % self.num_classes

    def class_counts(self) -> list[int]:
        base = self.size // self.num_classes
        remainder = self.size % self.num_classes
        return [base + (1 if idx < remainder else 0) for idx in range(self.num_classes)]

    def _sample_seed(self, index: int, label: int) -> int:
        split_offset = SYNTHETIC_SPLIT_OFFSETS.get(self.split.lower(), 10_000_000_019)
        return self.seed + split_offset + int(index) * 1_000_003 + int(label) * 9_176

    def sample_with_mask(self, index: int) -> tuple[torch.Tensor, int, torch.Tensor]:
        if index < 0:
            index = self.size + index
        if index < 0 or index >= self.size:
            raise IndexError(index)
        label = self.label_for_index(index)
        image, mask = self.renderer(label, self.image_size, _generator(self._sample_seed(index, label)))
        return image.float().clamp(0.0, 1.0), label, mask.long()

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        image, label, _mask = self.sample_with_mask(index)
        return image, label


SYNTHETIC_DATASETS: dict[str, tuple[str, tuple[str, ...], Callable[[int, int, torch.Generator], tuple[torch.Tensor, torch.Tensor]]]] = {
    "shape_color_objects": ("shape_color_objects", SHAPE_COLOR_OBJECT_CLASSES, render_shape_color_objects),
    "shapecolorobjects": ("shape_color_objects", SHAPE_COLOR_OBJECT_CLASSES, render_shape_color_objects),
    "line_field_orientation": ("line_field_orientation", LINE_FIELD_ORIENTATION_CLASSES, render_line_field_orientation),
    "linefieldorientation": ("line_field_orientation", LINE_FIELD_ORIENTATION_CLASSES, render_line_field_orientation),
    "texture_mosaic": ("texture_mosaic", TEXTURE_MOSAIC_CLASSES, render_texture_mosaic),
    "texturemosaic": ("texture_mosaic", TEXTURE_MOSAIC_CLASSES, render_texture_mosaic),
    "checker_grid_field": ("checker_grid_field", CHECKER_GRID_FIELD_CLASSES, render_checker_grid_field),
    "checkergridfield": ("checker_grid_field", CHECKER_GRID_FIELD_CLASSES, render_checker_grid_field),
}


def normalized_synthetic_name(name: str) -> str:
    return str(name).lower().replace("-", "_")


def canonical_synthetic_name(name: str) -> str | None:
    key = normalized_synthetic_name(name)
    if key not in SYNTHETIC_DATASETS:
        return None
    return SYNTHETIC_DATASETS[key][0]


def dataset_filter_key(name: str) -> str:
    return canonical_synthetic_name(name) or normalized_synthetic_name(name)


def is_synthetic_dataset_name(name: str) -> bool:
    return canonical_synthetic_name(name) is not None


def synthetic_split_size(name: str, split: str, requested_size: int | None = None) -> int:
    key = normalized_synthetic_name(name)
    if key not in SYNTHETIC_DATASETS:
        raise ValueError(f"Unknown synthetic dataset: {name}")
    _canonical, classes, _renderer = SYNTHETIC_DATASETS[key]
    default_size = DEFAULT_SYNTHETIC_TRAIN_SIZE if split == "train" else DEFAULT_SYNTHETIC_EVAL_SIZE
    return _balanced_size(default_size if requested_size is None else int(requested_size), len(classes))


def build_synthetic_dataset(
        name: str,
        image_size: int,
        split: str,
        seed: int = 0,
        size: int | None = None,
) -> SyntheticVisualDataset:
    key = normalized_synthetic_name(name)
    if key not in SYNTHETIC_DATASETS:
        raise ValueError(f"Unknown synthetic dataset: {name}")
    canonical, classes, renderer = SYNTHETIC_DATASETS[key]
    balanced_size = synthetic_split_size(name, split, requested_size=size)
    return SyntheticVisualDataset(
        name=canonical,
        class_names=classes,
        renderer=renderer,
        image_size=image_size,
        size=balanced_size,
        seed=seed,
        split=split,
    )


def build_synthetic_dataset_pair(
        name: str,
        image_size: int,
        seed: int = 0,
        train_size: int | None = None,
        eval_size: int | None = None,
        train_split: str = "train",
        eval_split: str = "test",
) -> tuple[SyntheticVisualDataset, SyntheticVisualDataset, str]:
    train = build_synthetic_dataset(name, image_size, split=train_split, seed=seed, size=train_size)
    eval_ds = build_synthetic_dataset(name, image_size, split=eval_split, seed=seed, size=eval_size)
    note = f"synthetic:{train.name}; train={len(train)}; eval={len(eval_ds)}; classes={train.num_classes}"
    return train, eval_ds, note
