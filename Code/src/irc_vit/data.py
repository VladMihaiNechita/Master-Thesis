from __future__ import annotations

import bisect
import json
import math
import os
import re
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Any, Callable, Iterator

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from src.irc_vit.config import (
    CENTER_CROP_DATASETS,
    DEFAULT_DATA_ROOT,
    DEFAULT_DTD_PARTITION,
    DEFAULT_EVAL_FRACTION,
    DEFAULT_EVAL_NUM_WORKERS,
    DEFAULT_EVAL_SPLIT,
    DEFAULT_FAKE_DATASET_CLASSES,
    DEFAULT_FAKE_DATASET_SIZE,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_REAL_FAKE_DATASET_CLASSES,
    DEFAULT_REAL_FAKE_DATASET_SIZE,
    DEFAULT_RESIZE_SHORTER_SIZE,
    DEFAULT_TRAIN_SPLIT,
    DEFAULT_VIDEO_BUFFER_SHORTER_SIZE,
    DEFAULT_VIDEO_BUFFER_SIZE,
    DEFAULT_VIDEO_COLOR_JITTER_STRENGTH,
    DEFAULT_VIDEO_CROP_RATIO,
    DEFAULT_VIDEO_CROP_SCALE,
    DEFAULT_VIDEO_DECODE_RETRIES,
    DEFAULT_VIDEO_FALLBACK_FPS,
    DEFAULT_VIDEO_HORIZONTAL_FLIP_PROB,
    DEFAULT_VIDEO_PATH,
    DEFAULT_VIDEO_RANDOM_GRAYSCALE_PROB,
    DEFAULT_VIDEO_REFRESH_EVERY,
    DEFAULT_VIDEO_REFRESH_FRACTION,
    DEFAULT_VIDEO_RESEEK_EVERY,
    DEFAULT_VIDEO_SAMPLE_FPS,
    DEFAULT_VIDEO_TEMPORAL_JITTER,
    FULL_RESIZE_DATASETS,
    STANDARD_IMAGEFOLDER_ROOTS,
)
from src.irc_vit.synthetic_datasets import (
    SYNTHETIC_DATASET_VERSION,
    build_synthetic_dataset_pair,
    canonical_synthetic_name,
    is_synthetic_dataset_name,
    synthetic_split_size,
)

CACHE_FORMAT_VERSION = "irc_vit_tensor_shards_v1"
DEV_EVAL_SPLITS = {"val", "validation", "dev"}


@dataclass
class DatasetSpec:
    name: str
    train_split: str = DEFAULT_TRAIN_SPLIT
    eval_split: str = DEFAULT_EVAL_SPLIT
    root: str = DEFAULT_DATA_ROOT
    image_size: int = DEFAULT_IMAGE_SIZE
    subset_train: int | None = None
    subset_eval: int | None = None
    eval_fraction: float = DEFAULT_EVAL_FRACTION
    download: bool = False
    path: str | None = None
    train_path: str | None = None
    eval_path: str | None = None
    transform_policy: str | None = None
    resize_shorter_size: int = DEFAULT_RESIZE_SHORTER_SIZE
    dtd_partition: int = DEFAULT_DTD_PARTITION
    allow_fake_if_missing: bool = False
    cache_root: str | None = None
    cache_name: str | None = None
    use_cache: bool = True
    eval_batch_size: int | None = None
    eval_num_workers: int | None = None


class LabelMappedDataset(Dataset):
    def __init__(self, dataset: Dataset, label_map: dict[int, int]):
        self.dataset = dataset
        self.label_map = label_map

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        return image, self.label_map[int(label)]


class PreprocessedTensorDataset(Dataset):
    def __init__(self, root: str | Path):
        self.root = Path(root)
        metadata_path = self.root / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Preprocessed dataset metadata not found: {metadata_path}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if self.metadata.get("format") != CACHE_FORMAT_VERSION:
            raise ValueError(f"Unsupported preprocessed dataset format in {metadata_path}")
        self.shards = list(self.metadata.get("shards", []))
        if not self.shards:
            raise ValueError(f"Preprocessed dataset has no shards: {metadata_path}")
        self.counts = [int(shard["count"]) for shard in self.shards]
        self.cumulative_counts: list[int] = []
        total = 0
        for count in self.counts:
            total += count
            self.cumulative_counts.append(total)
        self._loaded_shard_index: int | None = None
        self._loaded_shard: dict[str, torch.Tensor] | None = None

    def __len__(self) -> int:
        return self.cumulative_counts[-1]

    def __getitem__(self, index: int):
        if index < 0:
            index = len(self) + index
        if index < 0 or index >= len(self):
            raise IndexError(index)
        shard_index = bisect.bisect_right(self.cumulative_counts, index)
        previous_count = 0 if shard_index == 0 else self.cumulative_counts[shard_index - 1]
        local_index = index - previous_count
        shard = self._load_shard(shard_index)
        image = shard["images"][local_index].float().div(255.0)
        label = int(shard["labels"][local_index].item())
        return image, label

    def _load_shard(self, shard_index: int) -> dict[str, torch.Tensor]:
        if self._loaded_shard_index != shard_index:
            shard_path = self.root / self.shards[shard_index]["file"]
            payload = torch.load(shard_path, map_location="cpu")
            self._loaded_shard = {"images": payload["images"], "labels": payload["labels"].long()}
            self._loaded_shard_index = shard_index
        assert self._loaded_shard is not None
        return self._loaded_shard


def cache_dataset_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name.lower()).strip("_") or "dataset"


def _expand_path(path: str | Path) -> Path:
    return Path(os.path.expandvars(str(path))).expanduser()


def _preprocessed_dataset_root(spec: DatasetSpec) -> Path | None:
    if not spec.use_cache:
        return None
    cache_root = spec.cache_root or os.environ.get("IRC_VIT_DATA_CACHE_ROOT")
    if not cache_root:
        return None
    return _expand_path(cache_root) / cache_dataset_name(spec.cache_name or spec.name)


def _is_dev_eval_split(split: str) -> bool:
    return str(split).lower() in DEV_EVAL_SPLITS


def _cached_dataset_pair_if_available(spec: DatasetSpec, seed: int) -> tuple[Dataset, Dataset, str] | None:
    root = _preprocessed_dataset_root(spec)
    if root is None:
        return None
    train_root = root / spec.train_split
    eval_root = root / spec.eval_split
    if _is_dev_eval_split(spec.eval_split) and (train_root / "metadata.json").exists():
        if is_synthetic_dataset_name(spec.name) and not _valid_synthetic_cache_split(spec, train_root, spec.train_split, spec.subset_train):
            return None
        train_pool = PreprocessedTensorDataset(train_root)
        train, eval_ds = _split_dataset(train_pool, spec.eval_fraction, seed)
        return train, eval_ds, f"preprocessed_cache_dev_split:{root}"
    if (train_root / "metadata.json").exists() and (eval_root / "metadata.json").exists():
        if is_synthetic_dataset_name(spec.name) and not _valid_synthetic_cache_pair(spec, train_root, eval_root):
            return None
        return (
            PreprocessedTensorDataset(train_root),
            PreprocessedTensorDataset(eval_root),
            f"preprocessed_cache:{root}",
        )
    return None


def _read_cache_metadata(split_root: Path) -> dict[str, Any]:
    metadata_path = split_root / "metadata.json"
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _valid_synthetic_cache_split(spec: DatasetSpec, split_root: Path, split: str, requested_size: int | None) -> bool:
    try:
        metadata = _read_cache_metadata(split_root)
    except Exception:
        return False
    synthetic = metadata.get("synthetic")
    if not isinstance(synthetic, dict):
        return False
    canonical = canonical_synthetic_name(spec.name)
    expected_count = synthetic_split_size(spec.name, split, requested_size)
    return (
        synthetic.get("version") == SYNTHETIC_DATASET_VERSION
        and synthetic.get("name") == canonical
        and int(synthetic.get("class_count", -1)) > 0
        and int(metadata.get("image_size", -1)) == int(spec.image_size)
        and int(metadata.get("count", -1)) == expected_count
    )


def _valid_synthetic_cache_pair(spec: DatasetSpec, train_root: Path, eval_root: Path) -> bool:
    return (
        _valid_synthetic_cache_split(spec, train_root, spec.train_split, spec.subset_train)
        and _valid_synthetic_cache_split(spec, eval_root, spec.eval_split, spec.subset_eval)
    )


def _rand(generator: torch.Generator) -> float:
    return float(torch.rand((), generator=generator).item())


def _uniform(generator: torch.Generator, low: float, high: float) -> float:
    return low + (high - low) * _rand(generator)


def _randint(generator: torch.Generator, low: int, high: int) -> int:
    if high <= low:
        return low
    return int(torch.randint(low, high, (1,), generator=generator).item())


def _to_rgb(x: torch.Tensor) -> torch.Tensor:
    if x.shape[0] == 1:
        return x.repeat(3, 1, 1)
    return x[:3]


def default_transform_policy(name: str) -> str:
    lname = name.lower()
    if lname in CENTER_CROP_DATASETS:
        return "center_crop"
    return "full_resize"


def build_eval_transform(
        image_size: int,
        policy: str = "full_resize",
        resize_shorter_size: int = DEFAULT_RESIZE_SHORTER_SIZE,
) -> Callable:
    interpolation = transforms.InterpolationMode.BICUBIC
    if policy == "full_resize":
        geometry = [transforms.Resize((image_size, image_size), interpolation=interpolation, antialias=True)]
    elif policy == "center_crop":
        geometry = [
            transforms.Resize(resize_shorter_size, interpolation=interpolation, antialias=True),
            transforms.CenterCrop(image_size),
        ]
    else:
        raise ValueError(f"Unknown image transform policy: {policy}")
    return transforms.Compose([*geometry, transforms.ToTensor(), transforms.Lambda(_to_rgb)])


def build_train_transform(
        image_size: int,
        policy: str = "full_resize",
        resize_shorter_size: int = DEFAULT_RESIZE_SHORTER_SIZE,
) -> Callable:
    base = list(build_eval_transform(image_size, policy, resize_shorter_size).transforms)
    base.insert(-2, transforms.RandomHorizontalFlip())
    return transforms.Compose(base)


def _subset(ds: Dataset, limit: int | None, seed: int = 0) -> Dataset:
    if limit is None or limit <= 0 or limit >= len(ds):
        return ds
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=gen)[:limit].tolist()
    return Subset(ds, idx)


def _split_dataset(ds: Dataset, eval_fraction: float, seed: int = 0) -> tuple[Dataset, Dataset]:
    n_eval = max(1, int(len(ds) * eval_fraction))
    n_train = len(ds) - n_eval
    gen = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(ds), generator=gen).tolist()
    return Subset(ds, idx[:n_train]), Subset(ds, idx[n_train:])


def _fake_dataset(size: int, image_size: int, classes: int = 10) -> Dataset:
    return datasets.FakeData(
        size=size,
        image_size=(3, image_size, image_size),
        num_classes=classes,
        transform=transforms.ToTensor(),
    )


def _imagenet_split(split: str) -> str:
    return "val" if split in {"eval", "test", "validation"} else split


def _imagefolder_pair(
        root: Path,
        train_split: str,
        eval_split: str,
        transform: Callable,
        eval_fraction: float,
        seed: int,
) -> tuple[Dataset, Dataset]:
    train_root = root / train_split
    eval_root = root / eval_split
    if train_root.exists() and _is_dev_eval_split(eval_split):
        return _split_dataset(datasets.ImageFolder(train_root, transform=transform), eval_fraction, seed)
    if train_root.exists() and eval_root.exists():
        return datasets.ImageFolder(train_root, transform=transform), datasets.ImageFolder(eval_root, transform=transform)
    full = datasets.ImageFolder(root, transform=transform)
    return _split_dataset(full, eval_fraction, seed)


def _standard_imagefolder_root(name: str, spec: DatasetSpec) -> Path | None:
    if spec.path:
        return Path(spec.path)
    root = STANDARD_IMAGEFOLDER_ROOTS.get(name.lower())
    if root is None:
        return None
    path = Path(root)
    if (path / spec.train_split).exists() and (_is_dev_eval_split(spec.eval_split) or (path / spec.eval_split).exists()):
        return path
    return None


def _imagenet_train_dataset(root: Path, transform: Callable, download: bool) -> Dataset:
    try:
        return datasets.ImageNet(root=root, split="train", transform=transform)
    except Exception:
        train_root = root / "train"
        if train_root.exists():
            return datasets.ImageFolder(train_root, transform=transform)
        return datasets.ImageFolder(root, transform=transform)


def _imagenet_dataset(root: Path, split: str, transform: Callable) -> Dataset:
    imagenet_split = _imagenet_split(split)
    try:
        return datasets.ImageNet(root=root, split=imagenet_split, transform=transform)
    except Exception:
        split_root = root / imagenet_split
        if split_root.exists():
            return datasets.ImageFolder(split_root, transform=transform)
        return datasets.ImageFolder(root, transform=transform)


def _remap_imagefolder_labels(dataset: Dataset, reference_class_to_idx: dict[str, int], name: str) -> Dataset:
    class_to_idx = getattr(dataset, "class_to_idx", None)
    if not class_to_idx:
        raise RuntimeError(f"{name} must expose class_to_idx for label remapping")
    missing = sorted(set(class_to_idx) - set(reference_class_to_idx))
    if missing:
        preview = ", ".join(missing[:5])
        raise RuntimeError(f"{name} has classes not present in ImageNet train mapping: {preview}")
    label_map = {idx: reference_class_to_idx[class_name] for class_name, idx in class_to_idx.items()}
    return LabelMappedDataset(dataset, label_map)


def _torchvision_dataset(
        name: str,
        root: Path,
        split: str,
        transform: Callable,
        download: bool,
        dtd_partition: int = 1,
) -> Dataset:
    lname = name.lower()
    if lname == "cifar10":
        return datasets.CIFAR10(root=root, train=split == "train", transform=transform, download=download)
    if lname == "cifar100":
        return datasets.CIFAR100(root=root, train=split == "train", transform=transform, download=download)
    if lname == "stl10":
        stl_split = "train" if split == "train" else "test"
        return datasets.STL10(root=root, split=stl_split, transform=transform, download=download)
    if lname == "dtd":
        dtd_split = "train" if split == "train" else "test"
        return datasets.DTD(root=root, split=dtd_split, partition=dtd_partition, transform=transform, download=download)
    if lname == "eurosat":
        if not hasattr(datasets, "EuroSAT"):
            raise RuntimeError("torchvision.datasets.EuroSAT is not available")
        return datasets.EuroSAT(root=root, transform=transform, download=download)
    if lname in {"imagenet", "imagenet1k", "imagenet-1k", "imagenet_1k"}:
        return _imagenet_dataset(root, split, transform)
    if lname == "food101":
        food_split = "train" if split == "train" else "test"
        return datasets.Food101(root=root, split=food_split, transform=transform, download=download)
    if lname in {"pets", "oxford_iiit_pets", "oxfordiiitpet"}:
        pet_split = "trainval" if split == "train" else "test"
        return datasets.OxfordIIITPet(root=root, split=pet_split, transform=transform, download=download)
    if lname in {"cars", "stanford_cars"}:
        car_split = "train" if split == "train" else "test"
        return datasets.StanfordCars(root=root, split=car_split, transform=transform, download=download)
    if lname == "sun397":
        return datasets.SUN397(root=root, transform=transform, download=download)
    raise ValueError(f"Unknown torchvision dataset: {name}")


def build_dataset_pair(spec: DatasetSpec, seed: int = 0) -> tuple[Dataset, Dataset, str]:
    name = spec.name.lower()
    policy = spec.transform_policy or default_transform_policy(name)
    transform = build_eval_transform(spec.image_size, policy, spec.resize_shorter_size)
    cached = _cached_dataset_pair_if_available(spec, seed)

    try:
        if cached is not None:
            train, eval_ds, note = cached
        elif name == "fake":
            train = _fake_dataset(spec.subset_train or DEFAULT_FAKE_DATASET_SIZE, spec.image_size, classes=DEFAULT_FAKE_DATASET_CLASSES)
            eval_ds = _fake_dataset(spec.subset_eval or DEFAULT_FAKE_DATASET_SIZE, spec.image_size, classes=DEFAULT_FAKE_DATASET_CLASSES)
        elif is_synthetic_dataset_name(name):
            return build_synthetic_dataset_pair(
                spec.name,
                image_size=spec.image_size,
                seed=seed,
                train_size=spec.subset_train,
                eval_size=spec.subset_eval,
                train_split=spec.train_split,
                eval_split=spec.eval_split,
            )
        elif (standard_root := _standard_imagefolder_root(name, spec)) is not None:
            train, eval_ds = _imagefolder_pair(standard_root, spec.train_split, spec.eval_split, transform, spec.eval_fraction, seed)
        elif name in {"tiny_imagenet", "tiny-imagenet", "tinyimagenet"}:
            root = Path(spec.path or "data/tiny-imagenet-200")
            train, eval_ds = _imagefolder_pair(root, spec.train_split, spec.eval_split, transform, spec.eval_fraction, seed)
        elif name in {"resisc45", "resisc_45", "cars", "stanford_cars", "sun397"} and spec.path:
            if not spec.path:
                raise RuntimeError(f"{spec.name} requires an ImageFolder path")
            train, eval_ds = _imagefolder_pair(Path(spec.path), spec.train_split, spec.eval_split, transform, spec.eval_fraction, seed)
        elif name in {"imagenet_r", "imagenet-r"}:
            eval_root = Path(spec.eval_path or spec.path or "data/imagenet-r")
            if (eval_root / spec.eval_split).exists():
                eval_root = eval_root / spec.eval_split
            train_root = Path(spec.train_path or spec.root or "data/imagenet")
            train = _imagenet_train_dataset(train_root, transform, spec.download)
            eval_ds = datasets.ImageFolder(eval_root, transform=transform)
            eval_ds = _remap_imagefolder_labels(eval_ds, getattr(train, "class_to_idx", {}), spec.name)
        elif name in {"eurosat", "sun397"}:
            full = _torchvision_dataset(name, Path(spec.root), spec.train_split, transform, spec.download, spec.dtd_partition)
            if _is_dev_eval_split(spec.eval_split):
                train_pool, _final_eval = _split_dataset(full, spec.eval_fraction, seed)
                train, eval_ds = _split_dataset(train_pool, spec.eval_fraction, seed + 10_003)
            else:
                train, eval_ds = _split_dataset(full, spec.eval_fraction, seed)
        elif _is_dev_eval_split(spec.eval_split):
            root = Path(spec.root)
            train_pool = _torchvision_dataset(name, root, spec.train_split, transform, spec.download, spec.dtd_partition)
            train, eval_ds = _split_dataset(train_pool, spec.eval_fraction, seed)
        else:
            root = Path(spec.root)
            train = _torchvision_dataset(name, root, spec.train_split, transform, spec.download, spec.dtd_partition)
            eval_ds = _torchvision_dataset(name, root, spec.eval_split, transform, spec.download, spec.dtd_partition)
    except Exception as exc:
        if not spec.allow_fake_if_missing:
            raise
        note = f"{spec.name}: using FakeData fallback because {type(exc).__name__}: {exc}"
        train = _fake_dataset(spec.subset_train or DEFAULT_FAKE_DATASET_SIZE, spec.image_size, classes=DEFAULT_FAKE_DATASET_CLASSES)
        eval_ds = _fake_dataset(spec.subset_eval or DEFAULT_FAKE_DATASET_SIZE, spec.image_size, classes=DEFAULT_FAKE_DATASET_CLASSES)
        return train, eval_ds, note

    train = _subset(train, spec.subset_train, seed)
    eval_ds = _subset(eval_ds, spec.subset_eval, seed + 1)
    return train, eval_ds, note if cached is not None else "ok"


class RealImageBatchSource:
    def __init__(self, config: dict[str, Any], image_size: int, batch_size: int, seed: int = 0):
        self.config = config
        self.image_size = int(image_size)
        self.batch_size = int(batch_size)
        name = str(config.get("name", "video")).lower()
        if name == "video":
            self.video_source = VideoFrameBufferBatchSource(config, image_size, batch_size, seed)
            self.loader = None
            self._iterator = None
            return

        self.video_source = None
        self.dataset = self._build_dataset()
        generator = torch.Generator().manual_seed(seed)
        self.loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=int(config.get("num_workers", DEFAULT_EVAL_NUM_WORKERS)),
            pin_memory=torch.cuda.is_available(),
            generator=generator,
        )
        self._iterator: Iterator = cycle(self.loader)

    def _build_dataset(self) -> Dataset:
        name = str(self.config.get("name", "video")).lower()
        policy = str(self.config.get("transform_policy") or default_transform_policy(name))
        resize_shorter_size = int(self.config.get("resize_shorter_size", DEFAULT_RESIZE_SHORTER_SIZE))
        transform = build_train_transform(self.image_size, policy, resize_shorter_size)

        if name in {"cifar10", "cifar100", "stl10"}:
            raise ValueError(
                f"{name} is reserved for evaluation. "
                f"For real-data pretraining, use source.params.name=video and {DEFAULT_VIDEO_PATH}."
            )
        if name == "imagefolder":
            path = self.config.get("path")
            if not path:
                raise ValueError("real source 'imagefolder' requires source.params.path")
            return datasets.ImageFolder(path, transform=transform)
        if name == "fake":
            return _fake_dataset(
                int(self.config.get("size", DEFAULT_REAL_FAKE_DATASET_SIZE)),
                self.image_size,
                classes=int(self.config.get("classes", DEFAULT_REAL_FAKE_DATASET_CLASSES)),
            )
        raise ValueError(f"Unknown real pretraining source: {name}")

    def next_batch(self, device: torch.device) -> torch.Tensor:
        if self.video_source is not None:
            return self.video_source.next_batch(device)
        batch = next(self._iterator)
        images = batch[0] if isinstance(batch, (tuple, list)) else batch
        return images.to(device, non_blocking=True).float().clamp(0, 1)


class VideoFrameBufferBatchSource:
    def __init__(self, config: dict[str, Any], image_size: int, batch_size: int, seed: int = 0):
        self.config = config
        self.path = Path(config.get("path", DEFAULT_VIDEO_PATH))
        if not self.path.exists():
            raise FileNotFoundError(
                f"Video pretraining source not found: {self.path}. "
                f"Place the video at {DEFAULT_VIDEO_PATH} or set source.params.path."
            )

        try:
            import cv2
        except Exception as exc:  # pragma: no cover - depends on environment
            raise RuntimeError("Video pretraining source requires OpenCV. Install opencv-python-headless.") from exc

        self.cv2 = cv2
        self.image_size = int(image_size)
        self.batch_size = int(batch_size)
        self.generator = torch.Generator().manual_seed(seed)
        self.buffer_size = max(self.batch_size, int(config.get("buffer_size", DEFAULT_VIDEO_BUFFER_SIZE)))
        self.refresh_fraction = float(config.get("refresh_fraction", DEFAULT_VIDEO_REFRESH_FRACTION))
        self.refresh_every = max(1, int(config.get("refresh_every", DEFAULT_VIDEO_REFRESH_EVERY)))
        self.buffer_shorter_size = max(self.image_size, int(config.get("buffer_shorter_size", DEFAULT_VIDEO_BUFFER_SHORTER_SIZE)))
        self.horizontal_flip_prob = float(config.get("horizontal_flip_prob", DEFAULT_VIDEO_HORIZONTAL_FLIP_PROB))
        self.color_jitter_strength = float(config.get("color_jitter_strength", DEFAULT_VIDEO_COLOR_JITTER_STRENGTH))
        self.random_grayscale_prob = float(config.get("random_grayscale_prob", DEFAULT_VIDEO_RANDOM_GRAYSCALE_PROB))
        self.temporal_jitter = bool(config.get("temporal_jitter", DEFAULT_VIDEO_TEMPORAL_JITTER))
        self.crop_scale = tuple(float(x) for x in config.get("crop_scale", DEFAULT_VIDEO_CROP_SCALE))
        self.crop_ratio = tuple(float(x) for x in config.get("crop_ratio", DEFAULT_VIDEO_CROP_RATIO))
        self.decode_retries = max(1, int(config.get("decode_retries", DEFAULT_VIDEO_DECODE_RETRIES)))
        self.reseek_every = max(1, int(config.get("reseek_every", DEFAULT_VIDEO_RESEEK_EVERY)))
        self._frames_since_seek = self.reseek_every
        self._batches_served = 0

        self.cap = self._open_capture()
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_fps = float(self.cap.get(cv2.CAP_PROP_FPS) or DEFAULT_VIDEO_FALLBACK_FPS)
        sample_fps = float(config.get("sample_fps", DEFAULT_VIDEO_SAMPLE_FPS))
        default_stride = max(1, int(round(self.video_fps / max(sample_fps, 1e-6))))
        self.frame_stride = max(1, int(config.get("frame_stride", default_stride)))
        self.buffer = self._decode_initial_buffer()

    def _open_capture(self):
        cap = self.cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video source: {self.path}")
        return cap

    def _random_frame_index(self) -> int:
        if self.frame_count <= 0:
            return 0
        stride_slots = max(1, self.frame_count // self.frame_stride)
        index = _randint(self.generator, 0, stride_slots) * self.frame_stride
        if self.temporal_jitter:
            index += _randint(self.generator, 0, self.frame_stride)
        return min(self.frame_count - 1, index)

    def _seek_random_position(self) -> None:
        if self.frame_count > 0:
            self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, self._random_frame_index())
        self._frames_since_seek = 0

    def _decode_frame(self) -> torch.Tensor:
        if self._frames_since_seek >= self.reseek_every:
            self._seek_random_position()

        for _ in range(self.decode_retries):
            ok, frame = self.cap.read()
            if ok and frame is not None:
                for _ in range(max(0, self.frame_stride - 1)):
                    self.cap.grab()
                self._frames_since_seek += 1
                return self._prepare_frame(frame)
            self.cap.release()
            self.cap = self._open_capture()
            self._seek_random_position()
        raise RuntimeError(f"Could not decode a frame from video source: {self.path}")

    def _prepare_frame(self, frame) -> torch.Tensor:
        frame = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
        height, width = frame.shape[:2]
        scale = self.buffer_shorter_size / float(min(height, width))
        resized_width = max(self.image_size, int(round(width * scale)))
        resized_height = max(self.image_size, int(round(height * scale)))
        frame = self.cv2.resize(frame, (resized_width, resized_height), interpolation=self.cv2.INTER_CUBIC)
        return torch.from_numpy(frame).permute(2, 0, 1).contiguous()

    def _decode_initial_buffer(self) -> torch.Tensor:
        frames = [self._decode_frame() for _ in range(self.buffer_size)]
        return torch.stack(frames, dim=0)

    def _refresh_buffer(self) -> None:
        refresh_count = max(1, int(round(self.buffer_size * self.refresh_fraction)))
        refresh_count = min(self.buffer_size, refresh_count)
        positions = torch.randperm(self.buffer_size, generator=self.generator)[:refresh_count]
        for position in positions.tolist():
            self.buffer[position] = self._decode_frame()

    def _crop_params(self, height: int, width: int) -> tuple[int, int, int, int]:
        area = height * width
        min_scale, max_scale = self.crop_scale
        min_ratio, max_ratio = self.crop_ratio
        log_min_ratio = math.log(min_ratio)
        log_max_ratio = math.log(max_ratio)
        for _ in range(10):
            target_area = area * _uniform(self.generator, min_scale, max_scale)
            aspect_ratio = math.exp(_uniform(self.generator, log_min_ratio, log_max_ratio))
            crop_width = int(round(math.sqrt(target_area * aspect_ratio)))
            crop_height = int(round(math.sqrt(target_area / aspect_ratio)))
            if 0 < crop_width <= width and 0 < crop_height <= height:
                top = _randint(self.generator, 0, height - crop_height + 1)
                left = _randint(self.generator, 0, width - crop_width + 1)
                return top, left, crop_height, crop_width

        crop = min(height, width)
        top = (height - crop) // 2
        left = (width - crop) // 2
        return top, left, crop, crop

    def _spatial_augment(self, images: torch.Tensor) -> torch.Tensor:
        augmented = []
        for image in images:
            _, height, width = image.shape
            top, left, crop_height, crop_width = self._crop_params(height, width)
            crop = image[:, top:top + crop_height, left:left + crop_width].unsqueeze(0).float() / 255.0
            resized = F.interpolate(
                crop,
                size=(self.image_size, self.image_size),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            ).squeeze(0)
            if _rand(self.generator) < self.horizontal_flip_prob:
                resized = resized.flip(-1)
            augmented.append(resized)
        return torch.stack(augmented, dim=0).clamp(0.0, 1.0)

    def _color_augment(self, images: torch.Tensor) -> torch.Tensor:
        strength = self.color_jitter_strength
        if strength <= 0:
            return images

        batch = images
        n = batch.shape[0]
        brightness = 1.0 + (torch.rand(n, 1, 1, 1, generator=self.generator) * 2.0 - 1.0) * strength
        contrast = 1.0 + (torch.rand(n, 1, 1, 1, generator=self.generator) * 2.0 - 1.0) * strength
        saturation = 1.0 + (torch.rand(n, 1, 1, 1, generator=self.generator) * 2.0 - 1.0) * strength

        batch = batch * brightness
        mean = batch.mean(dim=(2, 3), keepdim=True)
        batch = (batch - mean) * contrast + mean
        gray = (0.2989 * batch[:, 0:1] + 0.5870 * batch[:, 1:2] + 0.1140 * batch[:, 2:3])
        batch = (batch - gray) * saturation + gray

        if self.random_grayscale_prob > 0:
            mask = torch.rand(n, 1, 1, 1, generator=self.generator) < self.random_grayscale_prob
            batch = torch.where(mask, gray.expand_as(batch), batch)
        return batch.clamp(0.0, 1.0)

    def next_batch(self, device: torch.device) -> torch.Tensor:
        if self._batches_served > 0 and self._batches_served % self.refresh_every == 0:
            self._refresh_buffer()
        self._batches_served += 1

        indices = torch.randint(0, self.buffer_size, (self.batch_size,), generator=self.generator)
        batch = self.buffer[indices].clone()
        batch = self._spatial_augment(batch)
        batch = self._color_augment(batch)
        return batch.to(device, non_blocking=True)
