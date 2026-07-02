from __future__ import annotations

import importlib.util
import json

import numpy as np
import pytest
import torch
from PIL import Image

from src.irc_vit.data import (
    CACHE_FORMAT_VERSION,
    DEFAULT_VIDEO_PATH,
    DatasetSpec,
    RealImageBatchSource,
    build_dataset_pair,
    build_eval_transform,
    default_transform_policy,
)


def test_full_resize_transform_outputs_target_size() -> None:
    image = Image.new("RGB", (32, 32))
    tensor = build_eval_transform(224, "full_resize")(image)
    assert tensor.shape == (3, 224, 224)


def test_center_crop_transform_outputs_target_size() -> None:
    image = Image.new("RGB", (640, 480))
    tensor = build_eval_transform(224, "center_crop", resize_shorter_size=256)(image)
    assert tensor.shape == (3, 224, 224)


def test_default_transform_policy_matches_dataset_protocols() -> None:
    assert default_transform_policy("cifar100") == "full_resize"
    assert default_transform_policy("tiny-imagenet") == "full_resize"
    assert default_transform_policy("eurosat") == "full_resize"
    assert default_transform_policy("dtd") == "center_crop"
    assert default_transform_policy("imagenet-1k") == "center_crop"
    assert default_transform_policy("imagenet_r") == "center_crop"


def test_configured_imagefolder_dataset_path_is_used(tmp_path) -> None:
    for split in ["train", "test"]:
        class_dir = tmp_path / "cars" / split / "class_a"
        class_dir.mkdir(parents=True)
        Image.new("RGB", (32, 32)).save(class_dir / "sample.jpg")

    train, eval_ds, note = build_dataset_pair(
        DatasetSpec(
            name="cars",
            path=str(tmp_path / "cars"),
            train_split="train",
            eval_split="test",
            image_size=224,
        )
    )

    assert note == "ok"
    assert len(train) == 1
    assert len(eval_ds) == 1
    assert train[0][0].shape == (3, 224, 224)


def test_imagefolder_validation_split_uses_train_directory(tmp_path) -> None:
    for idx in range(5):
        class_dir = tmp_path / "cars" / "train" / "class_a"
        class_dir.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 32), color=(idx, idx, idx)).save(class_dir / f"train_{idx}.jpg")
    test_dir = tmp_path / "cars" / "test" / "class_a"
    test_dir.mkdir(parents=True)
    Image.new("RGB", (32, 32), color=(255, 255, 255)).save(test_dir / "test.jpg")

    train, eval_ds, note = build_dataset_pair(
        DatasetSpec(
            name="cars",
            path=str(tmp_path / "cars"),
            train_split="train",
            eval_split="val",
            eval_fraction=0.4,
            image_size=32,
        ),
        seed=7,
    )

    assert note == "ok"
    assert len(train) == 3
    assert len(eval_ds) == 2


def write_cache_split(root, images, labels) -> None:
    root.mkdir(parents=True)
    torch.save({"images": images, "labels": labels}, root / "shard_00000.pt")
    metadata = {
        "format": CACHE_FORMAT_VERSION,
        "dataset": "cifar10",
        "split": root.name,
        "image_size": 32,
        "count": int(images.shape[0]),
        "dtype": "uint8",
        "shards": [{"file": "shard_00000.pt", "count": int(images.shape[0])}],
    }
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_dataset_pair_uses_preprocessed_cache_when_available(tmp_path) -> None:
    cache_root = tmp_path / "cache"
    train_images = torch.full((2, 3, 32, 32), 127, dtype=torch.uint8)
    test_images = torch.full((1, 3, 32, 32), 255, dtype=torch.uint8)
    write_cache_split(cache_root / "cifar10" / "train", train_images, torch.tensor([0, 1]))
    write_cache_split(cache_root / "cifar10" / "test", test_images, torch.tensor([1]))

    train, eval_ds, note = build_dataset_pair(
        DatasetSpec(name="cifar10", image_size=32, cache_root=str(cache_root))
    )

    assert note.startswith("preprocessed_cache:")
    assert len(train) == 2
    assert len(eval_ds) == 1
    assert train[0][0].shape == (3, 32, 32)
    assert eval_ds[0][0].max() == 1.0


def test_validation_split_uses_preprocessed_train_cache(tmp_path) -> None:
    cache_root = tmp_path / "cache"
    train_images = torch.full((5, 3, 32, 32), 127, dtype=torch.uint8)
    test_images = torch.full((1, 3, 32, 32), 255, dtype=torch.uint8)
    write_cache_split(cache_root / "cifar10" / "train", train_images, torch.arange(5))
    write_cache_split(cache_root / "cifar10" / "test", test_images, torch.tensor([9]))

    train, eval_ds, note = build_dataset_pair(
        DatasetSpec(name="cifar10", image_size=32, cache_root=str(cache_root), eval_split="val", eval_fraction=0.4),
        seed=7,
    )

    assert note.startswith("preprocessed_cache_dev_split:")
    assert len(train) == 3
    assert len(eval_ds) == 2
    assert eval_ds[0][0].max() < 1.0


def test_real_source_defaults_to_video_path() -> None:
    assert DEFAULT_VIDEO_PATH.endswith(".mp4")


def test_real_source_rejects_cifar_pretraining() -> None:
    with pytest.raises(ValueError, match="reserved for evaluation"):
        RealImageBatchSource({"name": "cifar10"}, image_size=32, batch_size=2, seed=0)


def test_video_source_returns_augmented_batches(tmp_path) -> None:
    if importlib.util.find_spec("cv2") is None:
        pytest.skip("OpenCV is not installed")

    import cv2

    video_path = tmp_path / "toy_video.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (64, 48),
    )
    for idx in range(24):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[:, :, 0] = (idx * 7) % 255
        frame[:, :, 1] = np.arange(64, dtype=np.uint8)[None, :]
        frame[:, :, 2] = np.arange(48, dtype=np.uint8)[:, None]
        writer.write(frame)
    writer.release()

    source = RealImageBatchSource(
        {
            "name": "video",
            "path": str(video_path),
            "buffer_size": 8,
            "refresh_fraction": 0.5,
            "refresh_every": 1,
            "buffer_shorter_size": 40,
            "sample_fps": 5.0,
            "crop_scale": [0.8, 1.0],
            "color_jitter_strength": 0.0,
            "random_grayscale_prob": 0.0,
        },
        image_size=32,
        batch_size=4,
        seed=0,
    )
    batch = source.next_batch(torch.device("cpu"))

    assert batch.shape == (4, 3, 32, 32)
    assert batch.dtype == torch.float32
    assert 0.0 <= float(batch.min())
    assert float(batch.max()) <= 1.0
