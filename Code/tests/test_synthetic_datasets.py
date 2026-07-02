from __future__ import annotations

import torch

from src.irc_vit.config import load_config
from src.irc_vit.data import DatasetSpec, PreprocessedTensorDataset, build_dataset_pair
from src.irc_vit.evaluate import wandb_dataset_role, wandb_eval_dataset_key
from src.irc_vit.preprocess_data import main as preprocess_main, preprocess_split
from src.irc_vit.submit_slurm_pipeline import dataset_names
from src.irc_vit.synthetic_datasets import (
    DEFAULT_SYNTHETIC_EVAL_SIZE,
    DEFAULT_SYNTHETIC_TRAIN_SIZE,
    SYNTHETIC_DATASETS,
    build_synthetic_dataset,
)


CANONICAL_SYNTHETIC_NAMES = (
    "shape_color_objects",
    "line_field_orientation",
    "texture_mosaic",
    "checker_grid_field",
)


def test_synthetic_dataset_defaults_have_requested_lengths_and_even_class_counts() -> None:
    for name in CANONICAL_SYNTHETIC_NAMES:
        train, eval_ds, note = build_dataset_pair(DatasetSpec(name=name, image_size=32), seed=123)
        class_count = len(getattr(train, "classes"))
        expected_train = DEFAULT_SYNTHETIC_TRAIN_SIZE - DEFAULT_SYNTHETIC_TRAIN_SIZE % class_count
        expected_eval = DEFAULT_SYNTHETIC_EVAL_SIZE - DEFAULT_SYNTHETIC_EVAL_SIZE % class_count

        assert note.startswith(f"synthetic:{name}")
        assert len(train) == expected_train
        assert len(eval_ds) == expected_eval
        assert getattr(train, "classes")
        train_counts = train.class_counts()
        eval_counts = eval_ds.class_counts()
        assert max(train_counts) == min(train_counts)
        assert max(eval_counts) == min(eval_counts)


def test_synthetic_dataset_samples_are_valid_and_deterministic() -> None:
    for name in CANONICAL_SYNTHETIC_NAMES:
        class_count = len(SYNTHETIC_DATASETS[name][1])
        train, _, _ = build_dataset_pair(
            DatasetSpec(name=name, image_size=32, subset_train=class_count * 2, subset_eval=class_count),
            seed=999,
        )

        image_a, label_a = train[class_count + 1]
        image_b, label_b = train[class_count + 1]
        eval_ds = build_synthetic_dataset(name, image_size=32, split="test", seed=999, size=class_count * 2)
        eval_image, eval_label = eval_ds[class_count + 1]
        image_c, label_c, mask = train.sample_with_mask(class_count + 1)

        assert image_a.shape == (3, 32, 32)
        assert image_a.dtype == torch.float32
        assert 0.0 <= float(image_a.min())
        assert float(image_a.max()) <= 1.0
        assert float(image_a.std()) > 0.01
        assert label_a == label_b == 1
        assert torch.equal(image_a, image_b)
        assert label_c == label_a
        assert torch.equal(image_c, image_a)
        assert mask.shape == (32, 32)
        assert mask.dtype == torch.long
        assert int(mask.max()) >= label_a
        assert eval_label == label_a
        assert not torch.equal(eval_image, image_a)

        if name != "shape_color_objects":
            counts = torch.bincount(mask.flatten(), minlength=class_count)
            assert int(counts.argmax().item()) == label_a


def test_synthetic_validation_and_test_splits_are_distinct() -> None:
    train = build_synthetic_dataset("texture_mosaic", image_size=32, split="train", seed=123, size=8)
    val = build_synthetic_dataset("texture_mosaic", image_size=32, split="val", seed=123, size=8)
    test = build_synthetic_dataset("texture_mosaic", image_size=32, split="test", seed=123, size=8)

    train_image, train_label = train[3]
    val_image, val_label = val[3]
    test_image, test_label = test[3]

    assert train_label == val_label == test_label
    assert not torch.equal(train_image, val_image)
    assert not torch.equal(val_image, test_image)
    assert not torch.equal(train_image, test_image)


def test_region_synthetic_masks_keep_label_class_dominant() -> None:
    for name in ("line_field_orientation", "texture_mosaic", "checker_grid_field"):
        dataset = build_synthetic_dataset(name, image_size=32, split="test", seed=5, size=24)
        class_count = len(dataset.classes)
        for index in range(24):
            _image, label, mask = dataset.sample_with_mask(index)
            counts = torch.bincount(mask.flatten(), minlength=class_count)
            assert int(counts.argmax().item()) == label


def test_synthetic_dataset_aliases_resolve_to_canonical_names() -> None:
    aliases = {
        "ShapeColorObjects": "shape_color_objects",
        "LineFieldOrientation": "line_field_orientation",
        "TextureMosaic": "texture_mosaic",
        "CheckerGridField": "checker_grid_field",
    }
    for alias, canonical in aliases.items():
        dataset = build_synthetic_dataset(alias, image_size=24, split="test", seed=1, size=4)
        assert dataset.name == canonical
        assert len(dataset) == 4


def test_slurm_dataset_filter_accepts_synthetic_aliases() -> None:
    config = {"datasets": [{"name": "shape_color_objects"}, {"name": "cifar10"}]}

    assert dataset_names(config, ["ShapeColorObjects"]) == ["shape_color_objects"]


def test_eval_all_config_includes_info_txt_synthetic_validation_datasets() -> None:
    cfg = load_config("configs/eval_all_datasets.yaml")
    names = {dataset["name"] for dataset in cfg["datasets"]}

    assert set(CANONICAL_SYNTHETIC_NAMES).issubset(names)


def test_synthetic_datasets_use_validation_wandb_keys() -> None:
    for name in CANONICAL_SYNTHETIC_NAMES:
        assert wandb_dataset_role(name) == "validation"
        assert wandb_eval_dataset_key(name, "cls_mean", "linear") == f"validation/{name}_top-1_accuracy"


def test_synthetic_dataset_preprocess_cache_roundtrip(tmp_path) -> None:
    dataset = build_synthetic_dataset("shape_color_objects", image_size=24, split="train", seed=7, size=8)
    out_dir = tmp_path / "cache" / "shape_color_objects" / "train"

    preprocess_split(
        dataset,
        out_dir,
        "shape_color_objects",
        "train",
        image_size=24,
        batch_size=4,
        num_workers=0,
        shard_size=5,
        overwrite=True,
        cache_root=tmp_path / "cache",
    )
    cached = PreprocessedTensorDataset(out_dir)

    assert len(cached) == 8
    image, label = cached[0]
    assert image.shape == (3, 24, 24)
    assert 0.0 <= float(image.min())
    assert float(image.max()) <= 1.0
    assert label == 0


def test_synthetic_preprocess_cli_alias_writes_valid_cache_used_by_loader(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    cache_root = tmp_path / "cache"
    config_path.write_text(
        """
seed: 11
model:
  image_size: 24
eval:
  image_size: 24
datasets:
  - name: shape_color_objects
    subset_train: 8
    subset_eval: 8
    cache_root: "{cache_root}"
""".format(cache_root=cache_root.as_posix()),
        encoding="utf-8",
    )

    preprocess_main([
        "--config", str(config_path),
        "--out", str(cache_root),
        "--datasets", "ShapeColorObjects",
        "--batch-size", "4",
        "--num-workers", "0",
        "--shard-size", "8",
        "--overwrite",
    ])
    train, eval_ds, note = build_dataset_pair(
        DatasetSpec(
            name="shape_color_objects",
            image_size=24,
            subset_train=8,
            subset_eval=8,
            cache_root=str(cache_root),
        ),
        seed=11,
    )

    assert note.startswith("preprocessed_cache:")
    assert len(train) == 8
    assert len(eval_ds) == 8


def test_stale_synthetic_cache_without_metadata_is_ignored(tmp_path) -> None:
    cache_root = tmp_path / "cache"
    for split in ("train", "test"):
        split_root = cache_root / "shape_color_objects" / split
        split_root.mkdir(parents=True)
        torch.save({"images": torch.zeros(1, 3, 24, 24, dtype=torch.uint8), "labels": torch.zeros(1, dtype=torch.long)}, split_root / "shard_00000.pt")
        (split_root / "metadata.json").write_text(
            '{"format":"irc_vit_tensor_shards_v1","dataset":"shape_color_objects","split":"%s","image_size":24,"count":1,"dtype":"uint8","shards":[{"file":"shard_00000.pt","count":1}]}'
            % split,
            encoding="utf-8",
        )

    train, eval_ds, note = build_dataset_pair(
        DatasetSpec(name="shape_color_objects", image_size=24, subset_train=8, subset_eval=8, cache_root=str(cache_root)),
        seed=0,
    )

    assert note.startswith("synthetic:shape_color_objects")
    assert len(train) == 8
    assert len(eval_ds) == 8
