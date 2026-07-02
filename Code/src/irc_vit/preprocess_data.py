from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.irc_vit.config import (
    DEFAULT_EVAL_BATCH_SIZE,
    DEFAULT_EVAL_NUM_WORKERS,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_PREPROCESS_OUTPUT_DIR,
    DEFAULT_PREPROCESS_SHARD_SIZE,
    load_config,
)
from src.irc_vit.data import CACHE_FORMAT_VERSION, DatasetSpec, build_dataset_pair, cache_dataset_name
from src.irc_vit.synthetic_datasets import SYNTHETIC_DATASET_VERSION, dataset_filter_key


def _argv_without_sbatch_go(argv: list[str] | None) -> list[str] | None:
    if argv and argv[0] == "go":
        return argv[1:]
    return argv


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess evaluation datasets into tensor shards")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--out", default=DEFAULT_PREPROCESS_OUTPUT_DIR)
    parser.add_argument("--datasets", nargs="*", default=None, help="Dataset names to preprocess; default: all in config")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_EVAL_BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_EVAL_NUM_WORKERS)
    parser.add_argument("--shard-size", type=int, default=DEFAULT_PREPROCESS_SHARD_SIZE)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(_argv_without_sbatch_go(argv))


def dataset_spec(raw: dict[str, Any], image_size: int) -> DatasetSpec:
    values = dict(raw)
    values.setdefault("image_size", image_size)
    values["use_cache"] = False
    return DatasetSpec(**values)


def _safe_remove_dir(path: Path, allowed_root: Path) -> None:
    resolved = path.resolve()
    allowed = allowed_root.resolve()
    if not str(resolved).startswith(str(allowed)):
        raise RuntimeError(f"Refusing to remove path outside cache root: {resolved}")
    shutil.rmtree(resolved)


def _write_shard(out_dir: Path, shard_index: int, images: list[torch.Tensor], labels: list[torch.Tensor]) -> dict[str, int | str]:
    image_tensor = torch.cat(images, dim=0).contiguous()
    label_tensor = torch.cat(labels, dim=0).long().contiguous()
    file_name = f"shard_{shard_index:05d}.pt"
    torch.save({"images": image_tensor, "labels": label_tensor}, out_dir / file_name)
    return {"file": file_name, "count": int(image_tensor.shape[0])}


def preprocess_split(
        dataset: Dataset,
        out_dir: Path,
        dataset_name: str,
        split: str,
        image_size: int,
        batch_size: int,
        num_workers: int,
        shard_size: int,
        overwrite: bool,
        cache_root: Path,
) -> None:
    if out_dir.exists():
        if not overwrite:
            print(f"skip existing {out_dir}")
            return
        _safe_remove_dir(out_dir, cache_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    shards = []
    image_buffer: list[torch.Tensor] = []
    label_buffer: list[torch.Tensor] = []
    buffered = 0
    total = 0

    for images, labels in tqdm(loader, desc=f"{dataset_name}/{split}", leave=False):
        images_u8 = images.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).cpu()
        labels = labels.long().cpu()
        image_buffer.append(images_u8)
        label_buffer.append(labels)
        buffered += int(images_u8.shape[0])
        total += int(images_u8.shape[0])
        if buffered >= shard_size:
            shards.append(_write_shard(out_dir, len(shards), image_buffer, label_buffer))
            image_buffer = []
            label_buffer = []
            buffered = 0

    if buffered:
        shards.append(_write_shard(out_dir, len(shards), image_buffer, label_buffer))

    metadata = {
        "format": CACHE_FORMAT_VERSION,
        "dataset": dataset_name,
        "split": split,
        "image_size": image_size,
        "count": total,
        "dtype": "uint8",
        "shards": shards,
    }
    classes = getattr(dataset, "classes", None)
    if classes is not None:
        metadata["synthetic"] = {
            "version": SYNTHETIC_DATASET_VERSION,
            "name": getattr(dataset, "name", dataset_name),
            "class_count": len(classes),
            "classes": list(classes),
        }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"wrote {dataset_name}/{split}: {total} images -> {out_dir}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.config)
    eval_cfg = cfg.get("eval", {})
    image_size = int(eval_cfg.get("image_size", cfg.get("model", {}).get("image_size", DEFAULT_IMAGE_SIZE)))
    seed = int(cfg.get("seed", 0))
    requested = {dataset_filter_key(name) for name in args.datasets} if args.datasets else None
    cache_root = Path(args.out).expanduser()
    cache_root.mkdir(parents=True, exist_ok=True)

    for raw_dataset in cfg.get("datasets", []):
        name = str(raw_dataset["name"])
        if requested is not None and dataset_filter_key(name) not in requested:
            continue
        spec = dataset_spec(raw_dataset, image_size)
        train_ds, eval_ds, note = build_dataset_pair(spec, seed=seed)
        print(f"{name}: {note}")
        cache_name = cache_dataset_name(spec.cache_name or spec.name)
        dataset_root = cache_root / cache_name
        preprocess_split(
            train_ds,
            dataset_root / spec.train_split,
            name,
            spec.train_split,
            image_size,
            args.batch_size,
            args.num_workers,
            args.shard_size,
            args.overwrite,
            cache_root,
        )
        preprocess_split(
            eval_ds,
            dataset_root / spec.eval_split,
            name,
            spec.eval_split,
            image_size,
            args.batch_size,
            args.num_workers,
            args.shard_size,
            args.overwrite,
            cache_root,
        )


if __name__ == "__main__":
    main()
