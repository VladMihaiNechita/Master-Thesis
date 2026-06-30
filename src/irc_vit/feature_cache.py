from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from src.irc_vit.data import cache_dataset_name


FEATURE_CACHE_FORMAT_VERSION = "irc_vit_feature_cache_v1"

FEATURE_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def feature_dtype(name: str) -> torch.dtype:
    try:
        return FEATURE_DTYPES[str(name).lower()]
    except KeyError as exc:
        supported = ", ".join(sorted(FEATURE_DTYPES))
        raise ValueError(f"Unsupported feature dtype {name!r}; choose one of: {supported}") from exc


def feature_cache_metadata_path(root: str | Path) -> Path:
    return Path(root) / "metadata.json"


def write_feature_cache_metadata(root: str | Path, metadata: dict[str, Any]) -> None:
    path = feature_cache_metadata_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(metadata)
    payload["format"] = FEATURE_CACHE_FORMAT_VERSION
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_feature_cache_metadata(root: str | Path) -> dict[str, Any]:
    path = feature_cache_metadata_path(root)
    if not path.exists():
        raise FileNotFoundError(f"Feature cache metadata not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != FEATURE_CACHE_FORMAT_VERSION:
        raise ValueError(f"Unsupported feature cache format in {path}")
    return payload


def feature_dataset_key(raw_dataset: dict[str, Any], split_role: str) -> str:
    name = str(raw_dataset.get("name", "")).lower()
    if name in {"imagenet_r", "imagenet-r"} and split_role == "train":
        return cache_dataset_name(str(raw_dataset.get("train_cache_name", "imagenet1k")))
    return cache_dataset_name(str(raw_dataset.get("cache_name") or raw_dataset.get("name")))


def feature_split_name(raw_dataset: dict[str, Any], split_role: str) -> str:
    if split_role == "train":
        return cache_dataset_name(str(raw_dataset.get("train_split", "train")))
    if split_role == "eval":
        return cache_dataset_name(str(raw_dataset.get("eval_split", "test")))
    raise ValueError(f"Unknown split role: {split_role}")


def feature_split_path(
        root: str | Path,
        raw_dataset: dict[str, Any],
        split_role: str,
        embedding_mode: str,
) -> Path:
    return (
        Path(root)
        / cache_dataset_name(str(embedding_mode))
        / feature_dataset_key(raw_dataset, split_role)
        / f"{feature_split_name(raw_dataset, split_role)}.pt"
    )


def save_feature_split(
        path: str | Path,
        features: torch.Tensor,
        labels: torch.Tensor,
        metadata: dict[str, Any],
        dtype: torch.dtype = torch.float32,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": FEATURE_CACHE_FORMAT_VERSION,
        "features": features.detach().cpu().to(dtype),
        "labels": labels.detach().cpu().long(),
        "metadata": dict(metadata),
    }
    torch.save(payload, path)


def load_feature_split(path: str | Path) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    path = Path(path)
    payload = torch.load(path, map_location="cpu")
    if payload.get("format") != FEATURE_CACHE_FORMAT_VERSION:
        raise ValueError(f"Unsupported feature cache split format in {path}")
    features = payload["features"].float()
    labels = payload["labels"].long()
    metadata = dict(payload.get("metadata", {}))
    return features, labels, metadata
