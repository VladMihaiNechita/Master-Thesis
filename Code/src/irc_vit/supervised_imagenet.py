from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from src.irc_vit.config import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_TRAIN_BATCH_SIZE,
    DEFAULT_TRAIN_BETAS,
    DEFAULT_TRAIN_CLIP_GRAD,
    DEFAULT_TRAIN_LOG_EVERY,
    DEFAULT_TRAIN_LR,
    DEFAULT_TRAIN_STEPS,
    DEFAULT_TRAIN_WEIGHT_DECAY,
    load_config,
    save_config,
)
from src.irc_vit.data import (
    CACHE_FORMAT_VERSION,
    DatasetSpec,
    build_dataset_pair,
    cache_dataset_name,
)
from src.irc_vit.evaluate import LINEAR_PROBE_HEAD_FORMAT, LINEAR_PROBE_HEAD_VERSION
from src.irc_vit.model import build_encoder
from src.irc_vit.train import (
    apply_target_instances,
    images_seen_for_step,
    init_wandb_if_enabled,
    learning_rate,
    load_checkpoint,
    save_checkpoint,
    set_optimizer_lr,
    wandb_log,
)
from src.irc_vit.utils import append_csv, append_jsonl, ensure_dir, git_commit_hash, seed_everything


EVAL_COLUMNS = [
    "run_id",
    "step",
    "images_seen",
    "dataset",
    "split",
    "num_samples",
    "loss",
    "top1_accuracy",
    "top5_accuracy",
    "checkpoint",
    "notes",
]


class SupervisedViTClassifier(nn.Module):
    _EMBEDDING_DIM_MULTIPLIERS = {
        "cls": 1,
        "mean": 1,
        "cls_mean": 2,
    }

    def __init__(self, model_config: dict[str, Any], num_classes: int, embedding_mode: str | None = None):
        super().__init__()
        self.encoder = build_encoder(model_config)
        self.embedding_mode = str(
            embedding_mode
            or model_config.get("classifier_embedding_mode")
            or model_config.get("embedding_mode")
            or "cls"
        )
        if self.embedding_mode not in self._EMBEDDING_DIM_MULTIPLIERS:
            supported = ", ".join(sorted(self._EMBEDDING_DIM_MULTIPLIERS))
            raise ValueError(f"Unsupported supervised classifier embedding mode: {self.embedding_mode}. Supported: {supported}")
        head_dim = self.encoder.config.embed_dim * self._EMBEDDING_DIM_MULTIPLIERS[self.embedding_mode]
        self.head = nn.Linear(head_dim, int(num_classes))
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder.encode(images, mode=self.embedding_mode))


class CachedTensorBatchStream:
    """Multi-shard shuffled batches from a preprocessed tensor cache."""

    def __init__(
            self,
            root: str | Path,
            batch_size: int,
            seed: int,
            horizontal_flip_prob: float = 0.0,
            shuffle_shards: bool = True,
            shuffle_within_shard: bool = True,
            shuffle_buffer_shards: int = 1,
    ):
        self.root = Path(root)
        metadata_path = self.root / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Preprocessed cache metadata not found: {metadata_path}")
        metadata = torch_load_json(metadata_path)
        if metadata.get("format") != CACHE_FORMAT_VERSION:
            raise ValueError(f"Unsupported preprocessed cache format in {metadata_path}")
        self.shards = list(metadata.get("shards", []))
        if not self.shards:
            raise ValueError(f"Preprocessed cache has no shards: {metadata_path}")
        self.count = int(metadata.get("count", sum(int(shard["count"]) for shard in self.shards)))
        self.batch_size = int(batch_size)
        self.horizontal_flip_prob = float(horizontal_flip_prob)
        self.shuffle_shards = bool(shuffle_shards)
        self.shuffle_within_shard = bool(shuffle_within_shard)
        self.shuffle_buffer_shards = max(1, int(shuffle_buffer_shards))
        self.generator = torch.Generator().manual_seed(int(seed))
        self._epoch = 0
        self._shard_order: list[int] = []
        self._shard_cursor = 0
        self._buffer_shard_indices: list[int] = []
        self._images: torch.Tensor | None = None
        self._labels: torch.Tensor | None = None
        self._sample_order: torch.Tensor | None = None
        self._sample_cursor = 0
        self._begin_epoch()

    def _begin_epoch(self) -> None:
        if self.shuffle_shards:
            self._shard_order = torch.randperm(len(self.shards), generator=self.generator).tolist()
        else:
            self._shard_order = list(range(len(self.shards)))
        self._shard_cursor = 0
        self._epoch += 1

    def _read_shards(self, shard_indices: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        image_parts = []
        label_parts = []
        for shard_index in shard_indices:
            payload = torch.load(self.root / self.shards[shard_index]["file"], map_location="cpu")
            image_parts.append(payload["images"])
            label_parts.append(payload["labels"].long())
        return torch.cat(image_parts, dim=0), torch.cat(label_parts, dim=0)

    def _load_buffer_indices(self, shard_indices: list[int]) -> None:
        self._buffer_shard_indices = list(shard_indices)
        self._images, self._labels = self._read_shards(self._buffer_shard_indices)
        if self.shuffle_within_shard:
            self._sample_order = torch.randperm(int(self._labels.shape[0]), generator=self.generator)
        else:
            self._sample_order = torch.arange(int(self._labels.shape[0]))
        self._sample_cursor = 0

    def _load_next_buffer(self) -> None:
        shard_indices = []
        while len(shard_indices) < self.shuffle_buffer_shards:
            if self._shard_cursor >= len(self._shard_order):
                self._begin_epoch()
            shard_indices.append(self._shard_order[self._shard_cursor])
            self._shard_cursor += 1
        self._load_buffer_indices(shard_indices)

    def next_batch(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        image_parts: list[torch.Tensor] = []
        label_parts: list[torch.Tensor] = []
        remaining = self.batch_size
        while remaining > 0:
            if self._images is None or self._sample_order is None or self._sample_cursor >= self._sample_order.numel():
                self._load_next_buffer()
            assert self._images is not None
            assert self._labels is not None
            assert self._sample_order is not None
            take = min(remaining, self._sample_order.numel() - self._sample_cursor)
            indices = self._sample_order[self._sample_cursor:self._sample_cursor + take]
            self._sample_cursor += take
            image_parts.append(self._images.index_select(0, indices))
            label_parts.append(self._labels.index_select(0, indices))
            remaining -= take

        images = torch.cat(image_parts, dim=0).float().div_(255.0)
        labels = torch.cat(label_parts, dim=0).long()
        if self.horizontal_flip_prob > 0:
            flip = torch.rand(images.shape[0], generator=self.generator) < self.horizontal_flip_prob
            if bool(flip.any()):
                images[flip] = images[flip].flip(-1)
        return images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

    def state_dict(self) -> dict[str, Any]:
        return {
            "generator_state": self.generator.get_state(),
            "epoch": self._epoch,
            "shard_order": list(self._shard_order),
            "shard_cursor": self._shard_cursor,
            "buffer_shard_indices": list(self._buffer_shard_indices),
            "sample_order": self._sample_order.clone() if self._sample_order is not None else None,
            "sample_cursor": self._sample_cursor,
        }

    def load_state_dict(self, state: dict[str, Any], device: torch.device | str | None = None) -> None:
        if state.get("generator_state") is not None:
            self.generator.set_state(state["generator_state"])
        self._epoch = int(state.get("epoch", self._epoch))
        self._shard_order = [int(x) for x in state.get("shard_order", self._shard_order)]
        self._shard_cursor = int(state.get("shard_cursor", self._shard_cursor))
        buffer_indices = [int(x) for x in state.get("buffer_shard_indices", [])]
        if buffer_indices:
            self._buffer_shard_indices = buffer_indices
            self._images, self._labels = self._read_shards(buffer_indices)
            sample_order = state.get("sample_order")
            self._sample_order = sample_order.clone().long() if sample_order is not None else torch.arange(int(self._labels.shape[0]))
            self._sample_cursor = int(state.get("sample_cursor", 0))


class LoaderBatchStream:
    def __init__(self, loader: DataLoader):
        self.loader = loader
        self._iterator: Iterator | None = None

    def next_batch(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if self._iterator is None:
            self._iterator = iter(self.loader)
        try:
            images, labels = next(self._iterator)
        except StopIteration:
            self._iterator = iter(self.loader)
            images, labels = next(self._iterator)
        return images.to(device, non_blocking=True), labels.long().to(device, non_blocking=True)


def torch_load_json(path: Path) -> dict[str, Any]:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def _argv_without_sbatch_go(argv: list[str] | None) -> list[str] | None:
    if argv and argv[0] == "go":
        return argv[1:]
    return argv


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supervised ViT training on ImageNet-1K")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--disable-wandb", action="store_true")
    raw_argv = sys.argv[1:] if argv is None else argv
    return parser.parse_args(_argv_without_sbatch_go(raw_argv))


def dataset_spec(config: dict[str, Any]) -> DatasetSpec:
    raw = config.get("dataset")
    if raw is None:
        datasets = config.get("datasets", [])
        if len(datasets) != 1:
            raise ValueError("supervised ImageNet config must define one dataset or exactly one datasets entry")
        raw = datasets[0]
    values = dict(raw)
    image_size = int(config.get("eval", {}).get("image_size", config.get("model", {}).get("image_size", DEFAULT_IMAGE_SIZE)))
    values.setdefault("image_size", image_size)
    return DatasetSpec(**values)


def cache_split_root(spec: DatasetSpec, split: str) -> Path | None:
    cache_root = spec.cache_root or os.environ.get("IRC_VIT_DATA_CACHE_ROOT")
    if not cache_root:
        return None
    root = Path(os.path.expandvars(str(cache_root))).expanduser() / cache_dataset_name(spec.cache_name or spec.name) / split
    return root if (root / "metadata.json").exists() else None


def build_train_stream(config: dict[str, Any], spec: DatasetSpec, train_ds, seed: int):
    train_cfg = config.get("train", {})
    batch_size = int(train_cfg.get("batch_size", DEFAULT_TRAIN_BATCH_SIZE))
    cache_root = cache_split_root(spec, spec.train_split)
    if cache_root is not None and bool(train_cfg.get("use_cache_stream", True)):
        return CachedTensorBatchStream(
            cache_root,
            batch_size=batch_size,
            seed=seed,
            horizontal_flip_prob=float(train_cfg.get("horizontal_flip_prob", 0.5)),
            shuffle_shards=bool(train_cfg.get("shuffle_shards", True)),
            shuffle_within_shard=bool(train_cfg.get("shuffle_within_shard", True)),
            shuffle_buffer_shards=int(train_cfg.get("shuffle_buffer_shards", 1)),
        )

    loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=int(train_cfg.get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
        generator=torch.Generator().manual_seed(seed),
    )
    return LoaderBatchStream(loader)


def build_eval_loader(config: dict[str, Any], spec: DatasetSpec, eval_ds) -> DataLoader:
    eval_cfg = config.get("eval", {})
    batch_size = int(eval_cfg.get("batch_size", spec.eval_batch_size or 128))
    num_workers = int(eval_cfg.get("num_workers", spec.eval_num_workers if spec.eval_num_workers is not None else 0))
    return DataLoader(
        eval_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def load_mae_encoder_weights(model: SupervisedViTClassifier, checkpoint: str | Path, strict: bool = True) -> dict[str, Any]:
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older torch
        payload = torch.load(checkpoint, map_location="cpu")
    state = payload.get("model", payload)
    encoder_state: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.startswith("module.encoder."):
            encoder_state[key.removeprefix("module.encoder.")] = value
        elif key.startswith("encoder."):
            encoder_state[key.removeprefix("encoder.")] = value
        elif key.startswith("module."):
            stripped = key.removeprefix("module.")
            if stripped in model.encoder.state_dict():
                encoder_state[stripped] = value
        elif key in model.encoder.state_dict():
            encoder_state[key] = value
    missing, unexpected = model.encoder.load_state_dict(encoder_state, strict=False)
    if strict and (missing or unexpected):
        raise RuntimeError(
            f"Could not strictly load encoder from {checkpoint}: "
            f"missing={list(missing)}, unexpected={list(unexpected)}"
        )
    return {
        "checkpoint": str(checkpoint),
        "source_step": int(payload.get("step", 0)) if isinstance(payload, dict) else 0,
        "loaded_keys": len(encoder_state),
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }


def load_linear_probe_head(
        model: SupervisedViTClassifier,
        checkpoint: str | Path,
        strict: bool = True,
        allow_partial_classes: bool = False,
) -> dict[str, Any]:
    try:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older torch
        payload = torch.load(checkpoint, map_location="cpu")

    if payload.get("format") != LINEAR_PROBE_HEAD_FORMAT:
        raise ValueError(f"Unsupported linear-probe head format in {checkpoint}: {payload.get('format')!r}")
    if int(payload.get("version", 0)) != LINEAR_PROBE_HEAD_VERSION:
        raise ValueError(f"Unsupported linear-probe head version in {checkpoint}: {payload.get('version')!r}")

    metadata = dict(payload.get("metadata", {}))
    mode = str(metadata.get("embedding_mode", ""))
    if strict and mode and mode != model.embedding_mode:
        raise ValueError(f"Linear-probe head embedding mode {mode!r} does not match model mode {model.embedding_mode!r}")

    head_state = payload.get("head", {})
    weight = head_state.get("weight")
    bias = head_state.get("bias")
    if not isinstance(weight, torch.Tensor) or not isinstance(bias, torch.Tensor):
        raise ValueError(f"Linear-probe head checkpoint is missing tensor weight/bias: {checkpoint}")
    if weight.ndim != 2 or bias.ndim != 1:
        raise ValueError(f"Linear-probe head has invalid shapes: weight={tuple(weight.shape)}, bias={tuple(bias.shape)}")
    if int(weight.shape[1]) != int(model.head.in_features):
        raise ValueError(
            f"Linear-probe head input dim {weight.shape[1]} does not match supervised head dim {model.head.in_features}"
        )

    classes = payload.get("classes")
    if not isinstance(classes, torch.Tensor):
        classes = torch.arange(weight.shape[0], dtype=torch.long)
    classes = classes.detach().long().cpu()
    if int(classes.numel()) != int(weight.shape[0]) or int(classes.numel()) != int(bias.shape[0]):
        raise ValueError("Linear-probe classes must align with saved head rows")
    if classes.numel() and (int(classes.min()) < 0 or int(classes.max()) >= model.head.out_features):
        raise ValueError(
            f"Linear-probe class ids must fit supervised head with {model.head.out_features} classes: "
            f"min={int(classes.min())}, max={int(classes.max())}"
        )

    expected_classes = torch.arange(model.head.out_features, dtype=torch.long)
    full_class_cover = torch.equal(classes.sort().values, expected_classes)
    if strict and not full_class_cover and not allow_partial_classes:
        raise ValueError(
            f"Linear-probe head covers {classes.numel()} classes, but supervised head expects "
            f"{model.head.out_features}. Set init.allow_partial_head_classes=true to permit this."
        )

    with torch.no_grad():
        weight = weight.to(device=model.head.weight.device, dtype=model.head.weight.dtype)
        bias = bias.to(device=model.head.bias.device, dtype=model.head.bias.dtype)
        if torch.equal(classes, expected_classes):
            model.head.weight.copy_(weight)
            model.head.bias.copy_(bias)
        else:
            target_classes = classes.to(model.head.weight.device)
            model.head.weight[target_classes] = weight
            model.head.bias[target_classes] = bias

    return {
        "checkpoint": str(checkpoint),
        "classes": int(classes.numel()),
        "embedding_mode": mode,
        "dataset": metadata.get("dataset"),
        "images_seen": metadata.get("images_seen"),
    }


def maybe_initialize_from_checkpoint(model: SupervisedViTClassifier, config: dict[str, Any]) -> dict[str, Any]:
    init_cfg = config.get("init", {})
    init_type = str(init_cfg.get("type", "scratch"))
    if init_type == "scratch":
        info = {"type": "scratch"}
        head_checkpoint = init_cfg.get("linear_probe_head") or init_cfg.get("head_checkpoint")
        if head_checkpoint:
            info["linear_probe_head"] = load_linear_probe_head(
                model,
                head_checkpoint,
                strict=bool(init_cfg.get("head_strict", True)),
                allow_partial_classes=bool(init_cfg.get("allow_partial_head_classes", False)),
            )
        return info
    if init_type in {"mae_encoder", "encoder"}:
        checkpoint = init_cfg.get("checkpoint")
        if not checkpoint:
            raise ValueError("init.type=mae_encoder requires init.checkpoint")
        info = load_mae_encoder_weights(model, checkpoint, strict=bool(init_cfg.get("strict", True)))
        info["type"] = init_type
        head_checkpoint = init_cfg.get("linear_probe_head") or init_cfg.get("head_checkpoint")
        if head_checkpoint:
            info["linear_probe_head"] = load_linear_probe_head(
                model,
                head_checkpoint,
                strict=bool(init_cfg.get("head_strict", True)),
                allow_partial_classes=bool(init_cfg.get("allow_partial_head_classes", False)),
            )
        return info
    raise ValueError(f"Unknown supervised initialization type: {init_type}")


def accuracy_counts(logits: torch.Tensor, labels: torch.Tensor, topk: tuple[int, ...] = (1, 5)) -> dict[int, int]:
    maxk = min(max(topk), logits.shape[1])
    pred = logits.topk(maxk, dim=1).indices
    out: dict[int, int] = {}
    for k in topk:
        kk = min(k, logits.shape[1])
        out[k] = int(pred[:, :kk].eq(labels.unsqueeze(1)).any(dim=1).sum().item())
    return out


@torch.no_grad()
def evaluate_classifier(
        model: nn.Module,
        loader: DataLoader,
        device: torch.device,
        amp: bool = True,
) -> dict[str, float | int]:
    model.eval()
    total = 0
    loss_sum = 0.0
    top1 = 0
    top5 = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True).float()
        labels = labels.to(device, non_blocking=True).long()
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available() and amp):
            logits = model(images)
            loss = F.cross_entropy(logits, labels, reduction="sum")
        counts = accuracy_counts(logits, labels, topk=(1, 5))
        batch = int(labels.numel())
        total += batch
        loss_sum += float(loss.detach().cpu())
        top1 += counts[1]
        top5 += counts[5]
    model.train()
    return {
        "num_samples": total,
        "loss": loss_sum / max(1, total),
        "top1_accuracy": top1 / max(1, total),
        "top5_accuracy": top5 / max(1, total),
    }


def append_eval_row(
        path: Path,
        run_id: str,
        step: int,
        images_seen: int,
        dataset: str,
        split: str,
        checkpoint: str,
        metrics: dict[str, float | int],
        notes: str,
) -> None:
    append_csv(
        path,
        {
            "run_id": run_id,
            "step": step,
            "images_seen": images_seen,
            "dataset": dataset,
            "split": split,
            "num_samples": metrics["num_samples"],
            "loss": metrics["loss"],
            "top1_accuracy": metrics["top1_accuracy"],
            "top5_accuracy": metrics["top5_accuracy"],
            "checkpoint": checkpoint,
            "notes": notes,
        },
        EVAL_COLUMNS,
    )


def completed_eval_images(path: Path) -> set[int]:
    if not path.exists():
        return set()
    with path.open("r", newline="", encoding="utf-8") as f:
        return {int(row["images_seen"]) for row in csv.DictReader(f) if row.get("images_seen")}


def next_eval_target(images_seen: int, interval: int) -> int:
    if interval <= 0:
        return 0
    return ((images_seen // interval) + 1) * interval


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    if args.disable_wandb:
        config.setdefault("wandb", {})["enabled"] = False
    seed = int(config.get("seed", 0))
    seed_everything(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_id = str(config.get("run_id", Path(args.config).stem))
    output_dir = ensure_dir(config.get("output_dir", f"results/runs/{run_id}"))
    train_cfg = config.setdefault("train", {})
    apply_target_instances(train_cfg, world_size=1)
    train_cfg.setdefault("lr_total_steps", int(train_cfg.get("steps", DEFAULT_TRAIN_STEPS)))

    if int(train_cfg.get("batch_size", DEFAULT_TRAIN_BATCH_SIZE)) <= 0:
        raise ValueError("train.batch_size must be positive")

    spec = dataset_spec(config)
    if spec.name.lower() not in {"imagenet", "imagenet1k", "imagenet-1k", "imagenet_1k"}:
        raise ValueError(f"This entry point is intentionally ImageNet-only, got {spec.name!r}")
    train_ds, eval_ds, note = build_dataset_pair(spec, seed=seed)
    train_stream = build_train_stream(config, spec, train_ds, seed=seed)
    eval_loader = build_eval_loader(config, spec, eval_ds)

    num_classes = int(config.get("num_classes", 1000))
    model = SupervisedViTClassifier(config.get("model", {}), num_classes=num_classes).to(device)
    init_info = maybe_initialize_from_checkpoint(model, config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", DEFAULT_TRAIN_LR)),
        weight_decay=float(train_cfg.get("weight_decay", DEFAULT_TRAIN_WEIGHT_DECAY)),
        betas=tuple(train_cfg.get("betas", DEFAULT_TRAIN_BETAS)),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available() and bool(train_cfg.get("amp", True)))
    start_step = load_checkpoint(args.resume, model, optimizer, scaler, source=train_stream, device=device) if args.resume else 0

    save_config(config, output_dir / "config.yaml")
    append_jsonl(
        output_dir / "metadata.jsonl",
        {
            "run_id": run_id,
            "git_commit": git_commit_hash(),
            "dataset_note": note,
            "init": init_info,
        },
    )

    wandb_run = init_wandb_if_enabled(config, output_dir, run_id)
    metrics_path = output_dir / "metrics.jsonl"
    eval_csv = output_dir / "supervised_eval.csv"
    steps = int(train_cfg.get("steps", DEFAULT_TRAIN_STEPS))
    log_every = int(train_cfg.get("log_every", DEFAULT_TRAIN_LOG_EVERY))
    interval = int(config.get("eval", {}).get("interval_instances", train_cfg.get("eval_interval_instances", 1_000_000)))
    clip_grad = float(train_cfg.get("clip_grad", DEFAULT_TRAIN_CLIP_GRAD))
    amp_enabled = bool(train_cfg.get("amp", True))
    checkpoint_every_instances = int(train_cfg.get("checkpoint_every_instances", interval))
    next_checkpoint_images = next_eval_target(images_seen_for_step(start_step, train_cfg, 1), checkpoint_every_instances)
    next_eval_images = next_eval_target(images_seen_for_step(start_step, train_cfg, 1), interval)

    def source_checkpoint_state() -> dict[str, Any] | None:
        state_fn = getattr(train_stream, "state_dict", None)
        if not callable(state_fn):
            return None
        return state_fn()

    def run_eval(step: int, images_seen: int, checkpoint: str) -> None:
        metrics = evaluate_classifier(model, eval_loader, device, amp=amp_enabled)
        append_eval_row(
            eval_csv,
            run_id=run_id,
            step=step,
            images_seen=images_seen,
            dataset=spec.name,
            split=spec.eval_split,
            checkpoint=checkpoint,
            metrics=metrics,
            notes=note,
        )
        wandb_log(
            wandb_run,
            {
                "test/imagenet1k_top-1_accuracy": float(metrics["top1_accuracy"]),
                "test/imagenet1k_top-5_accuracy": float(metrics["top5_accuracy"]),
                "test/imagenet1k_loss": float(metrics["loss"]),
            },
            images_seen,
        )

    model.train()
    if start_step == 0 and bool(config.get("eval", {}).get("initial", True)):
        initial_checkpoint = "random_init" if init_info.get("type") == "scratch" else str(init_info.get("checkpoint", "initialized"))
        run_eval(0, 0, initial_checkpoint)
    elif start_step > 0 and interval > 0:
        start_images = images_seen_for_step(start_step, train_cfg, 1)
        if start_images % interval == 0 and start_images not in completed_eval_images(eval_csv):
            run_eval(start_step, start_images, Path(args.resume).name)

    last_time = time.perf_counter()
    last_log_step = start_step
    for step in range(start_step, steps):
        lr = learning_rate(step, train_cfg)
        set_optimizer_lr(optimizer, lr)
        images, labels = train_stream.next_batch(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available() and amp_enabled):
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
        scaler.scale(loss).backward()
        if clip_grad > 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        else:
            grad_norm = torch.tensor(0.0, device=device)
        scaler.step(optimizer)
        scaler.update()

        images_seen = images_seen_for_step(step + 1, train_cfg, 1)
        if (step + 1) % log_every == 0 or step == start_step:
            now = time.perf_counter()
            elapsed = now - last_time
            last_time = now
            updates_since_log = max(1, step + 1 - last_log_step)
            last_log_step = step + 1
            counts = accuracy_counts(logits.detach(), labels, topk=(1, 5))
            batch_total = int(labels.numel())
            throughput = int(train_cfg.get("batch_size", DEFAULT_TRAIN_BATCH_SIZE)) * updates_since_log / max(elapsed, 1e-6)
            row = {
                "step": step + 1,
                "images_seen": images_seen,
                "loss": float(loss.detach().cpu()),
                "top1_accuracy": counts[1] / max(1, batch_total),
                "top5_accuracy": counts[5] / max(1, batch_total),
                "lr": lr,
                "grad_norm": float(grad_norm.detach().cpu()),
                "images_per_second": throughput,
            }
            append_jsonl(metrics_path, row)
            wandb_log(
                wandb_run,
                {
                    "train/loss": row["loss"],
                    "train/top-1_accuracy": row["top1_accuracy"],
                    "train/top-5_accuracy": row["top5_accuracy"],
                    "train/learning-rate": row["lr"],
                    "train/gradient_norm": row["grad_norm"],
                    "train/images_per_second": row["images_per_second"],
                },
                images_seen,
            )

        should_checkpoint = checkpoint_every_instances > 0 and next_checkpoint_images > 0 and images_seen >= next_checkpoint_images
        if should_checkpoint:
            while next_checkpoint_images <= images_seen:
                next_checkpoint_images += checkpoint_every_instances
            save_checkpoint(
                output_dir / f"checkpoint_step_{step + 1:07d}.pt",
                model,
                optimizer,
                scaler,
                step + 1,
                config,
                source_checkpoint_state(),
            )

        should_eval = interval > 0 and next_eval_images > 0 and images_seen >= next_eval_images
        if should_eval:
            eval_target = next_eval_images
            while next_eval_images <= images_seen:
                next_eval_images += interval
            checkpoint_name = f"checkpoint_step_{step + 1:07d}.pt" if should_checkpoint else ""
            run_eval(step + 1, eval_target, checkpoint_name)

    save_checkpoint(output_dir / "checkpoint_final.pt", model, optimizer, scaler, steps, config, source_checkpoint_state())
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
