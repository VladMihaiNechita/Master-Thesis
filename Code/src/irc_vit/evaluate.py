from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.irc_vit.config import (
    DEFAULT_EVAL_BATCH_SIZE,
    DEFAULT_EVAL_EMBEDDING_MODES,
    DEFAULT_EVAL_LINEAR_BATCH_SIZE,
    DEFAULT_EVAL_LINEAR_EPOCHS,
    DEFAULT_EVAL_LINEAR_LOSS,
    DEFAULT_EVAL_LINEAR_LR,
    DEFAULT_EVAL_LINEAR_MOMENTUM,
    DEFAULT_EVAL_LINEAR_OPTIMIZER,
    DEFAULT_EVAL_OUTPUT_CSV,
    DEFAULT_EVAL_LINEAR_WEIGHT_DECAY,
    DEFAULT_EVAL_NUM_WORKERS,
    DEFAULT_EVAL_PROBES,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TRAIN_BATCH_SIZE,
    DEFAULT_WANDB_PROJECT,
    dataset_eval_loader_override,
    eval_perceptron_training_config,
    load_config,
    validate_eval_embedding_modes,
    validate_eval_probes,
)
from src.irc_vit.data import DatasetSpec, build_dataset_pair, cache_dataset_name
from src.irc_vit.model import build_mae
from src.irc_vit.utils import append_csv, ensure_dir


RESULT_COLUMNS = [
    "run_id",
    "checkpoint",
    "pretrain_source",
    "generator",
    "generator_K",
    "model",
    "image_size",
    "pretrain_steps",
    "images_seen",
    "objective",
    "eval_dataset",
    "embedding_mode",
    "probe_type",
    "seed",
    "accuracy",
    "notes",
]

EVAL_IMAGES_SEEN_KEY = "_eval_images_seen"
LEGACY_EVAL_IMAGES_SEEN_KEY = "eval/images_seen"
VALIDATION_DATASETS = {
    "cifar10",
    "tiny_imagenet",
    "tinyimagenet",
    "food101",
    "pets",
    "oxford_iiit_pets",
    "oxfordiiitpet",
    "dtd",
    "eurosat",
    "shapecolorobjects",
    "shape_color_objects",
    "linefieldorientation",
    "line_field_orientation",
    "texturemosaic",
    "texture_mosaic",
    "checkergridfield",
    "checker_grid_field",
}
TEST_DATASETS = {
    "cifar100",
    "imagenet1k",
    "imagenet_1k",
    "imagenet_r",
    "imagenet",
    "cars",
    "stanford_cars",
    "resisc45",
    "sun397",
}
RECONSTRUCTION_PROBES = (
    "raw_pixel_mse",
    "raw_pixel_mae",
    "patch_normalized_mse",
    "patch_normalized_mae",
)
LINEAR_PROBE_HEAD_FORMAT = "irc_vit_linear_probe_head"
LINEAR_PROBE_HEAD_VERSION = 1


def _argv_without_sbatch_go(argv: list[str] | None) -> list[str] | None:
    if argv and argv[0] == "go":
        return argv[1:]
    return argv


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate frozen ViT representations")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default=DEFAULT_EVAL_OUTPUT_CSV)
    parser.add_argument("--disable-wandb", action="store_true")
    return parser.parse_args(_argv_without_sbatch_go(argv))


def dataset_spec(raw: dict[str, Any], image_size: int) -> DatasetSpec:
    values = dict(raw)
    values.setdefault("image_size", image_size)
    return DatasetSpec(**values)


def dataset_eval_loader_settings(raw: dict[str, Any], eval_cfg: dict[str, Any]) -> tuple[int, int]:
    defaults = dataset_eval_loader_override(str(raw.get("name", "")))
    batch_size = int(raw.get("eval_batch_size", defaults.get("batch_size", eval_cfg.get("batch_size", DEFAULT_EVAL_BATCH_SIZE))))
    num_workers = int(raw.get("eval_num_workers", defaults.get("num_workers", eval_cfg.get("num_workers", DEFAULT_EVAL_NUM_WORKERS))))
    return batch_size, num_workers


def checkpoint_metadata(checkpoint: str | Path, ckpt_config: dict[str, Any], checkpoint_step: int | None = None) -> dict[str, Any]:
    source = ckpt_config.get("source", {})
    generator = ""
    generator_k = ""
    if source.get("type", "generator") == "none":
        generator = "none"
    elif source.get("type", "generator") == "generator":
        gen = source.get("generator", {})
        generator = gen.get("name", "")
        generator_k = gen.get("params", {}).get("k", "")
    else:
        generator = source.get("params", {}).get("name", "real")
    return {
        "run_id": ckpt_config.get("run_id", Path(checkpoint).parent.name),
        "checkpoint": str(checkpoint),
        "pretrain_source": source.get("type", "generator"),
        "generator": generator,
        "generator_K": generator_k,
        "model": ckpt_config.get("model", {}).get("name", ""),
        "image_size": ckpt_config.get("model", {}).get("image_size", ""),
        "pretrain_steps": checkpoint_step if checkpoint_step is not None else ckpt_config.get("train", {}).get("steps", ""),
        "objective": "mae",
        "seed": ckpt_config.get("seed", ""),
    }


def checkpoint_images_seen(ckpt_config: dict[str, Any], checkpoint_step: int | None = None) -> int:
    train = ckpt_config.get("train", {})
    if checkpoint_step is not None:
        total_steps = int(train.get("steps", 0))
        effective_instances = int(train.get("effective_instances") or train.get("target_instances") or 0)
        if total_steps > 0 and effective_instances > 0:
            return round(float(checkpoint_step) / float(total_steps) * float(effective_instances))
        return int(checkpoint_step) * int(train.get("batch_size", DEFAULT_TRAIN_BATCH_SIZE))
    return int(train.get("effective_instances") or train.get("target_instances") or 0)


def append_result(path: Path, meta: dict[str, Any], dataset: str, mode: str, probe: str, accuracy, notes: str = "") -> None:
    row = dict(meta)
    row.update({
        "eval_dataset": dataset,
        "embedding_mode": mode,
        "probe_type": probe,
        "accuracy": "" if accuracy is None else accuracy,
        "notes": notes,
    })
    append_csv(path, row, RESULT_COLUMNS)


def init_wandb_if_enabled(cfg: dict[str, Any], output_dir: Path, run_id: str):
    wandb_cfg = cfg.get("wandb", {})
    if not bool(wandb_cfg.get("enabled", False)):
        return None
    wandb_root = output_dir / "wandb" / run_id
    for path in (wandb_root, wandb_root / "cache", wandb_root / "config", wandb_root / "data", wandb_root / "tmp"):
        path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_DIR", str(wandb_root))
    os.environ.setdefault("WANDB_CACHE_DIR", str(wandb_root / "cache"))
    os.environ.setdefault("WANDB_CONFIG_DIR", str(wandb_root / "config"))
    os.environ.setdefault("WANDB_DATA_DIR", str(wandb_root / "data"))
    try:
        import wandb
    except Exception as exc:
        raise RuntimeError("W&B eval logging is enabled, but wandb is not installed.") from exc
    return wandb.init(
        project=str(wandb_cfg.get("project", DEFAULT_WANDB_PROJECT)),
        entity=wandb_cfg.get("entity"),
        id=wandb_cfg.get("id"),
        name=str(wandb_cfg.get("name", f"eval-{run_id}")),
        group=str(wandb_cfg.get("group", "eval")),
        tags=list(wandb_cfg.get("tags", ["eval"])),
        notes=wandb_cfg.get("notes"),
        config=cfg,
        dir=str(wandb_root),
        mode=str(wandb_cfg.get("mode", os.environ.get("WANDB_MODE", "online"))),
        resume="allow",
    )


def define_wandb_eval_metrics(run, keys=None) -> None:
    if run is None:
        return
    run.define_metric(EVAL_IMAGES_SEEN_KEY, hidden=True)
    run.define_metric(LEGACY_EVAL_IMAGES_SEEN_KEY, hidden=True, overwrite=True)
    for key in keys or []:
        run.define_metric(str(key), step_metric=EVAL_IMAGES_SEEN_KEY)


def wandb_eval_log(run, values: dict[str, float], images_seen: int) -> None:
    if run is not None and values:
        define_wandb_eval_metrics(run, values.keys())
        payload = {EVAL_IMAGES_SEEN_KEY: images_seen}
        payload.update(values)
        run.log(payload)


def wandb_dataset_role(dataset: str) -> str:
    normalized = str(dataset).lower().replace("-", "_")
    if normalized in VALIDATION_DATASETS:
        return "validation"
    if normalized in TEST_DATASETS:
        return "test"
    return "validation"


def wandb_eval_metric_name(mode: str, probe: str) -> str | None:
    if mode == "reconstruction":
        return {
            "raw_pixel_mse": "raw_pixel_MSE",
            "raw_pixel_mae": "raw_pixel_MAE",
            "patch_normalized_mse": "patch-normalized_MSE",
            "patch_normalized_mae": "patch-normalized_MAE",
            "mse": "raw_pixel_MSE",
            "mae": "raw_pixel_MAE",
        }.get(probe)
    if probe == "linear":
        return "top-1_accuracy"
    if probe == "linear_top5":
        return "top-5_accuracy"
    if probe == "linear_loss":
        return "cross-entropy_loss"
    return None


def wandb_eval_dataset_key(dataset: str, mode: str, probe: str) -> str | None:
    metric = wandb_eval_metric_name(mode, probe)
    if metric is None:
        return None
    return f"{wandb_dataset_role(dataset)}/{dataset}_{metric}"


def wandb_eval_mean_key(role: str, mode: str, probe: str) -> str | None:
    if role != "validation":
        return None
    metric = wandb_eval_metric_name(mode, probe)
    if metric is None:
        return None
    return f"validation_mean/{metric}"


def wandb_eval_mean_keys(mode: str, probe: str, role: str = "validation") -> list[str]:
    means_key = wandb_eval_mean_key(role, mode, probe)
    if means_key is None:
        return []
    return [means_key]


def load_mae_from_checkpoint(checkpoint: str | Path, device: torch.device):
    payload = torch.load(checkpoint, map_location="cpu")
    config = payload["config"]
    model = build_mae(config)
    model.load_state_dict(payload["model"])
    return model.eval().to(device), config, int(payload.get("step", 0))


@torch.no_grad()
def extract_embeddings(
        encoder,
        dataset: Dataset,
        batch_size: int,
        device: torch.device,
        embedding_mode: str = "cls_mean",
        num_workers: int = DEFAULT_EVAL_NUM_WORKERS,
        desc: str = "extract",
) -> tuple[torch.Tensor, torch.Tensor]:
    embedding_mode = validate_eval_embedding_modes(embedding_mode)[0]
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda")
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    encoder.eval()
    for batch in tqdm(loader, desc=desc, leave=False):
        images, y = batch[0], batch[1]
        images = images.to(device, non_blocking=True).float()
        features.append(encoder.encode(images, mode=embedding_mode).cpu())
        labels.append(y.cpu().long())
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)


def resolve_probe_device(requested: torch.device | str | None = None) -> torch.device:
    if requested is None or str(requested) == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA probe evaluation was requested, but CUDA is not available")
    return device


def _normalized_linear_optimizer(name: str) -> str:
    return str(name).lower().replace("torch.optim.", "")


def linear_probe_head_save_path(
        eval_cfg: dict[str, Any],
        output_dir: str | Path,
        dataset_name: str,
        embedding_mode: str,
        probe: str,
) -> Path | None:
    if probe != "linear" or not bool(eval_cfg.get("save_linear_probe_heads", False)):
        return None
    root = eval_cfg.get("linear_probe_head_dir")
    root_path = Path(root) if root else Path(output_dir) / "linear_probe_heads"
    filename = f"{cache_dataset_name(dataset_name)}_{cache_dataset_name(embedding_mode)}_{cache_dataset_name(probe)}.pt"
    return ensure_dir(root_path) / filename


def save_linear_probe_head(
        path: str | Path,
        head: nn.Linear,
        classes: torch.Tensor,
        feature_mean: torch.Tensor,
        feature_std: torch.Tensor,
        metadata: dict[str, Any] | None = None,
) -> None:
    weight = head.weight.detach().float().cpu()
    bias = head.bias.detach().float().cpu()
    mean = feature_mean.detach().float().cpu()
    std = feature_std.detach().float().cpu().clamp_min(1e-6)

    raw_weight = weight / std.unsqueeze(0)
    raw_bias = bias - (weight * (mean / std).unsqueeze(0)).sum(dim=1)

    payload = {
        "format": LINEAR_PROBE_HEAD_FORMAT,
        "version": LINEAR_PROBE_HEAD_VERSION,
        "classes": classes.detach().long().cpu(),
        "embedding_dim": int(weight.shape[1]),
        "num_classes": int(weight.shape[0]),
        "feature_mean": mean,
        "feature_std": std,
        "normalized_head": {
            "weight": weight,
            "bias": bias,
        },
        "head": {
            "weight": raw_weight,
            "bias": raw_bias,
        },
        "metadata": dict(metadata or {}),
    }
    path = Path(path)
    ensure_dir(path.parent)
    torch.save(payload, path)


def linear_probe_metrics(
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        eval_features: torch.Tensor,
        eval_labels: torch.Tensor,
        epochs: int | None = None,
        batch_size: int = DEFAULT_EVAL_LINEAR_BATCH_SIZE,
        lr: float = DEFAULT_EVAL_LINEAR_LR,
        weight_decay: float = DEFAULT_EVAL_LINEAR_WEIGHT_DECAY,
        optimizer_name: str = DEFAULT_EVAL_LINEAR_OPTIMIZER,
        momentum: float = DEFAULT_EVAL_LINEAR_MOMENTUM,
        loss_name: str = DEFAULT_EVAL_LINEAR_LOSS,
        device: torch.device | str | None = None,
        seed: int = 0,
        save_head_path: str | Path | None = None,
        head_metadata: dict[str, Any] | None = None,
) -> dict[str, float]:
    compute_device = resolve_probe_device(device)
    x_train_cpu = train_features.detach().float().cpu()
    y_train_raw = train_labels.detach().long().cpu()
    x_eval_cpu = eval_features.detach().float().cpu()
    y_eval_raw = eval_labels.detach().long().cpu()
    if x_train_cpu.ndim != 2 or x_eval_cpu.ndim != 2:
        raise ValueError("linear_probe_metrics expects [N, D] train/eval feature tensors")
    if x_train_cpu.shape[0] == 0 or x_eval_cpu.shape[0] == 0:
        return {"accuracy": 0.0, "top1_accuracy": 0.0, "top5_accuracy": 0.0, "loss": float("nan")}
    if torch.unique(y_train_raw).numel() < 2:
        return {"accuracy": 0.0, "top1_accuracy": 0.0, "top5_accuracy": 0.0, "loss": float("nan")}

    classes = torch.unique(torch.cat([y_train_raw, y_eval_raw])).sort().values
    y_train = torch.searchsorted(classes, y_train_raw)
    y_eval = torch.searchsorted(classes, y_eval_raw)

    x_train = x_train_cpu.to(compute_device)
    x_eval = x_eval_cpu.to(compute_device)
    feature_mean = x_train.mean(dim=0)
    feature_std = x_train.std(dim=0, unbiased=False).clamp_min_(1e-6)
    x_train.sub_(feature_mean).div_(feature_std)
    x_eval.sub_(feature_mean).div_(feature_std)
    y_train = y_train.to(compute_device)
    y_eval = y_eval.to(compute_device)

    if _normalized_linear_optimizer(optimizer_name) != "sgd":
        raise ValueError(f"Unsupported linear probe optimizer: {optimizer_name}")
    if str(loss_name).lower() != "cross_entropy":
        raise ValueError(f"Unsupported linear probe loss: {loss_name}")
    with torch.random.fork_rng():
        torch.manual_seed(int(seed))
        head = nn.Linear(x_train.shape[1], int(classes.numel())).to(compute_device)
    optimizer = torch.optim.SGD(
        head.parameters(),
        lr=float(lr),
        momentum=float(momentum),
        weight_decay=float(weight_decay),
    )
    train_epochs = int(epochs if epochs is not None else DEFAULT_EVAL_LINEAR_EPOCHS)
    train_epochs = max(1, train_epochs)
    batch_size = max(1, int(batch_size))
    generator = torch.Generator(device=compute_device).manual_seed(int(seed))

    head.train()
    for _ in range(train_epochs):
        order = torch.randperm(x_train.shape[0], generator=generator, device=compute_device)
        for start in range(0, x_train.shape[0], batch_size):
            idx = order[start:start + batch_size]
            xb = x_train[idx]
            yb = y_train[idx]
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(head(xb), yb)
            loss.backward()
            optimizer.step()

    if save_head_path is not None:
        save_linear_probe_head(save_head_path, head, classes, feature_mean, feature_std, head_metadata)

    head.eval()
    top1_correct = 0
    top5_correct = 0
    total = 0
    loss_sum = 0.0
    with torch.no_grad():
        for start in range(0, x_eval.shape[0], batch_size):
            xb = x_eval[start:start + batch_size]
            yb = y_eval[start:start + batch_size]
            logits = head(xb)
            loss = F.cross_entropy(logits, yb, reduction="sum")
            pred = logits.argmax(dim=1)
            top1_correct += int((pred == yb).sum().item())
            topk = min(5, logits.shape[1])
            top5 = logits.topk(topk, dim=1).indices
            top5_correct += int(top5.eq(yb.unsqueeze(1)).any(dim=1).sum().item())
            total += int(yb.numel())
            loss_sum += float(loss.detach().cpu())

    top1 = top1_correct / max(1, total)
    return {
        "accuracy": top1,
        "top1_accuracy": top1,
        "top5_accuracy": top5_correct / max(1, total),
        "loss": loss_sum / max(1, total),
    }


def linear_probe_accuracy(
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        eval_features: torch.Tensor,
        eval_labels: torch.Tensor,
        epochs: int | None = None,
        batch_size: int = DEFAULT_EVAL_LINEAR_BATCH_SIZE,
        lr: float = DEFAULT_EVAL_LINEAR_LR,
        weight_decay: float = DEFAULT_EVAL_LINEAR_WEIGHT_DECAY,
        optimizer_name: str = DEFAULT_EVAL_LINEAR_OPTIMIZER,
        momentum: float = DEFAULT_EVAL_LINEAR_MOMENTUM,
        loss_name: str = DEFAULT_EVAL_LINEAR_LOSS,
        device: torch.device | str | None = None,
        seed: int = 0,
) -> float:
    return linear_probe_metrics(
        train_features,
        train_labels,
        eval_features,
        eval_labels,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        optimizer_name=optimizer_name,
        momentum=momentum,
        loss_name=loss_name,
        device=device,
        seed=seed,
    )["accuracy"]


@torch.no_grad()
def reconstruction_metrics(
        model,
        dataset,
        batch_size: int,
        device: torch.device,
        num_workers: int,
        seed: int,
) -> dict[str, float]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda")
    gen = torch.Generator(device=device).manual_seed(seed) if device.type != "cpu" else torch.Generator().manual_seed(seed)
    total_masked = 0.0
    sums = {probe: 0.0 for probe in RECONSTRUCTION_PROBES}
    model.eval()
    for batch in loader:
        images = batch[0].to(device, non_blocking=True).float()
        out = model(images, generator=gen)
        pred = out["pred"]
        target = out["target"]
        raw_abs = (pred - target).abs()
        raw_sq = (pred - target).pow(2)
        patch_mean = target.mean(dim=-1, keepdim=True)
        patch_std = target.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
        norm_pred = (pred - patch_mean) / patch_std
        norm_target = (target - patch_mean) / patch_std
        norm_abs = (norm_pred - norm_target).abs()
        norm_sq = (norm_pred - norm_target).pow(2)
        per_patch = {
            "raw_pixel_mae": raw_abs.mean(dim=-1),
            "raw_pixel_mse": raw_sq.mean(dim=-1),
            "patch_normalized_mae": norm_abs.mean(dim=-1),
            "patch_normalized_mse": norm_sq.mean(dim=-1),
        }
        mask = out["mask"]
        masked = float(mask.sum().detach().cpu())
        for probe, values in per_patch.items():
            sums[probe] += float((values * mask).sum().detach().cpu())
        total_masked += masked
    denom = max(total_masked, 1.0)
    return {probe: value / denom for probe, value in sums.items()}


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.disable_wandb:
        cfg.setdefault("wandb", {})["enabled"] = False
    out_csv = Path(args.out)
    output_dir = ensure_dir(cfg.get("output_dir", DEFAULT_OUTPUT_DIR))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if str(args.checkpoint).lower() in {"random_init", "none", "no_pretraining"}:
        mae_model = build_mae(cfg).to(device).eval()
        encoder = mae_model.encoder
        ckpt_config = {
            "run_id": "random_init",
            "source": {"type": "none"},
            "model": cfg.get("model", {}),
            "train": {"steps": 0},
            "seed": cfg.get("seed", 0),
        }
        checkpoint_step = 0
    else:
        mae_model, ckpt_config, checkpoint_step = load_mae_from_checkpoint(args.checkpoint, device)
        encoder = mae_model.encoder

    meta = checkpoint_metadata(args.checkpoint, ckpt_config, checkpoint_step)
    images_seen = checkpoint_images_seen(ckpt_config, checkpoint_step)
    meta["images_seen"] = images_seen
    wandb_run = init_wandb_if_enabled(cfg, Path(output_dir), str(meta["run_id"]))
    wandb_values: dict[tuple[str, str, str], list[float]] = {}

    eval_cfg = cfg.get("eval", {})
    linear_cfg = eval_perceptron_training_config(eval_cfg, ckpt_config.get("model", cfg.get("model", {})))
    image_size = int(eval_cfg.get("image_size", ckpt_config.get("model", {}).get("image_size", DEFAULT_IMAGE_SIZE)))
    embedding_modes = validate_eval_embedding_modes(eval_cfg.get("embedding_modes", DEFAULT_EVAL_EMBEDDING_MODES))
    probes = validate_eval_probes(eval_cfg.get("probes", DEFAULT_EVAL_PROBES))
    seed = int(cfg.get("seed", ckpt_config.get("seed", 0)))

    for raw_dataset in cfg.get("datasets", []):
        spec = dataset_spec(raw_dataset, image_size)
        batch_size, num_workers = dataset_eval_loader_settings(raw_dataset, eval_cfg)
        try:
            train_ds, eval_ds, note = build_dataset_pair(spec, seed=seed)
        except Exception as exc:
            append_result(out_csv, meta, spec.name, "", "skipped", None, f"{type(exc).__name__}: {exc}")
            continue

        if bool(eval_cfg.get("reconstruction_mae", False)):
            try:
                recon = reconstruction_metrics(mae_model, eval_ds, batch_size, device, num_workers, seed)
                payload = {}
                role = wandb_dataset_role(spec.name)
                for probe in RECONSTRUCTION_PROBES:
                    append_result(out_csv, meta, spec.name, "reconstruction", probe, recon[probe], note)
                    key = wandb_eval_dataset_key(spec.name, "reconstruction", probe)
                    if key is not None:
                        payload[key] = recon[probe]
                        wandb_values.setdefault((role, "reconstruction", probe), []).append(float(recon[probe]))
                wandb_eval_log(wandb_run, payload, images_seen)
            except Exception as exc:
                append_result(out_csv, meta, spec.name, "reconstruction", "raw_pixel_mae", None, f"{type(exc).__name__}: {exc}")

        for mode in embedding_modes:
            train_features, train_labels = extract_embeddings(
                encoder, train_ds, batch_size, device, mode, num_workers, desc=f"{spec.name} train {mode}",
            )
            eval_features, eval_labels = extract_embeddings(
                encoder, eval_ds, batch_size, device, mode, num_workers, desc=f"{spec.name} eval {mode}",
            )

            for probe in probes:
                try:
                    head_path = linear_probe_head_save_path(eval_cfg, output_dir, spec.name, mode, probe)
                    metrics = linear_probe_metrics(
                        train_features,
                        train_labels,
                        eval_features,
                        eval_labels,
                        epochs=int(linear_cfg["epochs"]),
                        batch_size=int(linear_cfg["batch_size"]),
                        lr=float(linear_cfg["lr"]),
                        weight_decay=float(linear_cfg["weight_decay"]),
                        optimizer_name=str(linear_cfg["optimizer"]),
                        momentum=float(linear_cfg["momentum"]),
                        loss_name=str(linear_cfg["loss"]),
                        device=linear_cfg["device"],
                        seed=seed,
                        save_head_path=head_path,
                        head_metadata={
                            "dataset": spec.name,
                            "embedding_mode": mode,
                            "probe": probe,
                            "checkpoint": str(args.checkpoint),
                            "images_seen": int(images_seen),
                            "seed": int(seed),
                            "linear": dict(linear_cfg),
                        },
                    )
                    acc = metrics["top1_accuracy"]
                    top5 = metrics["top5_accuracy"]
                    linear_loss = metrics["loss"]
                    role = wandb_dataset_role(spec.name)
                    append_result(out_csv, meta, spec.name, mode, probe, acc, note)
                    key = wandb_eval_dataset_key(spec.name, mode, probe)
                    if key is not None:
                        wandb_eval_log(wandb_run, {key: acc}, images_seen)
                        wandb_values.setdefault((role, mode, probe), []).append(float(acc))
                    if probe == "linear":
                        append_result(out_csv, meta, spec.name, mode, "linear_top5", top5, note)
                        top5_key = wandb_eval_dataset_key(spec.name, mode, "linear_top5")
                        if top5_key is not None:
                            wandb_eval_log(wandb_run, {top5_key: top5}, images_seen)
                            wandb_values.setdefault((role, mode, "linear_top5"), []).append(float(top5))
                        append_result(out_csv, meta, spec.name, mode, "linear_loss", linear_loss, note)
                        loss_key = wandb_eval_dataset_key(spec.name, mode, "linear_loss")
                        if loss_key is not None:
                            wandb_eval_log(wandb_run, {loss_key: linear_loss}, images_seen)
                            wandb_values.setdefault((role, mode, "linear_loss"), []).append(float(linear_loss))
                except Exception as exc:
                    append_result(out_csv, meta, spec.name, mode, probe, None, f"{type(exc).__name__}: {exc}")

    summary_path = Path(output_dir) / "eval_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({"checkpoint": str(args.checkpoint), "results_csv": str(out_csv)}, indent=2), encoding="utf-8")

    if wandb_run is not None:
        means = {}
        for (role, mode, probe), values in wandb_values.items():
            if not values:
                continue
            value = sum(values) / len(values)
            for key in wandb_eval_mean_keys(mode, probe, role):
                means[key] = value
        wandb_eval_log(wandb_run, means, images_seen)
        wandb_run.finish()


if __name__ == "__main__":
    main()
