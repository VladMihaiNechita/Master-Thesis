from __future__ import annotations

import argparse
import inspect
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from src.irc_vit.config import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_PREVIEW_GRID_NROW,
    DEFAULT_PREVIEW_MAX_IMAGES,
    DEFAULT_RECONSTRUCTION_PREVIEW_MAX_IMAGES,
    DEFAULT_TRAIN_BATCH_SIZE,
    DEFAULT_TRAIN_BETAS,
    DEFAULT_TRAIN_CHECKPOINT_EVERY,
    DEFAULT_TRAIN_CLIP_GRAD,
    DEFAULT_TRAIN_LOG_EVERY,
    DEFAULT_TRAIN_LR,
    DEFAULT_TRAIN_MIN_LR,
    DEFAULT_TRAIN_PREVIEW_EVERY,
    DEFAULT_TRAIN_STEPS,
    DEFAULT_TRAIN_VAL_BATCHES,
    DEFAULT_TRAIN_VAL_EVERY,
    DEFAULT_TRAIN_VALIDATION_INTERVAL_INSTANCES,
    DEFAULT_TRAIN_WARMUP_STEPS,
    DEFAULT_TRAIN_WEIGHT_DECAY,
    DEFAULT_WANDB_PROJECT,
    load_config,
    save_config,
)
from src.irc_vit.data import RealImageBatchSource
from src.irc_vit.generators import build_generator
from src.irc_vit.model import build_mae
from src.irc_vit.utils import (
    append_jsonl,
    barrier,
    ensure_dir,
    git_commit_hash,
    init_distributed_if_needed,
    is_main_process,
    save_grid,
    save_labeled_grid_rows,
    seed_everything,
)

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - tensorboard is optional on clusters
    SummaryWriter = None


def init_wandb_if_enabled(config: dict[str, Any], output_dir: Path, run_id: str):
    wandb_cfg = config.get("wandb", {})
    if not bool(wandb_cfg.get("enabled", False)):
        return None
    wandb_root = output_dir / "wandb"
    for path in (wandb_root, wandb_root / "cache", wandb_root / "config", wandb_root / "data", wandb_root / "tmp"):
        path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_DIR", str(wandb_root))
    os.environ.setdefault("WANDB_CACHE_DIR", str(wandb_root / "cache"))
    os.environ.setdefault("WANDB_CONFIG_DIR", str(wandb_root / "config"))
    os.environ.setdefault("WANDB_DATA_DIR", str(wandb_root / "data"))
    try:
        import wandb
    except Exception as exc:  # pragma: no cover - depends on local/cluster env
        raise RuntimeError("W&B logging is enabled, but the wandb package is not available.") from exc

    mode = str(wandb_cfg.get("mode", os.environ.get("WANDB_MODE", "online")))
    run = wandb.init(
        project=str(wandb_cfg.get("project", DEFAULT_WANDB_PROJECT)),
        entity=wandb_cfg.get("entity"),
        id=wandb_cfg.get("id"),
        name=str(wandb_cfg.get("name", run_id)),
        group=wandb_cfg.get("group"),
        tags=list(wandb_cfg.get("tags", [])),
        notes=wandb_cfg.get("notes"),
        config=config,
        dir=str(wandb_root),
        mode=mode,
        resume="allow",
    )
    run.define_metric("train/*")
    run.define_metric("validation/*")
    run.define_metric("images/*")
    return run


def wandb_log(run, values: dict[str, Any], images_seen: int) -> None:
    if run is not None:
        payload = dict(values)
        run.log(payload, step=images_seen)


def wandb_image(run, path: Path, key: str, caption: str, images_seen: int) -> None:
    if run is None:
        return
    import wandb

    run.log({key: wandb.Image(str(path), caption=caption)}, step=images_seen)


def preview_caption(run_id: str, images_seen: int) -> str:
    return f"{run_id} | images_seen={images_seen}"


@torch.no_grad()
def save_pretrain_source_preview(images: torch.Tensor, path: Path, run_id: str) -> None:
    save_grid(
        images[: min(DEFAULT_PREVIEW_MAX_IMAGES, images.shape[0])],
        path,
        nrow=DEFAULT_PREVIEW_GRID_NROW,
        title=f"{run_id} pretraining source",
    )


@torch.no_grad()
def save_reconstruction_preview(model: nn.Module, images: torch.Tensor, path: Path, run_id: str) -> None:
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    was_training = raw_model.training
    raw_model.eval()
    preview = raw_model.reconstruction_preview(images[: min(DEFAULT_RECONSTRUCTION_PREVIEW_MAX_IMAGES, images.shape[0])])
    save_labeled_grid_rows(
        [
            ("Original input", preview["original"]),
            ("Masked input", preview["masked_input"]),
            ("MAE reconstruction", preview["reconstruction"]),
        ],
        path,
        nrow=DEFAULT_PREVIEW_GRID_NROW,
        title=f"{run_id} reconstruction preview",
    )
    raw_model.train(was_training)


class GeneratorBatchSource:
    def __init__(self, source_config: dict[str, Any], image_size: int, batch_size: int):
        self.generator = build_generator(source_config.get("generator", source_config))
        self.image_size = int(image_size)
        self.batch_size = int(batch_size)
        self.supports_refresh = "refresh" in inspect.signature(self.generator.generate).parameters

    def set_step(self, step: int) -> None:
        setter = getattr(self.generator, "set_step", None)
        if callable(setter):
            setter(step)

    def set_progress(self, step: int, images_seen: int) -> None:
        setter = getattr(self.generator, "set_progress", None)
        if callable(setter):
            setter(step=step, images_seen=images_seen)
            return
        self.set_step(step)

    def next_batch(self, device: torch.device, refresh: bool = True) -> torch.Tensor:
        if self.supports_refresh:
            return self.generator.generate(self.batch_size, self.image_size, device, refresh=refresh)
        return self.generator.generate(self.batch_size, self.image_size, device)

    def state_dict(self) -> dict[str, Any] | None:
        state_fn = getattr(self.generator, "state_dict", None)
        if callable(state_fn):
            return state_fn()
        return None

    def load_state_dict(self, state: dict[str, Any], device: torch.device | str | None = None) -> None:
        load_fn = getattr(self.generator, "load_state_dict", None)
        if callable(load_fn):
            load_fn(state, device=device)


def _argv_without_sbatch_go(argv: list[str] | None) -> list[str] | None:
    if argv and argv[0] == "go":
        return argv[1:]
    return argv


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MAE-style ViT pretraining")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default="")
    return parser.parse_args(_argv_without_sbatch_go(argv))


def build_source(config: dict[str, Any], image_size: int, batch_size: int, seed: int):
    source_cfg = config.get("source", {})
    source_type = str(source_cfg.get("type", "generator"))
    if source_type == "generator":
        return GeneratorBatchSource(source_cfg, image_size, batch_size)
    if source_type == "real":
        return RealImageBatchSource(source_cfg.get("params", {}), image_size, batch_size, seed)
    raise ValueError(f"Unknown pretraining source type: {source_type}")


def learning_rate(step: int, cfg: dict[str, Any]) -> float:
    base_lr = float(cfg.get("lr", DEFAULT_TRAIN_LR))
    min_lr = float(cfg.get("min_lr", DEFAULT_TRAIN_MIN_LR))
    warmup = int(cfg.get("warmup_steps", DEFAULT_TRAIN_WARMUP_STEPS))
    total = max(1, int(cfg.get("lr_total_steps", cfg.get("steps", DEFAULT_TRAIN_STEPS))))
    if warmup > 0 and step < warmup:
        return base_lr * float(step + 1) / float(warmup)
    progress = (step - warmup) / max(1, total - warmup)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
    return min_lr + (base_lr - min_lr) * cosine


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def apply_target_instances(train_cfg: dict[str, Any], world_size: int) -> None:
    target_instances = int(train_cfg.get("target_instances", 0))
    if target_instances <= 0:
        return
    batch_size = int(train_cfg.get("batch_size", DEFAULT_TRAIN_BATCH_SIZE))
    global_batch_size = batch_size * max(1, world_size)
    steps = math.ceil(target_instances / global_batch_size)
    train_cfg["steps"] = steps
    train_cfg["effective_instances"] = steps * global_batch_size


def images_seen_for_step(step: int, train_cfg: dict[str, Any], world_size: int) -> int:
    batch_size = int(train_cfg.get("batch_size", DEFAULT_TRAIN_BATCH_SIZE))
    return int(step) * batch_size * max(1, world_size)


def next_validation_target(images_seen: int, interval: int) -> int:
    if interval <= 0:
        return 0
    return ((images_seen // interval) + 1) * interval


def save_checkpoint(
        path: Path,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler,
        step: int,
        config: dict[str, Any],
        source_state: dict[str, Any] | None = None,
) -> None:
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    payload = {
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": step,
        "config": config,
        "git_commit": git_commit_hash(),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
    }
    if source_state is not None:
        payload["source_state"] = source_state
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, model: nn.Module, optimizer=None, scaler=None, source=None, device=None) -> int:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - older torch
        payload = torch.load(path, map_location="cpu")
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    raw_model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    if source is not None and payload.get("source_state") is not None:
        loader = getattr(source, "load_state_dict", None)
        if callable(loader):
            loader(payload["source_state"], device=device)
    rng_state = payload.get("rng")
    if isinstance(rng_state, dict):
        if rng_state.get("python") is not None:
            random.setstate(rng_state["python"])
        if rng_state.get("numpy") is not None:
            np.random.set_state(rng_state["numpy"])
        if rng_state.get("torch") is not None:
            torch.set_rng_state(rng_state["torch"])
        if torch.cuda.is_available() and rng_state.get("cuda") is not None:
            torch.cuda.set_rng_state_all(rng_state["cuda"])
    return int(payload.get("step", 0))


@torch.no_grad()
def validation_loss(model: nn.Module, source, device: torch.device, batches: int) -> float:
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    raw_model.eval()
    losses = []
    for _ in range(max(0, batches)):
        if getattr(source, "supports_refresh", False):
            images = source.next_batch(device, refresh=False)
        else:
            images = source.next_batch(device)
        out = raw_model(images)
        losses.append(float(out["loss"].detach().cpu()))
    raw_model.train()
    return float(sum(losses) / max(1, len(losses)))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    seed = int(config.get("seed", 0))
    seed_everything(seed)

    rank, world_size, local_rank = init_distributed_if_needed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    run_id = str(config.get("run_id", Path(args.config).stem))
    output_dir = ensure_dir(config.get("output_dir", f"results/runs/{run_id}"))
    train_cfg = config.setdefault("train", {})
    apply_target_instances(train_cfg, world_size)

    if is_main_process():
        save_config(config, output_dir / "config.yaml")
        append_jsonl(output_dir / "metadata.jsonl", {"run_id": run_id, "git_commit": git_commit_hash()})

    model = build_mae(config).to(device)
    if world_size > 1:
        model = DistributedDataParallel(model, device_ids=[local_rank] if torch.cuda.is_available() else None)

    source = build_source(
        config,
        image_size=int(config.get("model", {}).get("image_size", DEFAULT_IMAGE_SIZE)),
        batch_size=int(train_cfg.get("batch_size", DEFAULT_TRAIN_BATCH_SIZE)),
        seed=seed + rank,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", DEFAULT_TRAIN_LR)),
        weight_decay=float(train_cfg.get("weight_decay", DEFAULT_TRAIN_WEIGHT_DECAY)),
        betas=tuple(train_cfg.get("betas", DEFAULT_TRAIN_BETAS)),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available() and bool(train_cfg.get("amp", True)))
    start_step = load_checkpoint(args.resume, model, optimizer, scaler, source=source, device=device) if args.resume else 0
    if start_step > 0:
        setter = getattr(source, "set_progress", None)
        if callable(setter):
            setter(start_step, images_seen_for_step(start_step, train_cfg, world_size))

    writer = SummaryWriter(str(output_dir / "tensorboard")) if is_main_process() and SummaryWriter else None
    wandb_run = init_wandb_if_enabled(config, output_dir, run_id) if is_main_process() else None
    metrics_path = output_dir / "metrics.jsonl"
    steps = int(train_cfg.get("steps", DEFAULT_TRAIN_STEPS))
    log_every = int(train_cfg.get("log_every", DEFAULT_TRAIN_LOG_EVERY))
    val_every = int(train_cfg.get("val_every", DEFAULT_TRAIN_VAL_EVERY))
    validation_interval = int(train_cfg.get("validation_interval_instances", DEFAULT_TRAIN_VALIDATION_INTERVAL_INSTANCES))
    max_validation_target = int(train_cfg.get("target_instances") or train_cfg.get("effective_instances") or 0)
    next_validation_images = next_validation_target(images_seen_for_step(start_step, train_cfg, world_size), validation_interval)
    checkpoint_every = int(train_cfg.get("checkpoint_every", DEFAULT_TRAIN_CHECKPOINT_EVERY))
    preview_every = int(train_cfg.get("preview_every", DEFAULT_TRAIN_PREVIEW_EVERY))
    val_batches = int(train_cfg.get("val_batches", DEFAULT_TRAIN_VAL_BATCHES))
    clip_grad = float(train_cfg.get("clip_grad", DEFAULT_TRAIN_CLIP_GRAD))
    checkpoint_source_state = bool(train_cfg.get("checkpoint_source_state", False))
    checkpoint_source_state_in_steps = bool(train_cfg.get("checkpoint_source_state_in_step_checkpoints", False))

    def source_checkpoint_state(include: bool) -> dict[str, Any] | None:
        if not include:
            return None
        state_fn = getattr(source, "state_dict", None)
        if not callable(state_fn):
            return None
        return state_fn()

    model.train()
    last_time = time.perf_counter()
    last_log_step = start_step
    for step in range(start_step, steps):
        lr = learning_rate(step, train_cfg)
        set_optimizer_lr(optimizer, lr)
        progress_setter = getattr(source, "set_progress", None)
        if callable(progress_setter):
            progress_setter(step, images_seen_for_step(step, train_cfg, world_size))
        images = source.next_batch(device)
        optimizer.zero_grad(set_to_none=True)

        if is_main_process() and step == start_step:
            with torch.no_grad():
                initial_images_seen = images_seen_for_step(start_step, train_cfg, world_size)
                preview_path = output_dir / f"pretrain_source_step_{start_step:07d}.png"
                save_pretrain_source_preview(images, preview_path, run_id)
                wandb_image(
                    wandb_run,
                    preview_path,
                    "images/buffer_images",
                    preview_caption(run_id, initial_images_seen),
                    initial_images_seen,
                )

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available() and bool(train_cfg.get("amp", True))):
            out = model(images)
            loss = out["loss"]

        scaler.scale(loss).backward()
        if clip_grad > 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        else:
            grad_norm = torch.tensor(0.0, device=device)
        scaler.step(optimizer)
        scaler.update()

        if is_main_process() and ((step + 1) % log_every == 0 or step == 0):
            now = time.perf_counter()
            elapsed = now - last_time
            last_time = now
            updates_since_log = max(1, step + 1 - last_log_step)
            last_log_step = step + 1
            images_seen = images_seen_for_step(step + 1, train_cfg, world_size)
            throughput = (int(train_cfg.get("batch_size", DEFAULT_TRAIN_BATCH_SIZE)) * world_size * updates_since_log) / max(elapsed, 1e-6)
            row = {
                "step": step + 1,
                "images_seen": images_seen,
                "loss": float(loss.detach().cpu()),
                "lr": lr,
                "grad_norm": float(grad_norm.detach().cpu()),
                "images_per_second": throughput,
            }
            append_jsonl(metrics_path, row)
            if writer:
                for key, value in row.items():
                    if key != "step":
                        writer.add_scalar(key, value, step + 1)
            wandb_log(
                wandb_run,
                {
                    "train/loss": row["loss"],
                    "train/learning-rate": row["lr"],
                    "train/gradient_norm": row["grad_norm"],
                    "train/images_per_second": row["images_per_second"],
                },
                images_seen,
            )

        images_seen = images_seen_for_step(step + 1, train_cfg, world_size)
        should_validate = False
        validation_target = images_seen
        if validation_interval > 0 and next_validation_images > 0:
            target_is_in_run = max_validation_target <= 0 or next_validation_images <= max_validation_target
            should_validate = target_is_in_run and images_seen >= next_validation_images
            if should_validate:
                validation_target = next_validation_images
                while next_validation_images <= images_seen:
                    next_validation_images += validation_interval
        elif val_every > 0:
            should_validate = (step + 1) % val_every == 0

        if is_main_process() and should_validate:
            val_loss = validation_loss(model, source, device, val_batches)
            append_jsonl(
                metrics_path,
                {
                    "step": step + 1,
                    "images_seen": images_seen,
                    "validation_target_images": validation_target,
                    "val_loss": val_loss,
                },
            )
            if writer:
                writer.add_scalar("val_loss", val_loss, step + 1)
            wandb_log(
                wandb_run,
                {
                    "validation/loss": val_loss,
                },
                images_seen,
            )

        should_log_preview = should_validate or (preview_every > 0 and (step + 1) % preview_every == 0)
        if is_main_process() and should_log_preview:
            preview_images_seen = images_seen_for_step(step + 1, train_cfg, world_size)
            source_path = output_dir / f"pretrain_source_step_{step + 1:07d}.png"
            save_pretrain_source_preview(images, source_path, run_id)
            wandb_image(
                wandb_run,
                source_path,
                "images/buffer_images",
                preview_caption(run_id, preview_images_seen),
                preview_images_seen,
            )
            recon_path = output_dir / f"reconstruction_step_{step + 1:07d}.png"
            save_reconstruction_preview(model, images, recon_path, run_id)
            wandb_image(
                wandb_run,
                recon_path,
                "images/reconstruction",
                f"Original input | masked input | MAE reconstruction | {preview_caption(run_id, preview_images_seen)}",
                preview_images_seen,
            )

        if is_main_process() and checkpoint_every > 0 and (step + 1) % checkpoint_every == 0:
            save_checkpoint(
                output_dir / f"checkpoint_step_{step + 1:07d}.pt",
                model,
                optimizer,
                scaler,
                step + 1,
                config,
                source_checkpoint_state(checkpoint_source_state and checkpoint_source_state_in_steps),
            )

    if is_main_process():
        save_checkpoint(
            output_dir / "checkpoint_final.pt",
            model,
            optimizer,
            scaler,
            steps,
            config,
            source_checkpoint_state(checkpoint_source_state),
        )
        if writer:
            writer.close()
        if wandb_run is not None:
            wandb_run.finish()
    barrier()


if __name__ == "__main__":
    main()
