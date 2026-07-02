from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from src.irc_vit.config import (
    DEFAULT_EVAL_EMBEDDING_MODES,
    DEFAULT_EVAL_FEATURE_OUTPUT_CSV,
    DEFAULT_EVAL_PROBES,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_OUTPUT_DIR,
    eval_perceptron_training_config,
    load_config,
    validate_eval_embedding_modes,
    validate_eval_probes,
)
from src.irc_vit.data import cache_dataset_name
from src.irc_vit.evaluate import (
    append_result,
    checkpoint_images_seen,
    checkpoint_metadata,
    dataset_spec,
    init_wandb_if_enabled,
    linear_probe_head_save_path,
    linear_probe_metrics,
    resolve_probe_device,
    wandb_dataset_role,
    wandb_eval_dataset_key,
    wandb_eval_log,
    wandb_eval_mean_keys,
)
from src.irc_vit.feature_cache import feature_split_path, load_feature_split, read_feature_cache_metadata
from src.irc_vit.synthetic_datasets import dataset_filter_key
from src.irc_vit.utils import ensure_dir


def _argv_without_sbatch_go(argv: list[str] | None) -> list[str] | None:
    if argv and argv[0] == "go":
        return argv[1:]
    return argv


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen linear probes from cached features")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--features", required=True, help="Feature-cache directory produced by extract_feature_cache")
    parser.add_argument("--out", default=DEFAULT_EVAL_FEATURE_OUTPUT_CSV)
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional dataset-name filter")
    parser.add_argument("--embedding-modes", nargs="*", default=None, help="Override eval.embedding_modes")
    parser.add_argument("--probes", nargs="*", default=None, help="Override eval.probes")
    parser.add_argument("--linear-device", default="config", choices=["config", "auto", "cpu", "cuda"], help="Device for the PyTorch linear probe")
    parser.add_argument("--overwrite", action="store_true", help="Remove --out before writing")
    parser.add_argument("--disable-wandb", action="store_true")
    return parser.parse_args(_argv_without_sbatch_go(argv))


def selected_dataset(raw_dataset: dict[str, Any], selected: set[str] | None) -> bool:
    if selected is None:
        return True
    name = str(raw_dataset.get("name", ""))
    return dataset_filter_key(name) in selected or dataset_filter_key(cache_dataset_name(name)) in selected


def checkpoint_config_and_step(checkpoint: str, fallback_config: dict[str, Any]) -> tuple[dict[str, Any], int]:
    if str(checkpoint).lower() in {"random_init", "none", "no_pretraining"}:
        return {
            "run_id": "random_init",
            "source": {"type": "none"},
            "model": fallback_config.get("model", {}),
            "train": {"steps": 0},
            "seed": fallback_config.get("seed", 0),
        }, 0
    payload = torch.load(checkpoint, map_location="cpu")
    return payload["config"], int(payload.get("step", 0))


def cached_note(train_meta: dict[str, Any], eval_meta: dict[str, Any], features_root: Path) -> str:
    train_note = str(train_meta.get("note", ""))
    eval_note = str(eval_meta.get("note", ""))
    note = eval_note or train_note or "ok"
    return f"feature_cache:{features_root}; {note}"


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.disable_wandb:
        cfg.setdefault("wandb", {})["enabled"] = False

    features_root = Path(args.features)
    cache_metadata = read_feature_cache_metadata(features_root)
    cached_checkpoint = str(cache_metadata.get("checkpoint", ""))
    if cached_checkpoint and cached_checkpoint != str(args.checkpoint):
        try:
            same = Path(cached_checkpoint).resolve() == Path(args.checkpoint).resolve()
        except Exception:
            same = False
        if not same:
            print(f"warning: feature cache was extracted from {cached_checkpoint}, evaluating as {args.checkpoint}")

    ckpt_config, checkpoint_step = checkpoint_config_and_step(args.checkpoint, cfg)
    cached_step = cache_metadata.get("checkpoint_step")
    if cached_step is not None and int(cached_step) != checkpoint_step:
        print(f"warning: feature cache training step ({cached_step}) does not match checkpoint step ({checkpoint_step})")

    meta = checkpoint_metadata(args.checkpoint, ckpt_config, checkpoint_step)
    images_seen = checkpoint_images_seen(ckpt_config, checkpoint_step)
    meta["images_seen"] = images_seen

    output_dir = ensure_dir(cfg.get("output_dir", DEFAULT_OUTPUT_DIR))
    wandb_run = init_wandb_if_enabled(cfg, Path(output_dir), str(meta["run_id"]))
    wandb_values: dict[tuple[str, str, str], list[float]] = {}

    eval_cfg = cfg.get("eval", {})
    linear_cfg = eval_perceptron_training_config(eval_cfg, ckpt_config.get("model", cfg.get("model", {})))
    image_size = int(eval_cfg.get("image_size", ckpt_config.get("model", {}).get("image_size", DEFAULT_IMAGE_SIZE)))
    embedding_modes = validate_eval_embedding_modes(
        args.embedding_modes if args.embedding_modes is not None else eval_cfg.get("embedding_modes", DEFAULT_EVAL_EMBEDDING_MODES)
    )
    probes = validate_eval_probes(args.probes or eval_cfg.get("probes", DEFAULT_EVAL_PROBES))
    selected = (
        {dataset_filter_key(name) for name in args.datasets}
        | {dataset_filter_key(cache_dataset_name(name)) for name in args.datasets}
        if args.datasets else None
    )
    out_csv = Path(args.out)
    if args.overwrite and out_csv.exists():
        out_csv.unlink()
    linear_device_name = args.linear_device
    if linear_device_name == "config":
        linear_device_name = str(linear_cfg["device"])
    linear_device: torch.device | None = None

    for raw_dataset in cfg.get("datasets", []):
        if not selected_dataset(raw_dataset, selected):
            continue
        spec = dataset_spec(raw_dataset, image_size)
        for mode in embedding_modes:
            train_path = feature_split_path(features_root, raw_dataset, "train", mode)
            eval_path = feature_split_path(features_root, raw_dataset, "eval", mode)
            try:
                train_features, train_labels, train_meta = load_feature_split(train_path)
                eval_features, eval_labels, eval_meta = load_feature_split(eval_path)
            except Exception as exc:
                append_result(out_csv, meta, spec.name, mode, "skipped", None, f"{type(exc).__name__}: {exc}")
                continue

            note = cached_note(train_meta, eval_meta, features_root)
            for probe in probes:
                try:
                    if linear_device is None:
                        linear_device = resolve_probe_device(linear_device_name)
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
                        device=linear_device,
                        seed=int(cfg.get("seed", ckpt_config.get("seed", 0))),
                        save_head_path=head_path,
                        head_metadata={
                            "dataset": spec.name,
                            "embedding_mode": mode,
                            "probe": probe,
                            "checkpoint": str(args.checkpoint),
                            "features": str(features_root),
                            "images_seen": int(images_seen),
                            "seed": int(cfg.get("seed", ckpt_config.get("seed", 0))),
                            "linear": dict(linear_cfg),
                        },
                    )
                    value = metrics["top1_accuracy"]
                    top5 = metrics["top5_accuracy"]
                    linear_loss = metrics["loss"]
                    role = wandb_dataset_role(spec.name)

                    append_result(out_csv, meta, spec.name, mode, probe, value, note)
                    key = wandb_eval_dataset_key(spec.name, mode, probe)
                    if key is not None:
                        wandb_eval_log(wandb_run, {key: value}, images_seen)
                        wandb_values.setdefault((role, mode, probe), []).append(float(value))

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
    print(f"evaluated cached features -> {out_csv}")


if __name__ == "__main__":
    main()
