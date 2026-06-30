from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import torch

from src.irc_vit.config import (
    DEFAULT_EVAL_MERGED_OUTPUT_CSV,
    DEFAULT_OUTPUT_DIR,
    load_config,
    validate_eval_embedding_modes,
)
from src.irc_vit.evaluate import (
    RESULT_COLUMNS,
    checkpoint_images_seen,
    init_wandb_if_enabled,
    wandb_dataset_role,
    wandb_eval_dataset_key,
    wandb_eval_log,
    wandb_eval_mean_keys,
)
from src.irc_vit.utils import append_csv, ensure_dir


def _argv_without_sbatch_go(argv: list[str] | None) -> list[str] | None:
    if argv and argv[0] == "go":
        return argv[1:]
    return argv


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge eval CSV shards and log them to W&B")
    parser.add_argument("--config", required=True)
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", default=DEFAULT_EVAL_MERGED_OUTPUT_CSV)
    parser.add_argument("--disable-wandb", action="store_true")
    return parser.parse_args(_argv_without_sbatch_go(argv))


def read_rows(paths: list[str]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for raw_path in paths:
        path = Path(raw_path)
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows.extend(dict(row) for row in reader)
    return rows


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    if math.isnan(result):
        return None
    return result


def _checkpoint_images_seen(checkpoint: str, fallback_config: dict[str, Any], fallback_step: int | None) -> int:
    if checkpoint.lower() in {"random_init", "none", "no_pretraining"}:
        return 0
    path = Path(checkpoint)
    if path.exists():
        payload = torch.load(path, map_location="cpu")
        return checkpoint_images_seen(payload.get("config", fallback_config), int(payload.get("step", fallback_step or 0)))
    return checkpoint_images_seen(fallback_config, fallback_step)


def row_images_seen(row: dict[str, str], config: dict[str, Any], cache: dict[str, int]) -> int:
    explicit = row.get("images_seen", "")
    if explicit:
        return int(float(explicit))

    checkpoint = row.get("checkpoint", "")
    if checkpoint in cache:
        return cache[checkpoint]

    raw_step = row.get("pretrain_steps", "")
    try:
        checkpoint_step = int(float(raw_step)) if raw_step != "" else None
    except ValueError:
        checkpoint_step = None

    images_seen = _checkpoint_images_seen(checkpoint, config, checkpoint_step)
    cache[checkpoint] = images_seen
    return images_seen


def validate_row_embedding_mode(row: dict[str, str]) -> None:
    probe = row.get("probe_type", "")
    if probe not in {"linear", "linear_top5", "linear_loss"}:
        return
    validate_eval_embedding_modes(row.get("embedding_mode", ""))


def normalize_rows(rows: list[dict[str, str]], config: dict[str, Any]) -> list[dict[str, str]]:
    cache: dict[str, int] = {}
    normalized = []
    for row in rows:
        row = dict(row)
        validate_row_embedding_mode(row)
        row["images_seen"] = str(row_images_seen(row, config, cache))
        normalized.append(row)
    deduped: dict[tuple[str, str, str, str, str], dict[str, str]] = {}
    for row in normalized:
        key = (
            row.get("images_seen", ""),
            row.get("checkpoint", ""),
            row.get("eval_dataset", ""),
            row.get("embedding_mode", ""),
            row.get("probe_type", ""),
        )
        deduped[key] = row
    return sorted(
        deduped.values(),
        key=lambda r: (
            int(float(r.get("images_seen") or 0)),
            r.get("eval_dataset", ""),
            r.get("embedding_mode", ""),
            r.get("probe_type", ""),
        ),
    )


def write_merged_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if path.exists():
        path.unlink()
    for row in rows:
        append_csv(path, row, RESULT_COLUMNS)


def wandb_payloads(rows: list[dict[str, str]]) -> dict[int, dict[str, float]]:
    by_step: dict[int, dict[str, float]] = {}
    mean_values: dict[tuple[int, str, str, str], list[float]] = {}

    for row in rows:
        value = _as_float(row.get("accuracy"))
        if value is None:
            continue
        images_seen = int(float(row.get("images_seen") or 0))
        mode = row.get("embedding_mode", "")
        probe = row.get("probe_type", "")
        dataset = row.get("eval_dataset", "")
        role = wandb_dataset_role(dataset)
        key = wandb_eval_dataset_key(dataset, mode, probe)
        if key is None:
            continue
        by_step.setdefault(images_seen, {})[key] = value
        mean_values.setdefault((images_seen, role, mode, probe), []).append(value)

    for (images_seen, role, mode, probe), values in mean_values.items():
        if not values:
            continue
        mean_value = sum(values) / len(values)
        for mean_key in wandb_eval_mean_keys(mode, probe, role):
            by_step.setdefault(images_seen, {})[mean_key] = mean_value
    return by_step


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config)
    if args.disable_wandb:
        config.setdefault("wandb", {})["enabled"] = False

    output_dir = ensure_dir(config.get("output_dir", DEFAULT_OUTPUT_DIR))
    rows = normalize_rows(read_rows(args.inputs), config)
    out_path = Path(args.out)
    write_merged_csv(out_path, rows)

    run_id = str(config.get("run_id", Path(args.config).stem))
    wandb_run = init_wandb_if_enabled(config, Path(output_dir), run_id)
    for images_seen, payload in sorted(wandb_payloads(rows).items()):
        wandb_eval_log(wandb_run, payload, images_seen)
    if wandb_run is not None:
        wandb_run.finish()
    print(f"merged {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
