from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from src.irc_vit.config import (
    DEFAULT_EVAL_EMBEDDING_MODES,
    DEFAULT_IMAGE_SIZE,
    load_config,
    validate_eval_embedding_modes,
)
from src.irc_vit.data import build_dataset_pair, cache_dataset_name
from src.irc_vit.evaluate import (
    RECONSTRUCTION_PROBES,
    append_result,
    checkpoint_images_seen,
    checkpoint_metadata,
    dataset_eval_loader_settings,
    dataset_spec,
    extract_embeddings,
    load_mae_from_checkpoint,
    reconstruction_metrics,
)
from src.irc_vit.feature_cache import (
    feature_cache_metadata_path,
    feature_dtype,
    feature_split_path,
    save_feature_split,
    write_feature_cache_metadata,
)
from src.irc_vit.model import build_mae
from src.irc_vit.synthetic_datasets import dataset_filter_key


def _argv_without_sbatch_go(argv: list[str] | None) -> list[str] | None:
    if argv and argv[0] == "go":
        return argv[1:]
    return argv


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frozen ViT features into reusable cache files")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True, help="Feature-cache output directory")
    parser.add_argument("--reconstruction-out", default="", help="Optional CSV for GPU reconstruction metrics")
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional dataset-name filter")
    parser.add_argument("--embedding-modes", nargs="*", default=None, help="Override eval.embedding_modes")
    parser.add_argument("--splits", nargs="+", choices=["train", "eval"], default=["train", "eval"])
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--disable-reconstruction", action="store_true")
    return parser.parse_args(_argv_without_sbatch_go(argv))


def selected_dataset(raw_dataset: dict[str, Any], selected: set[str] | None) -> bool:
    if selected is None:
        return True
    name = str(raw_dataset.get("name", ""))
    return dataset_filter_key(name) in selected or dataset_filter_key(cache_dataset_name(name)) in selected


def load_model_for_features(checkpoint: str, cfg: dict[str, Any], device: torch.device):
    if str(checkpoint).lower() in {"random_init", "none", "no_pretraining"}:
        model = build_mae(cfg).to(device).eval()
        ckpt_config = {
            "run_id": "random_init",
            "source": {"type": "none"},
            "model": cfg.get("model", {}),
            "train": {"steps": 0},
            "seed": cfg.get("seed", 0),
        }
        return model, ckpt_config, 0
    return load_mae_from_checkpoint(checkpoint, device)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_config(args.config)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt_config, checkpoint_step = load_model_for_features(args.checkpoint, cfg, device)
    encoder = model.encoder
    meta = checkpoint_metadata(args.checkpoint, ckpt_config, checkpoint_step)
    meta["images_seen"] = checkpoint_images_seen(ckpt_config, checkpoint_step)

    eval_cfg = cfg.get("eval", {})
    image_size = int(eval_cfg.get("image_size", ckpt_config.get("model", {}).get("image_size", DEFAULT_IMAGE_SIZE)))
    embedding_modes = validate_eval_embedding_modes(
        args.embedding_modes if args.embedding_modes is not None else eval_cfg.get("embedding_modes", DEFAULT_EVAL_EMBEDDING_MODES)
    )
    seed = int(cfg.get("seed", ckpt_config.get("seed", 0)))
    selected = (
        {dataset_filter_key(name) for name in args.datasets}
        | {dataset_filter_key(cache_dataset_name(name)) for name in args.datasets}
        if args.datasets else None
    )
    dtype = feature_dtype(args.dtype)
    reconstruction_out = Path(args.reconstruction_out) if args.reconstruction_out else None

    root_metadata = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint_step,
        "images_seen": meta["images_seen"],
        "config": str(args.config),
        "embedding_modes": embedding_modes,
        "feature_dtype": args.dtype,
        "device": str(device),
    }
    write_feature_cache_metadata(out_root, root_metadata)

    written: list[str] = []
    skipped: list[str] = []
    touched_paths: set[Path] = set()

    for raw_dataset in cfg.get("datasets", []):
        if not selected_dataset(raw_dataset, selected):
            continue
        spec = dataset_spec(raw_dataset, image_size)
        batch_size, num_workers = dataset_eval_loader_settings(raw_dataset, eval_cfg)
        try:
            train_ds, eval_ds, note = build_dataset_pair(spec, seed=seed)
        except Exception as exc:
            if reconstruction_out is not None:
                append_result(reconstruction_out, meta, spec.name, "", "skipped", None, f"{type(exc).__name__}: {exc}")
            skipped.append(f"{spec.name}: {type(exc).__name__}: {exc}")
            continue

        if reconstruction_out is not None and bool(eval_cfg.get("reconstruction_mae", False)) and not args.disable_reconstruction:
            try:
                recon = reconstruction_metrics(model, eval_ds, batch_size, device, num_workers, seed)
                for probe in RECONSTRUCTION_PROBES:
                    append_result(reconstruction_out, meta, spec.name, "reconstruction", probe, recon[probe], note)
            except Exception as exc:
                append_result(reconstruction_out, meta, spec.name, "reconstruction", "raw_pixel_mae", None, f"{type(exc).__name__}: {exc}")

        split_datasets = {"train": train_ds, "eval": eval_ds}
        for mode in embedding_modes:
            for split_role in args.splits:
                path = feature_split_path(out_root, raw_dataset, split_role, mode)
                if path in touched_paths:
                    skipped.append(f"duplicate:{path}")
                    continue
                if path.exists() and not args.overwrite:
                    skipped.append(str(path))
                    continue
                dataset = split_datasets[split_role]
                features, labels = extract_embeddings(
                    encoder,
                    dataset,
                    batch_size,
                    device,
                    mode,
                    num_workers,
                    desc=f"{spec.name} {split_role} {mode}",
                )
                split_metadata = {
                    "dataset": spec.name,
                    "split_role": split_role,
                    "embedding_mode": mode,
                    "num_examples": int(labels.numel()),
                    "feature_dim": int(features.shape[1]) if features.ndim == 2 else None,
                    "note": note,
                }
                save_feature_split(path, features, labels, split_metadata, dtype=dtype)
                touched_paths.add(path)
                written.append(str(path))

    summary_path = out_root / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "metadata": root_metadata,
                "metadata_path": str(feature_cache_metadata_path(out_root)),
                "written": written,
                "skipped": skipped,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"feature cache written: {len(written)} split files -> {out_root}")
    if skipped:
        print(f"feature cache skipped: {len(skipped)} existing or failed items")


if __name__ == "__main__":
    main()
