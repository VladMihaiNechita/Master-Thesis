from __future__ import annotations

import pytest

from src.irc_vit.log_eval_csv_to_wandb import normalize_rows, wandb_payloads


def test_normalize_rows_fills_images_seen_and_deduplicates() -> None:
    config = {"train": {"steps": 10, "target_instances": 1000, "batch_size": 100}}
    rows = [
        {
            "checkpoint": "missing_checkpoint.pt",
            "pretrain_steps": "2",
            "eval_dataset": "cifar10",
            "embedding_mode": "cls_mean",
            "probe_type": "linear",
            "accuracy": "0.1",
        },
        {
            "checkpoint": "missing_checkpoint.pt",
            "pretrain_steps": "2",
            "eval_dataset": "cifar10",
            "embedding_mode": "cls_mean",
            "probe_type": "linear",
            "accuracy": "0.2",
        },
    ]

    normalized = normalize_rows(rows, config)

    assert len(normalized) == 1
    assert normalized[0]["images_seen"] == "200"
    assert normalized[0]["accuracy"] == "0.2"


def test_wandb_payloads_uses_eval_metric_names_and_means() -> None:
    rows = [
        {
            "images_seen": "100",
            "eval_dataset": "cifar10",
            "embedding_mode": "cls_mean",
            "probe_type": "linear",
            "accuracy": "0.5",
        },
        {
            "images_seen": "100",
            "eval_dataset": "cifar10",
            "embedding_mode": "cls_mean",
            "probe_type": "linear_top5",
            "accuracy": "0.9",
        },
        {
            "images_seen": "100",
            "eval_dataset": "cifar100",
            "embedding_mode": "cls_mean",
            "probe_type": "linear",
            "accuracy": "0.25",
        },
        {
            "images_seen": "100",
            "eval_dataset": "cifar10",
            "embedding_mode": "reconstruction",
            "probe_type": "raw_pixel_mse",
            "accuracy": "0.1",
        },
        {
            "images_seen": "100",
            "eval_dataset": "cifar100",
            "embedding_mode": "reconstruction",
            "probe_type": "raw_pixel_mse",
            "accuracy": "0.3",
        },
        {
            "images_seen": "100",
            "eval_dataset": "cifar10",
            "embedding_mode": "reconstruction",
            "probe_type": "raw_pixel_mae",
            "accuracy": "0.05",
        },
        {
            "images_seen": "100",
            "eval_dataset": "cifar10",
            "embedding_mode": "reconstruction",
            "probe_type": "patch_normalized_mse",
            "accuracy": "0.4",
        },
        {
            "images_seen": "100",
            "eval_dataset": "cifar10",
            "embedding_mode": "reconstruction",
            "probe_type": "patch_normalized_mae",
            "accuracy": "0.2",
        },
        {
            "images_seen": "100",
            "eval_dataset": "cifar10",
            "embedding_mode": "cls_mean",
            "probe_type": "linear_loss",
            "accuracy": "1.0",
        },
        {
            "images_seen": "100",
            "eval_dataset": "cifar100",
            "embedding_mode": "cls_mean",
            "probe_type": "linear_loss",
            "accuracy": "2.0",
        },
    ]

    payload = wandb_payloads(rows)[100]

    assert payload["validation/cifar10_top-1_accuracy"] == 0.5
    assert payload["validation/cifar10_top-5_accuracy"] == 0.9
    assert payload["test/cifar100_top-1_accuracy"] == 0.25
    assert payload["validation_mean/top-1_accuracy"] == 0.5
    assert payload["validation_mean/top-5_accuracy"] == 0.9
    assert payload["validation_mean/cross-entropy_loss"] == 1.0
    assert payload["validation/cifar10_raw_pixel_MSE"] == 0.1
    assert payload["test/cifar100_raw_pixel_MSE"] == 0.3
    assert payload["validation/cifar10_raw_pixel_MAE"] == 0.05
    assert payload["validation/cifar10_patch-normalized_MSE"] == 0.4
    assert payload["validation/cifar10_patch-normalized_MAE"] == 0.2
    assert payload["validation_mean/raw_pixel_MSE"] == 0.1
    assert payload["validation_mean/raw_pixel_MAE"] == 0.05
    assert payload["validation_mean/patch-normalized_MSE"] == 0.4
    assert payload["validation_mean/patch-normalized_MAE"] == 0.2
    assert "validation_mean/test_top-1_accuracy" not in payload
    assert "means/eval_cls_mean_linear_accuracy" not in payload


def test_normalize_rows_rejects_stale_linear_embedding_modes() -> None:
    rows = [
        {
            "checkpoint": "missing_checkpoint.pt",
            "pretrain_steps": "2",
            "eval_dataset": "cifar10",
            "embedding_mode": "cls",
            "probe_type": "linear",
            "accuracy": "0.1",
        },
    ]

    with pytest.raises(ValueError, match="Unsupported eval embedding"):
        normalize_rows(rows, {"train": {"steps": 10, "target_instances": 1000}})
