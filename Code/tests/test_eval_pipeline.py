from __future__ import annotations

import csv

import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

import pytest

from src.irc_vit.config import (
    DEFAULT_EVAL_PROBES,
    eval_perceptron_training_config,
    validate_eval_embedding_modes,
    validate_eval_probes,
)
from src.irc_vit.extract_feature_cache import main as extract_feature_cache_main
from src.irc_vit.evaluate import (
    RECONSTRUCTION_PROBES,
    dataset_eval_loader_settings,
    dataset_spec,
    extract_embeddings,
    linear_probe_accuracy,
    linear_probe_metrics,
    reconstruction_metrics,
)
from src.irc_vit.evaluate_feature_cache import main as evaluate_feature_cache_main
from src.irc_vit.feature_cache import feature_split_path
from src.irc_vit.model import build_mae
from src.irc_vit.submit_slurm_pipeline import main as submit_slurm_pipeline_main


def test_extract_and_linear_probe_on_fake_dataset() -> None:
    ds_train = datasets.FakeData(size=24, image_size=(3, 64, 64), num_classes=3, transform=transforms.ToTensor())
    ds_eval = datasets.FakeData(size=12, image_size=(3, 64, 64), num_classes=3, transform=transforms.ToTensor())
    model = build_mae({
        "model": {"image_size": 64, "patch_size": 16, "embed_dim": 48, "depth": 1, "num_heads": 3},
        "mae": {"decoder_embed_dim": 48, "decoder_depth": 1, "decoder_num_heads": 3},
    })
    train_x, train_y = extract_embeddings(model.encoder, ds_train, 8, torch.device("cpu"), "cls_mean", num_workers=0)
    eval_x, eval_y = extract_embeddings(model.encoder, ds_eval, 8, torch.device("cpu"), "cls_mean", num_workers=0)

    assert train_x.shape[0] == len(ds_train)
    assert eval_x.shape[0] == len(ds_eval)
    accuracy = linear_probe_accuracy(train_x, train_y, eval_x, eval_y, epochs=3, batch_size=8)
    assert 0.0 <= accuracy <= 1.0
    metrics = linear_probe_metrics(train_x, train_y, eval_x, eval_y, epochs=3, batch_size=8)
    assert 0.0 <= metrics["top1_accuracy"] <= 1.0
    assert 0.0 <= metrics["top5_accuracy"] <= 1.0
    assert metrics["top5_accuracy"] >= metrics["top1_accuracy"]


def test_linear_probe_head_save_folds_feature_standardization(tmp_path) -> None:
    train_x = torch.tensor(
        [
            [0.0, 1.0],
            [0.2, 0.8],
            [3.0, 4.0],
            [3.2, 4.2],
        ]
    )
    train_y = torch.tensor([0, 0, 1, 1])
    eval_x = torch.tensor([[0.1, 0.9], [3.1, 4.1]])
    eval_y = torch.tensor([0, 1])
    head_path = tmp_path / "probe_head.pt"

    linear_probe_metrics(
        train_x,
        train_y,
        eval_x,
        eval_y,
        epochs=3,
        batch_size=2,
        save_head_path=head_path,
        head_metadata={"dataset": "fake", "embedding_mode": "cls_mean"},
    )

    payload = torch.load(head_path, map_location="cpu", weights_only=False)
    normalized = (eval_x - payload["feature_mean"]) / payload["feature_std"]
    normalized_logits = F.linear(
        normalized,
        payload["normalized_head"]["weight"],
        payload["normalized_head"]["bias"],
    )
    raw_logits = F.linear(eval_x, payload["head"]["weight"], payload["head"]["bias"])

    assert payload["metadata"]["embedding_mode"] == "cls_mean"
    assert torch.allclose(raw_logits, normalized_logits, atol=1e-5)


def test_dataset_eval_loader_settings_can_override_global_defaults() -> None:
    raw = {"name": "fake", "eval_batch_size": 17, "eval_num_workers": 0}
    spec = dataset_spec(raw, image_size=64)

    assert spec.eval_batch_size == 17
    assert dataset_eval_loader_settings(raw, {"batch_size": 512, "num_workers": 4}) == (17, 0)
    assert dataset_eval_loader_settings({"name": "fake"}, {"batch_size": 512, "num_workers": 4}) == (512, 4)


def test_reconstruction_metrics_report_raw_and_patch_normalized_values() -> None:
    ds = datasets.FakeData(size=4, image_size=(3, 64, 64), num_classes=2, transform=transforms.ToTensor())
    model = build_mae({
        "model": {"image_size": 64, "patch_size": 16, "embed_dim": 48, "depth": 1, "num_heads": 3},
        "mae": {"decoder_embed_dim": 48, "decoder_depth": 1, "decoder_num_heads": 3},
    })

    metrics = reconstruction_metrics(model, ds, batch_size=2, device=torch.device("cpu"), num_workers=0, seed=0)

    assert set(metrics) == set(RECONSTRUCTION_PROBES)
    assert all(value >= 0.0 for value in metrics.values())


def test_default_probe_is_linear_only_in_pipeline(tmp_path) -> None:
    assert DEFAULT_EVAL_PROBES == ("linear",)
    assert validate_eval_probes([]) == []
    assert validate_eval_embedding_modes(None) == ["cls_mean"]
    with pytest.raises(ValueError, match="Unsupported eval embedding"):
        validate_eval_embedding_modes(["cls"])

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
run_id: pipeline_default_probe_smoke
output_dir: "{output_dir}"
seed: 1729
wandb:
  enabled: false
model:
  name: vit_tiny_patch16_224
  image_size: 224
  patch_size: 16
eval:
  image_size: 224
  embedding_modes: [cls_mean]
  reconstruction_mae: false
datasets:
  - name: imagenet1k
""".format(output_dir=(tmp_path / "train").as_posix()),
        encoding="utf-8",
    )
    work_dir = tmp_path / "pipeline"

    submit_slurm_pipeline_main([
        "--config", str(config_path),
        "--checkpoint", "random_init",
        "--work-dir", str(work_dir),
        "--split-eval-by-dataset",
        "--disable-wandb-merge",
    ])

    slurm_dir = work_dir / "slurm"
    assert (slurm_dir / "03_cpu_random_init_imagenet1k.sbatch").exists()
    assert not list(slurm_dir.glob("04_*.sbatch"))


def test_perceptron_training_config_uses_preset_and_overrides() -> None:
    defaults = eval_perceptron_training_config({}, {"name": "vit_small_patch16_224"})
    assert defaults["epochs"] == 30
    assert defaults["batch_size"] == 8192
    assert defaults["lr"] == 0.1
    assert defaults["optimizer"] == "torch.optim.SGD"

    resolved = eval_perceptron_training_config(
        {
            "perceptron": {"epochs": 5, "batch_size": 16, "device": "cpu"},
            "linear_lr": 0.02,
            "linear_weight_decay": 0.03,
        },
        {
            "name": "vit_small_patch16_224",
            "perceptron": {"momentum": 0.7},
        },
    )

    assert resolved["epochs"] == 5
    assert resolved["batch_size"] == 16
    assert resolved["lr"] == 0.02
    assert resolved["weight_decay"] == 0.03
    assert resolved["momentum"] == 0.7
    assert resolved["device"] == "cpu"


def test_slurm_pipeline_rejects_non_cls_mean_config_embedding_modes(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
run_id: pipeline_bad_embedding_mode_smoke
output_dir: "{output_dir}"
seed: 1729
wandb:
  enabled: false
model:
  name: vit_tiny_patch16_224
  image_size: 224
  patch_size: 16
eval:
  image_size: 224
  embedding_modes: [cls]
  reconstruction_mae: false
datasets:
  - name: cifar10
""".format(output_dir=(tmp_path / "train").as_posix()),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported eval embedding"):
        submit_slurm_pipeline_main([
            "--config", str(config_path),
            "--checkpoint", "random_init",
            "--work-dir", str(tmp_path / "pipeline"),
            "--disable-wandb-merge",
        ])


def test_empty_probe_list_skips_eval_and_merge_jobs(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
run_id: pipeline_empty_probe_smoke
output_dir: "{output_dir}"
seed: 1729
wandb:
  enabled: false
model:
  name: vit_tiny_patch16_224
  image_size: 224
  patch_size: 16
eval:
  image_size: 224
  embedding_modes: [cls_mean]
  probes: []
  reconstruction_mae: false
datasets:
  - name: cifar10
""".format(output_dir=(tmp_path / "train").as_posix()),
        encoding="utf-8",
    )
    work_dir = tmp_path / "pipeline"

    submit_slurm_pipeline_main([
        "--config", str(config_path),
        "--checkpoint", "random_init",
        "--work-dir", str(work_dir),
        "--disable-wandb-merge",
    ])

    slurm_dir = work_dir / "slurm"
    assert (slurm_dir / "02_extract_features_random_init.sbatch").exists()
    assert not (slurm_dir / "03_eval_random_init.sbatch").exists()
    assert not (slurm_dir / "05_merge_eval.sbatch").exists()


def test_cached_feature_eval_roundtrip(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
run_id: cache_eval_smoke
output_dir: "{output_dir}"
seed: 1729
wandb:
  enabled: false
model:
  image_size: 64
  patch_size: 16
  embed_dim: 48
  depth: 1
  num_heads: 3
mae:
  decoder_embed_dim: 48
  decoder_depth: 1
  decoder_num_heads: 3
eval:
  image_size: 64
  batch_size: 4
  num_workers: 0
  embedding_modes: [cls_mean]
  probes: [linear]
  linear_epochs: 3
  linear_batch_size: 8
  reconstruction_mae: false
datasets:
  - name: fake
    subset_train: 16
    subset_eval: 8
""".format(output_dir=(tmp_path / "results").as_posix()),
        encoding="utf-8",
    )

    features_root = tmp_path / "features"
    eval_csv = tmp_path / "eval.csv"
    extract_feature_cache_main([
        "--config", str(config_path),
        "--checkpoint", "random_init",
        "--out", str(features_root),
    ])
    evaluate_feature_cache_main([
        "--config", str(config_path),
        "--checkpoint", "random_init",
        "--features", str(features_root),
        "--out", str(eval_csv),
        "--disable-wandb",
    ])

    with eval_csv.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    probes = {row["probe_type"] for row in rows}
    modes = {row["embedding_mode"] for row in rows}
    assert probes == {"linear", "linear_top5", "linear_loss"}
    assert modes == {"cls_mean"}
    assert all(row["eval_dataset"] == "fake" for row in rows)


def test_imagenet_r_reuses_imagenet1k_train_feature_path(tmp_path) -> None:
    imagenet = {"name": "imagenet1k", "eval_split": "test"}
    imagenet_r = {"name": "imagenet_r", "eval_split": "test"}

    assert feature_split_path(tmp_path, imagenet_r, "train", "cls_mean") == feature_split_path(tmp_path, imagenet, "train", "cls_mean")
    assert feature_split_path(tmp_path, imagenet_r, "eval", "cls_mean") != feature_split_path(tmp_path, imagenet, "eval", "cls_mean")


def test_slurm_pipeline_dry_run_writes_split_linear_eval_jobs(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
run_id: pipeline_smoke
output_dir: "{output_dir}"
seed: 1729
wandb:
  enabled: false
model:
  name: vit_tiny_patch16_224
  image_size: 224
  patch_size: 16
eval:
  image_size: 224
  embedding_modes: [cls_mean]
  probes: [linear]
  reconstruction_mae: true
datasets:
  - name: cifar10
  - name: imagenet1k
""".format(output_dir=(tmp_path / "train").as_posix()),
        encoding="utf-8",
    )
    work_dir = tmp_path / "pipeline"

    submit_slurm_pipeline_main([
        "--config", str(config_path),
        "--checkpoint", "random_init",
        "--work-dir", str(work_dir),
        "--split-eval-by-dataset",
        "--disable-wandb-merge",
    ])

    slurm_dir = work_dir / "slurm"
    assert (slurm_dir / "02_extract_features_random_init.sbatch").exists()
    assert (slurm_dir / "03_cpu_random_init_cifar10.sbatch").exists()
    assert (slurm_dir / "03_cpu_random_init_imagenet1k.sbatch").exists()
    for script in slurm_dir.glob("*.sbatch"):
        assert "evaluate_feature_cache" not in script.read_text(encoding="utf-8") or "--linear-device" in script.read_text(encoding="utf-8")


def test_slurm_pipeline_dry_run_can_group_eval_by_checkpoint(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
run_id: pipeline_grouped_eval_smoke
output_dir: "{output_dir}"
seed: 1729
wandb:
  enabled: false
model:
  name: vit_tiny_patch16_224
  image_size: 224
  patch_size: 16
eval:
  image_size: 224
  embedding_modes: [cls_mean]
  probes: [linear]
  reconstruction_mae: false
datasets:
  - name: cifar10
  - name: cifar100
""".format(output_dir=(tmp_path / "train").as_posix()),
        encoding="utf-8",
    )
    work_dir = tmp_path / "pipeline"

    submit_slurm_pipeline_main([
        "--config", str(config_path),
        "--checkpoint", "random_init",
        "--work-dir", str(work_dir),
        "--group-eval-by-checkpoint",
        "--disable-wandb-merge",
    ])

    slurm_dir = work_dir / "slurm"
    grouped = slurm_dir / "03_eval_random_init.sbatch"
    online_merge = slurm_dir / "04_merge_random_init.sbatch"
    final_merge = slurm_dir / "05_merge_eval.sbatch"
    assert grouped.exists()
    assert online_merge.exists()
    assert final_merge.exists()
    assert not (slurm_dir / "03_cpu_random_init_cifar10.sbatch").exists()
    content = grouped.read_text(encoding="utf-8")
    assert "#SBATCH --partition=gpu_h100" in content
    assert "#SBATCH --gpus=1" in content
    assert "--linear-device cuda" in content
    assert "eval_features_random_init.csv" in online_merge.read_text(encoding="utf-8")
    assert "--disable-wandb" in online_merge.read_text(encoding="utf-8")
    assert "--disable-wandb" in final_merge.read_text(encoding="utf-8")


def test_grouped_slurm_linear_device_can_defer_to_config(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
run_id: pipeline_grouped_eval_config_device_smoke
output_dir: "{output_dir}"
seed: 1729
wandb:
  enabled: false
model:
  name: vit_tiny_patch16_224
  image_size: 224
  patch_size: 16
eval:
  image_size: 224
  embedding_modes: [cls_mean]
  probes: [linear]
  linear_device: cuda
  reconstruction_mae: false
datasets:
  - name: cifar10
""".format(output_dir=(tmp_path / "train").as_posix()),
        encoding="utf-8",
    )
    work_dir = tmp_path / "pipeline"

    submit_slurm_pipeline_main([
        "--config", str(config_path),
        "--checkpoint", "random_init",
        "--work-dir", str(work_dir),
        "--group-eval-by-checkpoint",
        "--linear-device", "config",
        "--disable-wandb-merge",
    ])

    content = (work_dir / "slurm" / "03_eval_random_init.sbatch").read_text(encoding="utf-8")
    assert "--linear-device config" in content


def test_slurm_pipeline_dry_run_supports_multiple_training_checkpoints(tmp_path) -> None:
    config_path = tmp_path / "config.yaml"
    output_dir = tmp_path / "train"
    config_path.write_text(
        """
run_id: pipeline_multi_ckpt_smoke
output_dir: "{output_dir}"
seed: 1729
wandb:
  enabled: false
model:
  name: vit_tiny_patch16_224
  image_size: 224
  patch_size: 16
eval:
  image_size: 224
  embedding_modes: [cls_mean]
  probes: [linear]
  reconstruction_mae: false
datasets:
  - name: cifar10
""".format(output_dir=output_dir.as_posix()),
        encoding="utf-8",
    )
    work_dir = tmp_path / "pipeline"

    submit_slurm_pipeline_main([
        "--config", str(config_path),
        "--checkpoints", str(output_dir / "checkpoint_step_0000100.pt"), str(output_dir / "checkpoint_final.pt"),
        "--work-dir", str(work_dir),
        "--split-eval-by-dataset",
        "--disable-wandb-merge",
    ])

    slurm_dir = work_dir / "slurm"
    assert (slurm_dir / "01_train.sbatch").exists()
    assert (slurm_dir / "02_extract_features_checkpoint_step_0000100.sbatch").exists()
    assert (slurm_dir / "02_extract_features_checkpoint_final.sbatch").exists()
    assert (slurm_dir / "03_cpu_checkpoint_step_0000100_cifar10.sbatch").exists()
    assert (slurm_dir / "03_cpu_checkpoint_final_cifar10.sbatch").exists()


def test_slurm_pipeline_can_resume_training_with_dependency(tmp_path, capsys) -> None:
    config_path = tmp_path / "config.yaml"
    output_dir = tmp_path / "train"
    resume_path = tmp_path / "checkpoint_step_0000320.pt"
    config_path.write_text(
        """
run_id: pipeline_resume_train_smoke
output_dir: "{output_dir}"
seed: 1729
wandb:
  enabled: false
model:
  name: vit_tiny_patch16_224
  image_size: 224
  patch_size: 16
eval:
  image_size: 224
  embedding_modes: [cls_mean]
  probes: [linear]
  reconstruction_mae: false
datasets:
  - name: cifar10
""".format(output_dir=output_dir.as_posix()),
        encoding="utf-8",
    )
    work_dir = tmp_path / "pipeline"

    submit_slurm_pipeline_main([
        "--config", str(config_path),
        "--train-resume", str(resume_path),
        "--train-dependency", "12345",
        "--checkpoints", str(output_dir / "checkpoint_final.pt"),
        "--work-dir", str(work_dir),
        "--group-eval-by-checkpoint",
    ])

    output = capsys.readouterr().out
    train_script = (work_dir / "slurm" / "01_train.sbatch").read_text(encoding="utf-8")
    assert "--dependency=afterok:12345" in output
    assert "--resume" in train_script
    assert str(resume_path) in train_script


def test_grouped_slurm_online_merges_are_chained_for_multiple_checkpoints(tmp_path, capsys) -> None:
    config_path = tmp_path / "config.yaml"
    output_dir = tmp_path / "train"
    config_path.write_text(
        """
run_id: pipeline_grouped_multi_ckpt_smoke
output_dir: "{output_dir}"
seed: 1729
wandb:
  enabled: true
model:
  name: vit_tiny_patch16_224
  image_size: 224
  patch_size: 16
eval:
  image_size: 224
  embedding_modes: [cls_mean]
  probes: [linear]
  reconstruction_mae: false
datasets:
  - name: cifar10
""".format(output_dir=output_dir.as_posix()),
        encoding="utf-8",
    )
    work_dir = tmp_path / "pipeline"

    submit_slurm_pipeline_main([
        "--config", str(config_path),
        "--checkpoints", str(output_dir / "checkpoint_step_0000100.pt"), str(output_dir / "checkpoint_final.pt"),
        "--work-dir", str(work_dir),
        "--group-eval-by-checkpoint",
    ])

    output = capsys.readouterr().out
    assert (
        "--dependency=afterok:<pipeline_grouped_multi_ckpt_smoke-eval-checkpoint_final>:"
        "<pipeline_grouped_multi_ckpt_smoke-merge-checkpoint_step_0000100>"
    ) in output
