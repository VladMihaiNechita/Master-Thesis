from __future__ import annotations

import json

import torch

from src.irc_vit.data import CACHE_FORMAT_VERSION
from src.irc_vit.evaluate import LINEAR_PROBE_HEAD_FORMAT, LINEAR_PROBE_HEAD_VERSION
from src.irc_vit.supervised_imagenet import (
    CachedTensorBatchStream,
    SupervisedViTClassifier,
    accuracy_counts,
    load_mae_encoder_weights,
    load_linear_probe_head,
    next_eval_target,
    parse_args,
)


def tiny_model_config() -> dict:
    return {
        "image_size": 32,
        "patch_size": 16,
        "embed_dim": 24,
        "depth": 1,
        "num_heads": 3,
        "mlp_ratio": 2.0,
    }


def test_supervised_classifier_outputs_class_logits() -> None:
    model = SupervisedViTClassifier(tiny_model_config(), num_classes=7)
    logits = model(torch.rand(2, 3, 32, 32))
    assert logits.shape == (2, 7)


def test_supervised_classifier_can_use_cls_mean_embedding() -> None:
    model = SupervisedViTClassifier({**tiny_model_config(), "classifier_embedding_mode": "cls_mean"}, num_classes=7)
    logits = model(torch.rand(2, 3, 32, 32))
    assert logits.shape == (2, 7)
    assert model.head.in_features == 48


def test_parse_args_strips_submit_wrapper_go_prefix() -> None:
    args = parse_args(["go", "--config", "config.yaml"])
    assert args.config == "config.yaml"


def test_accuracy_counts_handles_top1_and_top5() -> None:
    logits = torch.tensor(
        [
            [5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            [0.0, 5.0, 4.0, 3.0, 2.0, 1.0],
        ]
    )
    labels = torch.tensor([0, 2, 0])
    counts = accuracy_counts(logits, labels, topk=(1, 5))
    assert counts[1] == 1
    assert counts[5] == 2


def test_load_mae_encoder_weights_strips_encoder_prefix(tmp_path) -> None:
    source = SupervisedViTClassifier(tiny_model_config(), num_classes=3)
    with torch.no_grad():
        for param in source.encoder.parameters():
            param.fill_(0.123)
    checkpoint = tmp_path / "mae.pt"
    torch.save(
        {
            "model": {f"encoder.{key}": value.clone() for key, value in source.encoder.state_dict().items()},
            "step": 42,
        },
        checkpoint,
    )

    target = SupervisedViTClassifier(tiny_model_config(), num_classes=3)
    info = load_mae_encoder_weights(target, checkpoint, strict=True)

    assert info["source_step"] == 42
    assert info["loaded_keys"] == len(source.encoder.state_dict())
    for key, value in source.encoder.state_dict().items():
        assert torch.allclose(target.encoder.state_dict()[key], value)


def test_load_linear_probe_head_initializes_supervised_head_rows(tmp_path) -> None:
    model = SupervisedViTClassifier({**tiny_model_config(), "classifier_embedding_mode": "cls_mean"}, num_classes=3)
    weight = torch.randn(3, model.head.in_features)
    bias = torch.randn(3)
    classes = torch.tensor([2, 0, 1])
    checkpoint = tmp_path / "linear_probe_head.pt"
    torch.save(
        {
            "format": LINEAR_PROBE_HEAD_FORMAT,
            "version": LINEAR_PROBE_HEAD_VERSION,
            "classes": classes,
            "head": {"weight": weight, "bias": bias},
            "metadata": {"embedding_mode": "cls_mean", "dataset": "imagenet1k", "images_seen": 20_000_000},
        },
        checkpoint,
    )

    info = load_linear_probe_head(model, checkpoint)

    assert info["classes"] == 3
    assert info["dataset"] == "imagenet1k"
    for row, class_id in enumerate(classes.tolist()):
        assert torch.allclose(model.head.weight[class_id], weight[row])
        assert torch.allclose(model.head.bias[class_id], bias[row])


def test_cached_tensor_batch_stream_reads_shuffled_batches(tmp_path) -> None:
    root = tmp_path / "train"
    root.mkdir()
    images0 = torch.arange(4 * 3 * 8 * 8, dtype=torch.uint8).reshape(4, 3, 8, 8)
    labels0 = torch.tensor([0, 1, 2, 3])
    images1 = torch.arange(4 * 3 * 8 * 8, dtype=torch.uint8).reshape(4, 3, 8, 8)
    labels1 = torch.tensor([4, 5, 6, 7])
    torch.save({"images": images0, "labels": labels0}, root / "shard_00000.pt")
    torch.save({"images": images1, "labels": labels1}, root / "shard_00001.pt")
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "format": CACHE_FORMAT_VERSION,
                "dataset": "fake",
                "split": "train",
                "image_size": 8,
                "count": 8,
                "dtype": "uint8",
                "shards": [
                    {"file": "shard_00000.pt", "count": 4},
                    {"file": "shard_00001.pt", "count": 4},
                ],
            }
        ),
        encoding="utf-8",
    )

    stream = CachedTensorBatchStream(root, batch_size=5, seed=123, shuffle_shards=True, shuffle_within_shard=True)
    images, labels = stream.next_batch(torch.device("cpu"))

    assert images.shape == (5, 3, 8, 8)
    assert labels.shape == (5,)
    assert images.dtype == torch.float32
    assert 0.0 <= float(images.min()) <= float(images.max()) <= 1.0
    assert set(labels.tolist()).issubset(set(range(8)))


def test_cached_tensor_batch_stream_restores_state(tmp_path) -> None:
    root = tmp_path / "train"
    root.mkdir()
    shards = []
    for shard_index in range(3):
        images = torch.full((4, 3, 8, 8), shard_index, dtype=torch.uint8)
        labels = torch.arange(shard_index * 4, shard_index * 4 + 4)
        file_name = f"shard_{shard_index:05d}.pt"
        torch.save({"images": images, "labels": labels}, root / file_name)
        shards.append({"file": file_name, "count": 4})
    (root / "metadata.json").write_text(
        json.dumps(
            {
                "format": CACHE_FORMAT_VERSION,
                "dataset": "fake",
                "split": "train",
                "image_size": 8,
                "count": 12,
                "dtype": "uint8",
                "shards": shards,
            }
        ),
        encoding="utf-8",
    )

    stream = CachedTensorBatchStream(root, batch_size=3, seed=123, shuffle_buffer_shards=2)
    stream.next_batch(torch.device("cpu"))
    state = stream.state_dict()
    expected_images, expected_labels = stream.next_batch(torch.device("cpu"))

    restored = CachedTensorBatchStream(root, batch_size=3, seed=999, shuffle_buffer_shards=2)
    restored.load_state_dict(state)
    actual_images, actual_labels = restored.next_batch(torch.device("cpu"))

    assert torch.equal(actual_images, expected_images)
    assert torch.equal(actual_labels, expected_labels)


def test_next_eval_target_uses_next_million_boundary() -> None:
    assert next_eval_target(0, 1_000_000) == 1_000_000
    assert next_eval_target(1_000_000, 1_000_000) == 2_000_000
    assert next_eval_target(1_001_000, 1_000_000) == 2_000_000
