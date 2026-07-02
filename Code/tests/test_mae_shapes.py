from __future__ import annotations

import torch

from src.irc_vit.model import build_mae


def tiny_config():
    return {
        "model": {
            "name": "vit_tiny_patch16_224",
            "image_size": 64,
            "patch_size": 16,
            "embed_dim": 48,
            "depth": 2,
            "num_heads": 3,
        },
        "mae": {
            "mask_ratio": 0.75,
            "decoder_embed_dim": 48,
            "decoder_depth": 1,
            "decoder_num_heads": 3,
        },
    }


def test_mae_forward_and_reconstruct_shapes() -> None:
    model = build_mae(tiny_config())
    x = torch.rand(2, 3, 64, 64)
    out = model(x)
    assert out["loss"].ndim == 0
    assert out["pred"].shape == (2, 16, 16 * 16 * 3)
    preview = model.reconstruction_preview(x)
    assert set(preview) == {"original", "masked_input", "reconstruction"}
    assert preview["original"].shape == x.shape
    assert preview["masked_input"].shape == x.shape
    assert preview["reconstruction"].shape == x.shape
    recon = model.reconstruct(x)
    assert recon.shape == x.shape
    assert float(recon.min()) >= 0.0
    assert float(recon.max()) <= 1.0


def test_one_tiny_training_step() -> None:
    model = build_mae(tiny_config())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.rand(2, 3, 64, 64)
    loss = model(x)["loss"]
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)


def test_model_preset_provides_decoder_defaults() -> None:
    model = build_mae({"model": {"name": "vit_small_patch16_224", "image_size": 32, "patch_size": 16}})
    assert model.decoder_embed.out_features == 192
    assert len(model.decoder_blocks) == 4
    assert model.decoder_blocks[0].attn.num_heads == 6


def test_mae_block_overrides_decoder_preset() -> None:
    model = build_mae({
        "model": {"name": "vit_small_patch16_224", "image_size": 32, "patch_size": 16},
        "mae": {"decoder_embed_dim": 48, "decoder_depth": 1, "decoder_num_heads": 3},
    })
    assert model.decoder_embed.out_features == 48
    assert len(model.decoder_blocks) == 1
    assert model.decoder_blocks[0].attn.num_heads == 3
