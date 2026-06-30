from __future__ import annotations

import torch

from src.irc_vit.model import patchify, unpatchify


def test_patchify_roundtrip() -> None:
    x = torch.rand(2, 3, 64, 64)
    patches = patchify(x, patch_size=16)
    assert patches.shape == (2, 16, 16 * 16 * 3)
    y = unpatchify(patches, patch_size=16, image_size=64)
    assert torch.allclose(x, y)
