from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from src.irc_vit.config import (
    DEFAULT_DECODER_DEPTH,
    DEFAULT_DECODER_EMBED_DIM,
    DEFAULT_DECODER_NUM_HEADS,
    DEFAULT_IN_CHANS,
    DEFAULT_VIT_DEPTH,
    DEFAULT_VIT_DROPOUT,
    DEFAULT_VIT_EMBED_DIM,
    DEFAULT_VIT_MLP_RATIO,
    DEFAULT_VIT_NUM_HEADS,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_MASK_RATIO,
    DEFAULT_MODEL_NAME,
    DEFAULT_PATCH_SIZE,
    MODEL_PRESETS,
)


@dataclass
class ViTConfig:
    image_size: int = DEFAULT_IMAGE_SIZE
    patch_size: int = DEFAULT_PATCH_SIZE
    in_chans: int = DEFAULT_IN_CHANS
    embed_dim: int = DEFAULT_VIT_EMBED_DIM
    depth: int = DEFAULT_VIT_DEPTH
    num_heads: int = DEFAULT_VIT_NUM_HEADS
    mlp_ratio: float = DEFAULT_VIT_MLP_RATIO
    dropout: float = DEFAULT_VIT_DROPOUT


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn
        return x + self.mlp(self.norm2(x))


class PatchEmbed(nn.Module):
    def __init__(self, image_size: int, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class VisionTransformerEncoder(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config
        self.patch_embed = PatchEmbed(config.image_size, config.patch_size, config.in_chans, config.embed_dim)
        self.num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, config.embed_dim))
        self.blocks = nn.ModuleList([
            TransformerBlock(config.embed_dim, config.num_heads, config.mlp_ratio, config.dropout)
            for _ in range(config.depth)
        ])
        self.norm = nn.LayerNorm(config.embed_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        return self.patch_embed(x) + self.pos_embed[:, 1:]

    def forward_tokens(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        b = patch_tokens.shape[0]
        cls = self.cls_token.expand(b, -1, -1) + self.pos_embed[:, :1]
        x = torch.cat([cls, patch_tokens], dim=1)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_tokens(self.patch_tokens(x))

    def encode(self, x: torch.Tensor, mode: str = "cls") -> torch.Tensor:
        tokens = self.forward(x)
        cls = tokens[:, 0]
        mean = tokens[:, 1:].mean(dim=1)
        if mode == "cls":
            return cls
        if mode == "mean":
            return mean
        if mode == "cls_mean":
            return torch.cat([cls, mean], dim=1)
        raise ValueError(f"Unknown embedding mode: {mode}")


def build_vit_config(config: dict[str, Any]) -> ViTConfig:
    name = str(config.get("name", config.get("preset", DEFAULT_MODEL_NAME)))
    encoder_keys = ("image_size", "patch_size", "in_chans", "embed_dim", "depth", "num_heads", "mlp_ratio", "dropout")
    preset = MODEL_PRESETS.get(name, {})
    values = {key: preset[key] for key in encoder_keys if key in preset}
    for key in encoder_keys:
        if key in config:
            values[key] = config[key]
    return ViTConfig(**values)


def build_encoder(config: dict[str, Any]) -> VisionTransformerEncoder:
    return VisionTransformerEncoder(build_vit_config(config))


def build_mae_config(config: dict[str, Any]) -> dict[str, Any]:
    model_cfg = config.get("model", config)
    name = str(model_cfg.get("name", model_cfg.get("preset", DEFAULT_MODEL_NAME)))
    preset = MODEL_PRESETS.get(name, {})
    values = {
        "decoder_embed_dim": DEFAULT_DECODER_EMBED_DIM,
        "decoder_depth": DEFAULT_DECODER_DEPTH,
        "decoder_num_heads": DEFAULT_DECODER_NUM_HEADS,
        "mask_ratio": DEFAULT_MASK_RATIO,
    }
    for key in ("decoder_embed_dim", "decoder_depth", "decoder_num_heads"):
        if key in preset:
            values[key] = preset[key]
        if key in model_cfg:
            values[key] = model_cfg[key]
    mae_cfg = config.get("mae", {})
    for key in ("decoder_embed_dim", "decoder_depth", "decoder_num_heads", "mask_ratio"):
        if key in mae_cfg:
            values[key] = mae_cfg[key]
    return values


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    b, c, h, w = images.shape
    if h != w or h % patch_size != 0:
        raise ValueError("patchify expects square images divisible by patch_size")
    p = patch_size
    n = h // p
    x = images.reshape(b, c, n, p, n, p)
    x = x.permute(0, 2, 4, 3, 5, 1)
    return x.reshape(b, n * n, p * p * c)


def unpatchify(patches: torch.Tensor, patch_size: int, image_size: int, channels: int = 3) -> torch.Tensor:
    b, num_patches, dim = patches.shape
    n = image_size // patch_size
    if num_patches != n * n or dim != patch_size * patch_size * channels:
        raise ValueError("bad patch shape for unpatchify")
    x = patches.reshape(b, n, n, patch_size, patch_size, channels)
    x = x.permute(0, 5, 1, 3, 2, 4)
    return x.reshape(b, channels, image_size, image_size)


class MaskedAutoencoderViT(nn.Module):
    def __init__(
            self,
            encoder: VisionTransformerEncoder,
            decoder_embed_dim: int = DEFAULT_DECODER_EMBED_DIM,
            decoder_depth: int = DEFAULT_DECODER_DEPTH,
            decoder_num_heads: int = DEFAULT_DECODER_NUM_HEADS,
            mask_ratio: float = DEFAULT_MASK_RATIO,
    ):
        super().__init__()
        self.encoder = encoder
        self.mask_ratio = float(mask_ratio)
        self.patch_size = encoder.config.patch_size
        self.image_size = encoder.config.image_size
        self.num_patches = encoder.num_patches
        self.patch_dim = self.patch_size * self.patch_size * encoder.config.in_chans

        enc_dim = encoder.config.embed_dim
        self.decoder_embed = nn.Linear(enc_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, decoder_embed_dim))
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(decoder_embed_dim, decoder_num_heads, 4.0, 0.0)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, self.patch_dim)
        self._init_decoder()

    def _init_decoder(self) -> None:
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.decoder_pred.weight, std=0.02)
        nn.init.zeros_(self.decoder_pred.bias)

    def random_mask(self, batch_size: int, device: torch.device, generator: torch.Generator | None = None):
        n = self.num_patches
        keep = max(1, int(n * (1.0 - self.mask_ratio)))
        noise = torch.rand(batch_size, n, device=device, generator=generator)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :keep]
        mask = torch.ones(batch_size, n, device=device)
        mask[:, :keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return ids_keep, ids_restore, mask

    def forward(self, images: torch.Tensor, generator: torch.Generator | None = None) -> dict[str, torch.Tensor]:
        b = images.shape[0]
        ids_keep, ids_restore, mask = self.random_mask(b, images.device, generator)
        patch_tokens = self.encoder.patch_tokens(images)
        visible = torch.gather(patch_tokens, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, patch_tokens.shape[-1]))
        latent = self.encoder.forward_tokens(visible)
        decoded_visible = self.decoder_embed(latent[:, 1:])

        n_mask = self.num_patches - decoded_visible.shape[1]
        mask_tokens = self.mask_token.expand(b, n_mask, -1)
        decoded = torch.cat([decoded_visible, mask_tokens], dim=1)
        decoded = torch.gather(decoded, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, decoded.shape[-1]))
        decoded = decoded + self.decoder_pos_embed
        for block in self.decoder_blocks:
            decoded = block(decoded)
        pred = self.decoder_pred(self.decoder_norm(decoded))
        target = patchify(images, self.patch_size)
        loss_per_patch = (pred - target).pow(2).mean(dim=-1)
        loss = (loss_per_patch * mask).sum() / mask.sum().clamp_min(1.0)
        return {"loss": loss, "pred": pred, "target": target, "mask": mask}

    @torch.no_grad()
    def reconstruction_preview(
            self,
            images: torch.Tensor,
            generator: torch.Generator | None = None,
            mask_value: float = 0.5,
    ) -> dict[str, torch.Tensor]:
        out = self.forward(images, generator=generator)
        target = out["target"]
        mask = out["mask"].bool()

        masked = target.clone()
        masked[mask] = mask_value

        reconstructed = target.clone()
        reconstructed[mask] = out["pred"][mask]

        return {
            "original": images.clamp(0, 1),
            "masked_input": unpatchify(masked, self.patch_size, self.image_size, images.shape[1]).clamp(0, 1),
            "reconstruction": unpatchify(reconstructed, self.patch_size, self.image_size, images.shape[1]).clamp(0, 1),
        }

    @torch.no_grad()
    def reconstruct(self, images: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        return self.reconstruction_preview(images, generator=generator)["reconstruction"]


def build_mae(config: dict[str, Any]) -> MaskedAutoencoderViT:
    encoder = build_encoder(config.get("model", config))
    mae_cfg = build_mae_config(config)
    return MaskedAutoencoderViT(
        encoder=encoder,
        decoder_embed_dim=int(mae_cfg["decoder_embed_dim"]),
        decoder_depth=int(mae_cfg["decoder_depth"]),
        decoder_num_heads=int(mae_cfg["decoder_num_heads"]),
        mask_ratio=float(mae_cfg["mask_ratio"]),
    )
