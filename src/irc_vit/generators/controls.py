from __future__ import annotations

import torch

from src.irc_vit.config import DEFAULT_PATCH_SIZE


def pixel_shuffle(images: torch.Tensor, seed: int | None = None) -> torch.Tensor:
    gen = torch.Generator(device=images.device)
    if seed is not None:
        gen.manual_seed(seed)
    b, c, h, w = images.shape
    flat = images.view(b, c, h * w)
    idx = torch.randperm(h * w, device=images.device, generator=gen)
    return flat[:, :, idx].view(b, c, h, w)


def patch_shuffle(images: torch.Tensor, patch_size: int = DEFAULT_PATCH_SIZE, seed: int | None = None) -> torch.Tensor:
    gen = torch.Generator(device=images.device)
    if seed is not None:
        gen.manual_seed(seed)
    b, c, h, w = images.shape
    if h % patch_size or w % patch_size:
        raise ValueError("image size must be divisible by patch_size")
    patches = images.reshape(b, c, h // patch_size, patch_size, w // patch_size, patch_size)
    patches = patches.permute(0, 2, 4, 1, 3, 5).reshape(b, -1, c, patch_size, patch_size)
    idx = torch.randperm(patches.shape[1], device=images.device, generator=gen)
    patches = patches[:, idx]
    patches = patches.reshape(b, h // patch_size, w // patch_size, c, patch_size, patch_size)
    return patches.permute(0, 3, 1, 4, 2, 5).reshape(b, c, h, w)


def phase_randomize(images: torch.Tensor, seed: int | None = None) -> torch.Tensor:
    """Randomize Fourier phase while preserving each channel amplitude spectrum."""
    gen = torch.Generator(device=images.device)
    if seed is not None:
        gen.manual_seed(seed)
    coeffs = torch.fft.rfft2(images.float(), dim=(-2, -1), norm="ortho")
    phase = torch.rand(coeffs.shape, device=images.device, generator=gen) * 2 * torch.pi
    randomized = torch.abs(coeffs) * torch.exp(1j * phase)
    out = torch.fft.irfft2(randomized, s=images.shape[-2:], dim=(-2, -1), norm="ortho")
    minv = out.amin(dim=(-2, -1), keepdim=True)
    maxv = out.amax(dim=(-2, -1), keepdim=True)
    return ((out - minv) / (maxv - minv).clamp_min(1e-6)).clamp(0, 1)
