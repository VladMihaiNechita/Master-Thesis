from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from src.irc_vit.config import (
    DEFAULT_IRC_BUFFER_SIZE,
    DEFAULT_IRC_FOURIER_FRACTION,
    DEFAULT_IRC_FOURIER_FRACTION_AFTER,
    DEFAULT_IRC_FOURIER_FRACTION_SCHEDULE,
    DEFAULT_IRC_FOURIER_FRACTION_SWITCH_IMAGES,
    DEFAULT_IRC_FOURIER_FRACTION_SWITCH_STEP,
    DEFAULT_IRC_HORIZONTAL_FLIP_PROB,
    DEFAULT_IRC_INIT_SOURCE,
    DEFAULT_IRC_INIT_RANDOM_COLOR_FRACTION,
    DEFAULT_IRC_K,
    DEFAULT_IRC_K_CHOICES,
    DEFAULT_IRC_LATENT_GLOBAL_CHANNELS,
    DEFAULT_IRC_LATENT_LOWRES_CHANNELS,
    DEFAULT_IRC_LATENT_LOWRES_SCALES,
    DEFAULT_IRC_LATENT_PIXEL_CHANNELS,
    DEFAULT_IRC_LOCAL_BUFFER_LIMIT,
    DEFAULT_IRC_MODE,
    DEFAULT_IRC_OUTPUT_FOURIER_FRACTION,
    DEFAULT_IRC_OUTPUT_NOISE_FRACTION,
    DEFAULT_IRC_QUANTIZE,
    DEFAULT_IRC_RANDOM_COLOR_FRACTION,
    DEFAULT_IRC_RANDOM_COLOR_FRACTION_AFTER,
    DEFAULT_IRC_RANDOM_COLOR_FRACTION_SCHEDULE,
    DEFAULT_IRC_RANDOM_COLOR_FRACTION_SWITCH_IMAGES,
    DEFAULT_IRC_RANDOM_COLOR_FRACTION_SWITCH_STEP,
    DEFAULT_IRC_REFRESH_FRACTION,
    DEFAULT_IRC_REFRESH_FRACTION_AFTER,
    DEFAULT_IRC_REFRESH_FRACTION_SCHEDULE,
    DEFAULT_IRC_REFRESH_FRACTION_SWITCH_IMAGES,
    DEFAULT_IRC_REFRESH_FRACTION_SWITCH_STEP,
    DEFAULT_IRC_RESET_FRACTION,
    DEFAULT_IRC_RESET_FRACTION_AFTER,
    DEFAULT_IRC_RESET_FRACTION_SCHEDULE,
    DEFAULT_IRC_RESET_FRACTION_SWITCH_IMAGES,
    DEFAULT_IRC_RESET_FRACTION_SWITCH_STEP,
    DEFAULT_IRC_SPECTRAL_ALPHA_MAX,
    DEFAULT_IRC_SPECTRAL_ALPHA_MIN,
    DEFAULT_IRC_SPECTRAL_LATENT_CHANNELS,
    DEFAULT_IRC_UPDATE_STRENGTH,
    DEFAULT_IRC_UPDATE_STRENGTH_AFTER,
    DEFAULT_IRC_UPDATE_OUTPUT_ACTIVATION,
    DEFAULT_IRC_UPDATE_RULE,
    DEFAULT_IRC_UPDATE_STRENGTH_SCHEDULE,
    DEFAULT_IRC_UPDATE_STRENGTH_SWITCH_IMAGES,
    DEFAULT_IRC_UPDATE_STRENGTH_SWITCH_STEP,
    DEFAULT_IRC_VERTICAL_FLIP_PROB,
    DEFAULT_RANDOM_CONV_ACTIVATION,
    DEFAULT_RANDOM_CONV_CHANNELS,
    DEFAULT_RANDOM_CONV_DEPTH,
    DEFAULT_RANDOM_CONV_DILATION,
    DEFAULT_RANDOM_CONV_KERNEL_SIZE,
    DEFAULT_RANDOM_CONV_MULTI_SCALE,
    DEFAULT_RANDOM_CONV_WEIGHT_STD_MAX,
    DEFAULT_RANDOM_CONV_WEIGHT_STD_MIN,
    DEFAULT_RANDOM_CONV_WIDTH,
    DEFAULT_SEED,
)
from src.irc_vit.generators.base import validate_images
from src.irc_vit.generators.fourier import FourierTexture
from src.irc_vit.generators.noise import IIDRGBNoise
from src.irc_vit.generators.shapes import SimpleShapes
from src.irc_vit.utils import torch_generator


def _fraction_count(buffer_size: int, fraction: float, *, minimum_one: bool = False) -> int:
    if fraction <= 0:
        return 0
    count = int(buffer_size * fraction)
    if minimum_one:
        count = max(1, count)
    return min(buffer_size, count)


def _sample_indices(buffer_size: int, count: int, device: torch.device, gen: torch.Generator | None) -> torch.Tensor:
    return torch.randperm(buffer_size, device=device, generator=gen)[:count]


def _quantize_to_uint8_grid(images: torch.Tensor) -> torch.Tensor:
    return (images.clamp(0, 1) * 255.0).round() / 255.0


def _squash_update(update: torch.Tensor, activation: str) -> torch.Tensor:
    activation = activation.lower()
    if activation == "sigmoid":
        return torch.sigmoid(update)
    if activation in {"tanh01", "scaled_tanh"}:
        return (torch.tanh(update) + 1.0) * 0.5
    raise ValueError(f"Unknown IRC update_output_activation '{activation}'. Use 'sigmoid' or 'tanh01'.")


def _apply_update_rule(x: torch.Tensor, target: torch.Tensor, strength: float, rule: str) -> torch.Tensor:
    rule = rule.lower()
    if rule == "blend":
        return ((1.0 - strength) * x + strength * target).clamp(0, 1)
    if rule == "residual":
        return (x + strength * (target - 0.5)).clamp(0, 1)
    raise ValueError(f"Unknown IRC update_rule '{rule}'. Use 'blend' or 'residual'.")


def _random_solid_color_images(
        count: int,
        image_size: int,
        device: torch.device,
        gen: torch.Generator | None,
) -> torch.Tensor:
    colors = torch.randint(0, 256, (count, 3, 1, 1), device=device, generator=gen, dtype=torch.int64)
    colors = colors.to(torch.float32) / 255.0
    return colors.expand(count, 3, image_size, image_size).contiguous()


def _normalize_scales(scales: list[int] | tuple[int, ...] | int | None, image_size: int) -> tuple[int, ...]:
    if scales is None:
        return ()
    if isinstance(scales, int):
        raw = (scales,)
    else:
        raw = tuple(int(value) for value in scales)
    return tuple(sorted({value for value in raw if 1 <= value <= image_size}))


def _spectral_random_fields(
        batch_size: int,
        channels: int,
        image_size: int,
        device: torch.device,
        gen: torch.Generator | None,
        alpha_min: float,
        alpha_max: float,
) -> torch.Tensor:
    if channels <= 0:
        return torch.empty(batch_size, 0, image_size, image_size, device=device)
    fy = torch.fft.fftfreq(image_size, device=device).abs()
    fx = torch.fft.rfftfreq(image_size, device=device).abs()
    radius = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    radius[0, 0] = 1.0
    radius = radius.view(1, 1, image_size, image_size // 2 + 1)
    alpha = torch.empty(batch_size, channels, 1, 1, device=device).uniform_(alpha_min, alpha_max, generator=gen)
    amplitude = radius.pow(-alpha)
    amplitude[..., 0, 0] = 0.0
    shape = (batch_size, channels, image_size, image_size // 2 + 1)
    real = torch.randn(shape, device=device, generator=gen)
    imag = torch.randn(shape, device=device, generator=gen)
    fields = torch.fft.irfft2(torch.complex(real, imag) * amplitude, s=(image_size, image_size), dim=(-2, -1), norm="ortho")
    mean = fields.mean(dim=(-2, -1), keepdim=True)
    std = fields.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return (fields - mean) / std


def _latent_fields(
        batch_size: int,
        image_size: int,
        device: torch.device,
        gen: torch.Generator | None,
        pixel_channels: int,
        lowres_channels: int,
        lowres_scales: tuple[int, ...],
        global_channels: int,
        spectral_channels: int,
        spectral_alpha_min: float,
        spectral_alpha_max: float,
) -> torch.Tensor | None:
    parts: list[torch.Tensor] = []
    if pixel_channels > 0:
        parts.append(torch.randn(batch_size, pixel_channels, image_size, image_size, device=device, generator=gen))
    if lowres_channels > 0:
        for scale in lowres_scales:
            z = torch.randn(batch_size, lowres_channels, scale, scale, device=device, generator=gen)
            parts.append(F.interpolate(z, size=(image_size, image_size), mode="bilinear", align_corners=False))
    if global_channels > 0:
        z = torch.randn(batch_size, global_channels, 1, 1, device=device, generator=gen)
        parts.append(z.expand(-1, -1, image_size, image_size))
    if spectral_channels > 0:
        parts.append(_spectral_random_fields(batch_size, spectral_channels, image_size, device, gen, spectral_alpha_min, spectral_alpha_max))
    if not parts:
        return None
    return torch.cat(parts, dim=1)


def _normalize_k_choices(k: int, k_choices: list[int] | tuple[int, ...] | None) -> tuple[int, ...]:
    choices = (int(k),) if k_choices is None else tuple(int(value) for value in k_choices)
    if not choices:
        raise ValueError("k_choices must contain at least one value")
    invalid = [value for value in choices if value not in {0, 1, 2, 3, 4, 8}]
    if invalid:
        raise ValueError(f"IRC k values must be one of {{0, 1, 2, 3, 4, 8}}, got {invalid}")
    return choices


def _split_channels(total: int, parts: int) -> list[int]:
    base, extra = divmod(int(total), int(parts))
    return [base + (1 if idx < extra else 0) for idx in range(parts)]


def _scheduled_value(
        start: float,
        end: float | None,
        horizon: int,
        schedule: str,
        step: int,
) -> float:
    if end is None or horizon <= 0:
        return float(start)
    schedule = schedule.lower()
    if schedule == "step":
        return float(end) if step >= horizon else float(start)
    if schedule == "linear":
        progress = min(1.0, max(0.0, float(step) / float(horizon)))
        return float(start) + (float(end) - float(start)) * progress
    raise ValueError(f"Unknown IRC schedule '{schedule}'. Use 'step' or 'linear'.")


def _schedule_position_and_horizon(
        step: int,
        images_seen: int,
        switch_step: int,
        switch_images: int | None,
) -> tuple[int, int]:
    if switch_images is not None and int(switch_images) > 0:
        return max(0, int(images_seen)), int(switch_images)
    return max(0, int(step)), int(switch_step)


class RandomConvNet(nn.Module):
    def __init__(
            self,
            in_channels: int = DEFAULT_RANDOM_CONV_CHANNELS,
            width: int = DEFAULT_RANDOM_CONV_WIDTH,
            depth: int = DEFAULT_RANDOM_CONV_DEPTH,
            kernel_size: int = DEFAULT_RANDOM_CONV_KERNEL_SIZE,
            dilation: int = DEFAULT_RANDOM_CONV_DILATION,
            multi_scale: bool = DEFAULT_RANDOM_CONV_MULTI_SCALE,
            out_channels: int = DEFAULT_RANDOM_CONV_CHANNELS,
            activation: str = DEFAULT_RANDOM_CONV_ACTIVATION,
            weight_std_min: float = DEFAULT_RANDOM_CONV_WEIGHT_STD_MIN,
            weight_std_max: float = DEFAULT_RANDOM_CONV_WEIGHT_STD_MAX,
            seed: int | None = DEFAULT_SEED,
            device: torch.device | str = "cpu",
    ):
        super().__init__()
        self.weight_std_min = float(weight_std_min)
        self.weight_std_max = float(weight_std_max)
        act: nn.Module
        if activation == "relu":
            act = nn.ReLU()
        elif activation == "tanh":
            act = nn.Tanh()
        else:
            act = nn.GELU()
        layers: list[nn.Module] = [
            self._make_hidden_conv(in_channels, width, kernel_size, dilation, bool(multi_scale)),
            act,
        ]
        for _ in range(depth - 2):
            layers += [self._make_hidden_conv(width, width, kernel_size, dilation, bool(multi_scale)), act]
        pad = int(dilation) * (kernel_size // 2)
        layers += [nn.Conv2d(width, out_channels, kernel_size, padding=pad, dilation=dilation)]
        self.net = nn.Sequential(*layers).to(device)
        self.reset_random(seed, device)

    @staticmethod
    def _make_hidden_conv(
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            dilation: int,
            multi_scale: bool,
    ) -> nn.Module:
        if not multi_scale:
            pad = int(dilation) * (kernel_size // 2)
            return nn.Conv2d(in_channels, out_channels, kernel_size, padding=pad, dilation=dilation)
        return MultiScaleRandomConv(in_channels, out_channels, dilation)

    def reset_random(self, seed: int | None, device: torch.device | str) -> None:
        gen = torch_generator(torch.device(device), seed)
        low = min(self.weight_std_min, self.weight_std_max)
        high = max(self.weight_std_min, self.weight_std_max)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                std = low
                if high > low:
                    std = float(torch.empty((), device=module.weight.device).uniform_(low, high, generator=gen).item())
                std *= float(getattr(module, "_irc_weight_std_scale", 1.0))
                module.weight.data.normal_(mean=0.0, std=std, generator=gen)
                if module.bias is not None:
                    module.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiScaleRandomConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1):
        super().__init__()
        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}")
        specs = (
            (1, 1),
            (3, 1),
            (5, 1),
            (3, 2),
            (3, 4),
        )
        channels = _split_channels(out_channels, len(specs))
        branches: list[nn.Module] = []
        for width, (kernel_size, branch_dilation) in zip(channels, specs, strict=True):
            if width <= 0:
                continue
            effective_dilation = int(dilation) * branch_dilation
            padding = effective_dilation * (kernel_size // 2)
            conv = nn.Conv2d(in_channels, width, kernel_size, padding=padding, dilation=effective_dilation)
            conv._irc_weight_std_scale = 3.0 / float(kernel_size)
            branches.append(conv)
        self.branches = nn.ModuleList(branches)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([branch(x) for branch in self.branches], dim=1)


@dataclass
class ImageBuffer:
    buffer_size: int
    image_size: int
    device: torch.device | str
    init_source: str = DEFAULT_IRC_INIT_SOURCE
    seed: int = DEFAULT_SEED
    quantize: bool = DEFAULT_IRC_QUANTIZE
    init_random_color_fraction: float = DEFAULT_IRC_INIT_RANDOM_COLOR_FRACTION

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        source = SimpleShapes() if self.init_source == "simple_shapes" else IIDRGBNoise()
        self.images = source.generate(self.buffer_size, self.image_size, self.device, seed=self.seed)
        n_colors = _fraction_count(self.buffer_size, self.init_random_color_fraction)
        if n_colors > 0:
            gen = torch_generator(self.device, self.seed + 17_171)
            idx = _sample_indices(self.buffer_size, n_colors, self.device, gen)
            self.images[idx] = _random_solid_color_images(n_colors, self.image_size, self.device, gen)
        if self.quantize:
            self.images = _quantize_to_uint8_grid(self.images)

    def state_dict(self) -> dict[str, object]:
        state: dict[str, object] = {
            "buffer_size": self.buffer_size,
            "image_size": self.image_size,
            "init_source": self.init_source,
            "seed": self.seed,
            "quantize": self.quantize,
            "init_random_color_fraction": self.init_random_color_fraction,
        }
        if self.quantize:
            state["images_uint8"] = (self.images.detach().cpu().clamp(0, 1) * 255.0).round().to(torch.uint8)
        else:
            state["images"] = self.images.detach().cpu()
        return state

    @classmethod
    def from_state_dict(cls, state: dict[str, object], device: torch.device | str) -> "ImageBuffer":
        buffer = cls.__new__(cls)
        buffer.buffer_size = int(state["buffer_size"])
        buffer.image_size = int(state["image_size"])
        buffer.device = torch.device(device)
        buffer.init_source = str(state.get("init_source", DEFAULT_IRC_INIT_SOURCE))
        buffer.seed = int(state.get("seed", DEFAULT_SEED))
        buffer.quantize = bool(state.get("quantize", DEFAULT_IRC_QUANTIZE))
        buffer.init_random_color_fraction = float(state.get("init_random_color_fraction", DEFAULT_IRC_INIT_RANDOM_COLOR_FRACTION))
        if "images_uint8" in state:
            buffer.images = state["images_uint8"].to(buffer.device, dtype=torch.float32) / 255.0
        else:
            buffer.images = state["images"].to(buffer.device, dtype=torch.float32)
        return buffer

    def sample(
            self,
            batch_size: int,
            seed: int | None = None,
            horizontal_flip_prob: float = 0.0,
            vertical_flip_prob: float = 0.0,
    ) -> torch.Tensor:
        gen = torch_generator(self.device, seed)
        idx = torch.randint(0, self.buffer_size, (batch_size,), device=self.device, generator=gen)
        images = self.images[idx].detach().clone()
        if horizontal_flip_prob > 0:
            flip = torch.rand(batch_size, device=self.device, generator=gen) < horizontal_flip_prob
            if flip.any():
                images[flip] = images[flip].flip(-1)
        if vertical_flip_prob > 0:
            flip = torch.rand(batch_size, device=self.device, generator=gen) < vertical_flip_prob
            if flip.any():
                images[flip] = images[flip].flip(-2)
        return images

    def refresh(
            self,
            net: RandomConvNet,
            steps: int | tuple[int, ...],
            refresh_fraction: float,
            reset_fraction: float,
            random_color_fraction: float,
            fourier_fraction: float,
            update_strength: float,
            update_rule: str,
            update_output_activation: str,
            latent_sampler=None,
            seed: int | None = None,
    ) -> None:
        gen = torch_generator(self.device, seed)
        n_refresh = max(1, _fraction_count(self.buffer_size, refresh_fraction))
        idx = _sample_indices(self.buffer_size, n_refresh, self.device, gen)
        with torch.no_grad():
            x = self.images[idx].detach()
            strength = min(1.0, max(0.0, float(update_strength)))
            if isinstance(steps, int):
                step_counts = None
                max_steps = max(0, int(steps))
            else:
                choices = torch.tensor(list(steps), device=self.device, dtype=torch.long)
                choice_idx = torch.randint(0, choices.numel(), (n_refresh,), device=self.device, generator=gen)
                step_counts = choices[choice_idx]
                max_steps = int(step_counts.max().item()) if step_counts.numel() else 0

            for step in range(max_steps):
                if step_counts is None:
                    net_input = x
                    if latent_sampler is not None:
                        net_input = torch.cat([x, latent_sampler(x.shape[0], gen)], dim=1)
                    update = net(net_input)
                    x = _apply_update_rule(x, _squash_update(update, update_output_activation), strength, update_rule)
                    continue
                active = step_counts > step
                if active.any():
                    active_x = x[active]
                    net_input = active_x
                    if latent_sampler is not None:
                        net_input = torch.cat([active_x, latent_sampler(active_x.shape[0], gen)], dim=1)
                    update = net(net_input)
                    x[active] = _apply_update_rule(
                        x[active],
                        _squash_update(update, update_output_activation),
                        strength,
                        update_rule,
                    )
            if self.quantize:
                x = _quantize_to_uint8_grid(x)
            self.images[idx] = x.detach()
            n_reset = _fraction_count(self.buffer_size, reset_fraction)
            if n_reset > 0:
                ridx = _sample_indices(self.buffer_size, n_reset, self.device, gen)
                fresh = torch.rand(n_reset, 3, self.image_size, self.image_size, device=self.device, generator=gen)
                self.images[ridx] = _quantize_to_uint8_grid(fresh) if self.quantize else fresh
            n_colors = _fraction_count(self.buffer_size, random_color_fraction)
            if n_colors > 0:
                cidx = _sample_indices(self.buffer_size, n_colors, self.device, gen)
                colors = _random_solid_color_images(n_colors, self.image_size, self.device, gen)
                self.images[cidx] = _quantize_to_uint8_grid(colors) if self.quantize else colors
            n_fourier = _fraction_count(self.buffer_size, fourier_fraction, minimum_one=True)
            if n_fourier > 0:
                fidx = _sample_indices(self.buffer_size, n_fourier, self.device, gen)
                fseed = None if seed is None else seed + 1_000_003
                fourier = FourierTexture().generate(n_fourier, self.image_size, self.device, seed=fseed)
                self.images[fidx] = _quantize_to_uint8_grid(fourier) if self.quantize else fourier


class IRCConvGenerator:
    name = "irc_conv"

    def __init__(
            self,
            k: int = DEFAULT_IRC_K,
            mode: str = DEFAULT_IRC_MODE,
            width: int = DEFAULT_RANDOM_CONV_WIDTH,
            depth: int = DEFAULT_RANDOM_CONV_DEPTH,
            kernel_size: int = DEFAULT_RANDOM_CONV_KERNEL_SIZE,
            dilation: int = DEFAULT_RANDOM_CONV_DILATION,
            multi_scale: bool = DEFAULT_RANDOM_CONV_MULTI_SCALE,
            activation: str = DEFAULT_RANDOM_CONV_ACTIVATION,
            weight_std_min: float = DEFAULT_RANDOM_CONV_WEIGHT_STD_MIN,
            weight_std_max: float = DEFAULT_RANDOM_CONV_WEIGHT_STD_MAX,
            buffer_size: int = DEFAULT_IRC_BUFFER_SIZE,
            refresh_fraction: float = DEFAULT_IRC_REFRESH_FRACTION,
            refresh_fraction_after: float | None = DEFAULT_IRC_REFRESH_FRACTION_AFTER,
            refresh_fraction_switch_step: int = DEFAULT_IRC_REFRESH_FRACTION_SWITCH_STEP,
            refresh_fraction_switch_images: int | None = DEFAULT_IRC_REFRESH_FRACTION_SWITCH_IMAGES,
            refresh_fraction_schedule: str = DEFAULT_IRC_REFRESH_FRACTION_SCHEDULE,
            reset_fraction: float = DEFAULT_IRC_RESET_FRACTION,
            init_random_color_fraction: float = DEFAULT_IRC_INIT_RANDOM_COLOR_FRACTION,
            random_color_fraction: float = DEFAULT_IRC_RANDOM_COLOR_FRACTION,
            random_color_fraction_after: float | None = DEFAULT_IRC_RANDOM_COLOR_FRACTION_AFTER,
            random_color_fraction_switch_step: int = DEFAULT_IRC_RANDOM_COLOR_FRACTION_SWITCH_STEP,
            random_color_fraction_switch_images: int | None = DEFAULT_IRC_RANDOM_COLOR_FRACTION_SWITCH_IMAGES,
            random_color_fraction_schedule: str = DEFAULT_IRC_RANDOM_COLOR_FRACTION_SCHEDULE,
            fourier_fraction: float = DEFAULT_IRC_FOURIER_FRACTION,
            fourier_fraction_after: float | None = DEFAULT_IRC_FOURIER_FRACTION_AFTER,
            fourier_fraction_switch_step: int = DEFAULT_IRC_FOURIER_FRACTION_SWITCH_STEP,
            fourier_fraction_switch_images: int | None = DEFAULT_IRC_FOURIER_FRACTION_SWITCH_IMAGES,
            fourier_fraction_schedule: str = DEFAULT_IRC_FOURIER_FRACTION_SCHEDULE,
            output_fourier_fraction: float = DEFAULT_IRC_OUTPUT_FOURIER_FRACTION,
            output_noise_fraction: float = DEFAULT_IRC_OUTPUT_NOISE_FRACTION,
            update_strength: float = DEFAULT_IRC_UPDATE_STRENGTH,
            update_rule: str = DEFAULT_IRC_UPDATE_RULE,
            update_output_activation: str = DEFAULT_IRC_UPDATE_OUTPUT_ACTIVATION,
            update_strength_after: float | None = DEFAULT_IRC_UPDATE_STRENGTH_AFTER,
            update_strength_switch_step: int = DEFAULT_IRC_UPDATE_STRENGTH_SWITCH_STEP,
            update_strength_switch_images: int | None = DEFAULT_IRC_UPDATE_STRENGTH_SWITCH_IMAGES,
            update_strength_schedule: str = DEFAULT_IRC_UPDATE_STRENGTH_SCHEDULE,
            reset_fraction_after: float | None = DEFAULT_IRC_RESET_FRACTION_AFTER,
            reset_fraction_switch_step: int = DEFAULT_IRC_RESET_FRACTION_SWITCH_STEP,
            reset_fraction_switch_images: int | None = DEFAULT_IRC_RESET_FRACTION_SWITCH_IMAGES,
            reset_fraction_schedule: str = DEFAULT_IRC_RESET_FRACTION_SCHEDULE,
            k_choices: list[int] | tuple[int, ...] | None = DEFAULT_IRC_K_CHOICES,
            quantize: bool = DEFAULT_IRC_QUANTIZE,
            horizontal_flip_prob: float = DEFAULT_IRC_HORIZONTAL_FLIP_PROB,
            vertical_flip_prob: float = DEFAULT_IRC_VERTICAL_FLIP_PROB,
            latent_pixel_channels: int = DEFAULT_IRC_LATENT_PIXEL_CHANNELS,
            latent_lowres_channels: int = DEFAULT_IRC_LATENT_LOWRES_CHANNELS,
            latent_lowres_scales: list[int] | tuple[int, ...] | int | None = DEFAULT_IRC_LATENT_LOWRES_SCALES,
            latent_global_channels: int = DEFAULT_IRC_LATENT_GLOBAL_CHANNELS,
            spectral_latent_channels: int = DEFAULT_IRC_SPECTRAL_LATENT_CHANNELS,
            spectral_alpha_min: float = DEFAULT_IRC_SPECTRAL_ALPHA_MIN,
            spectral_alpha_max: float = DEFAULT_IRC_SPECTRAL_ALPHA_MAX,
            init_source: str = DEFAULT_IRC_INIT_SOURCE,
            seed: int = DEFAULT_SEED,
    ):
        if int(k) not in {0, 1, 2, 3, 4, 8}:
            raise ValueError("IRC K must be one of {0, 1, 2, 3, 4, 8}")
        self.k = int(k)
        self.k_choices = _normalize_k_choices(self.k, k_choices)
        self.mode = mode
        self.width = int(width)
        self.depth = int(depth)
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.multi_scale = bool(multi_scale)
        self.activation = activation
        self.weight_std_min = float(weight_std_min)
        self.weight_std_max = float(weight_std_max)
        self.buffer_size = int(buffer_size)
        self.refresh_fraction = float(refresh_fraction)
        self.refresh_fraction_after = None if refresh_fraction_after is None else float(refresh_fraction_after)
        self.refresh_fraction_switch_step = int(refresh_fraction_switch_step)
        self.refresh_fraction_switch_images = None if refresh_fraction_switch_images is None else int(refresh_fraction_switch_images)
        self.refresh_fraction_schedule = refresh_fraction_schedule
        self.reset_fraction = float(reset_fraction)
        self.init_random_color_fraction = float(init_random_color_fraction)
        self.random_color_fraction = float(random_color_fraction)
        self.random_color_fraction_after = None if random_color_fraction_after is None else float(random_color_fraction_after)
        self.random_color_fraction_switch_step = int(random_color_fraction_switch_step)
        self.random_color_fraction_switch_images = None if random_color_fraction_switch_images is None else int(random_color_fraction_switch_images)
        self.random_color_fraction_schedule = random_color_fraction_schedule
        self.fourier_fraction = float(fourier_fraction)
        self.fourier_fraction_after = None if fourier_fraction_after is None else float(fourier_fraction_after)
        self.fourier_fraction_switch_step = int(fourier_fraction_switch_step)
        self.fourier_fraction_switch_images = None if fourier_fraction_switch_images is None else int(fourier_fraction_switch_images)
        self.fourier_fraction_schedule = fourier_fraction_schedule
        self.output_fourier_fraction = float(output_fourier_fraction)
        self.output_noise_fraction = float(output_noise_fraction)
        self.update_strength = float(update_strength)
        self.update_rule = str(update_rule).lower()
        if self.update_rule not in {"blend", "residual"}:
            raise ValueError("IRC update_rule must be 'blend' or 'residual'")
        self.update_output_activation = str(update_output_activation).lower()
        if self.update_output_activation == "scaled_tanh":
            self.update_output_activation = "tanh01"
        if self.update_output_activation not in {"sigmoid", "tanh01"}:
            raise ValueError("IRC update_output_activation must be 'sigmoid' or 'tanh01'")
        self.update_strength_after = None if update_strength_after is None else float(update_strength_after)
        self.update_strength_switch_step = int(update_strength_switch_step)
        self.update_strength_switch_images = None if update_strength_switch_images is None else int(update_strength_switch_images)
        self.update_strength_schedule = update_strength_schedule
        self.reset_fraction_after = None if reset_fraction_after is None else float(reset_fraction_after)
        self.reset_fraction_switch_step = int(reset_fraction_switch_step)
        self.reset_fraction_switch_images = None if reset_fraction_switch_images is None else int(reset_fraction_switch_images)
        self.reset_fraction_schedule = reset_fraction_schedule
        self.quantize = bool(quantize)
        self.horizontal_flip_prob = float(horizontal_flip_prob)
        self.vertical_flip_prob = float(vertical_flip_prob)
        self.latent_pixel_channels = int(latent_pixel_channels)
        self.latent_lowres_channels = int(latent_lowres_channels)
        self.latent_lowres_scales = _normalize_scales(latent_lowres_scales, 10_000)
        self.latent_global_channels = int(latent_global_channels)
        self.spectral_latent_channels = int(spectral_latent_channels)
        self.spectral_alpha_min = float(spectral_alpha_min)
        self.spectral_alpha_max = float(spectral_alpha_max)
        latent_counts = (
            self.latent_pixel_channels,
            self.latent_lowres_channels,
            self.latent_global_channels,
            self.spectral_latent_channels,
        )
        if any(value < 0 for value in latent_counts):
            raise ValueError("IRC latent channel counts must be non-negative")
        self.init_source = init_source
        self.seed = int(seed)
        self._buffer: ImageBuffer | None = None
        self._fixed_net: RandomConvNet | None = None
        self._calls = 0
        self._images_seen = 0

    def _ensure_state(self, image_size: int, device: torch.device) -> None:
        if self._buffer is None or self._buffer.image_size != image_size or self._buffer.device != device:
            calls = self._calls
            images_seen = self._images_seen
            self._buffer = ImageBuffer(
                self.buffer_size,
                image_size,
                device,
                self.init_source,
                seed=self.seed,
                quantize=self.quantize,
                init_random_color_fraction=self.init_random_color_fraction,
            )
            self._fixed_net = None
            self._calls = calls
            self._images_seen = images_seen
        if self.mode == "fixed_generator_per_run" and self._fixed_net is None:
            self._fixed_net = RandomConvNet(
                self._net_in_channels(image_size),
                self.width,
                self.depth,
                self.kernel_size,
                self.dilation,
                self.multi_scale,
                3,
                self.activation,
                self.weight_std_min,
                self.weight_std_max,
                self.seed,
                device,
            )

    def _latent_channels(self, image_size: int) -> int:
        lowres_scales = _normalize_scales(self.latent_lowres_scales, image_size)
        return (
            self.latent_pixel_channels
            + self.latent_lowres_channels * len(lowres_scales)
            + self.latent_global_channels
            + self.spectral_latent_channels
        )

    def _net_in_channels(self, image_size: int) -> int:
        return 3 + self._latent_channels(image_size)

    def _make_net(self, image_size: int, device: torch.device, seed: int | None) -> RandomConvNet:
        if self.mode == "fixed_generator_per_run":
            assert self._fixed_net is not None
            return self._fixed_net
        return RandomConvNet(
            self._net_in_channels(image_size),
            self.width,
            self.depth,
            self.kernel_size,
            self.dilation,
            self.multi_scale,
            3,
            self.activation,
            self.weight_std_min,
            self.weight_std_max,
            seed,
            device,
        )

    def _latent_sampler(self, image_size: int, device: torch.device):
        lowres_scales = _normalize_scales(self.latent_lowres_scales, image_size)
        if self._latent_channels(image_size) <= 0:
            return None

        def sample(batch_size: int, gen: torch.Generator | None) -> torch.Tensor:
            latents = _latent_fields(
                batch_size,
                image_size,
                device,
                gen,
                self.latent_pixel_channels,
                self.latent_lowres_channels,
                lowres_scales,
                self.latent_global_channels,
                self.spectral_latent_channels,
                self.spectral_alpha_min,
                self.spectral_alpha_max,
            )
            assert latents is not None
            return latents

        return sample

    def _current_fourier_fraction(self) -> float:
        position, horizon = _schedule_position_and_horizon(
            self._calls,
            self._images_seen,
            self.fourier_fraction_switch_step,
            self.fourier_fraction_switch_images,
        )
        return _scheduled_value(
            self.fourier_fraction,
            self.fourier_fraction_after,
            horizon,
            self.fourier_fraction_schedule,
            position,
        )

    def _current_random_color_fraction(self) -> float:
        position, horizon = _schedule_position_and_horizon(
            self._calls,
            self._images_seen,
            self.random_color_fraction_switch_step,
            self.random_color_fraction_switch_images,
        )
        return _scheduled_value(
            self.random_color_fraction,
            self.random_color_fraction_after,
            horizon,
            self.random_color_fraction_schedule,
            position,
        )

    def _current_refresh_fraction(self) -> float:
        position, horizon = _schedule_position_and_horizon(
            self._calls,
            self._images_seen,
            self.refresh_fraction_switch_step,
            self.refresh_fraction_switch_images,
        )
        return _scheduled_value(
            self.refresh_fraction,
            self.refresh_fraction_after,
            horizon,
            self.refresh_fraction_schedule,
            position,
        )

    def _current_update_strength(self) -> float:
        position, horizon = _schedule_position_and_horizon(
            self._calls,
            self._images_seen,
            self.update_strength_switch_step,
            self.update_strength_switch_images,
        )
        return _scheduled_value(
            self.update_strength,
            self.update_strength_after,
            horizon,
            self.update_strength_schedule,
            position,
        )

    def _current_reset_fraction(self) -> float:
        position, horizon = _schedule_position_and_horizon(
            self._calls,
            self._images_seen,
            self.reset_fraction_switch_step,
            self.reset_fraction_switch_images,
        )
        return _scheduled_value(
            self.reset_fraction,
            self.reset_fraction_after,
            horizon,
            self.reset_fraction_schedule,
            position,
        )

    def set_step(self, step: int) -> None:
        self._calls = max(0, int(step))

    def set_images_seen(self, images_seen: int) -> None:
        self._images_seen = max(0, int(images_seen))

    def set_progress(self, step: int | None = None, images_seen: int | None = None) -> None:
        if step is not None:
            self.set_step(step)
        if images_seen is not None:
            self.set_images_seen(images_seen)

    def state_dict(self) -> dict[str, object]:
        state: dict[str, object] = {
            "calls": self._calls,
            "images_seen": self._images_seen,
        }
        if self._buffer is not None:
            state["buffer"] = self._buffer.state_dict()
        if self._fixed_net is not None:
            state["fixed_net"] = {key: value.detach().cpu() for key, value in self._fixed_net.state_dict().items()}
        return state

    def load_state_dict(self, state: dict[str, object], device: torch.device | str | None = None) -> None:
        self._calls = int(state.get("calls", 0))
        self._images_seen = int(state.get("images_seen", 0))
        if "buffer" in state:
            buffer_state = state["buffer"]
            if not isinstance(buffer_state, dict):
                raise TypeError("IRCConvGenerator buffer state must be a dict")
            load_device = torch.device(device) if device is not None else torch.device("cpu")
            self._buffer = ImageBuffer.from_state_dict(buffer_state, load_device)
        if "fixed_net" in state and self._buffer is not None:
            fixed_net = RandomConvNet(
                self._net_in_channels(self._buffer.image_size),
                self.width,
                self.depth,
                self.kernel_size,
                self.dilation,
                self.multi_scale,
                3,
                self.activation,
                self.weight_std_min,
                self.weight_std_max,
                self.seed,
                self._buffer.device,
            )
            fixed_net.load_state_dict(state["fixed_net"])
            self._fixed_net = fixed_net

    def _refresh_steps(self) -> int | tuple[int, ...]:
        if self.k_choices == (self.k,):
            return self.k
        return self.k_choices

    def _apply_output_mixture(self, images: torch.Tensor, image_size: int, device: torch.device, seed: int | None) -> torch.Tensor:
        n_fourier = _fraction_count(images.shape[0], self.output_fourier_fraction, minimum_one=True)
        n_noise = _fraction_count(images.shape[0], self.output_noise_fraction, minimum_one=True)
        total = min(images.shape[0], n_fourier + n_noise)
        if total <= 0:
            return images

        gen = torch_generator(device, seed)
        positions = torch.randperm(images.shape[0], device=device, generator=gen)
        offset = 0
        if n_fourier > 0:
            n = min(n_fourier, total - offset)
            idx = positions[offset:offset + n]
            fseed = None if seed is None else seed + 2_000_003
            fresh = FourierTexture().generate(n, image_size, device, seed=fseed)
            images[idx] = _quantize_to_uint8_grid(fresh) if self.quantize else fresh
            offset += n
        if n_noise > 0 and offset < total:
            n = min(n_noise, total - offset)
            idx = positions[offset:offset + n]
            fresh = torch.rand(n, 3, image_size, image_size, device=device, generator=gen)
            images[idx] = _quantize_to_uint8_grid(fresh) if self.quantize else fresh
        return images

    def generate(self, batch_size: int, image_size: int, device, seed: int | None = None, refresh: bool = True) -> torch.Tensor:
        device = torch.device(device)
        fourier_fraction = self._current_fourier_fraction()
        random_color_fraction = self._current_random_color_fraction()
        refresh_fraction = self._current_refresh_fraction()
        update_strength = self._current_update_strength()
        reset_fraction = self._current_reset_fraction()
        if seed is not None:
            local = IRCConvGenerator(
                k=self.k, mode=self.mode, width=self.width, depth=self.depth,
                kernel_size=self.kernel_size, dilation=self.dilation,
                multi_scale=self.multi_scale,
                activation=self.activation,
                weight_std_min=self.weight_std_min,
                weight_std_max=self.weight_std_max,
                buffer_size=max(batch_size, min(self.buffer_size, DEFAULT_IRC_LOCAL_BUFFER_LIMIT)),
                refresh_fraction=1.0, reset_fraction=0.0,
                init_random_color_fraction=self.init_random_color_fraction,
                random_color_fraction=random_color_fraction,
                random_color_fraction_schedule=self.random_color_fraction_schedule,
                fourier_fraction=fourier_fraction,
                fourier_fraction_schedule=self.fourier_fraction_schedule,
                output_fourier_fraction=self.output_fourier_fraction,
                output_noise_fraction=self.output_noise_fraction,
                update_strength=update_strength,
                update_rule=self.update_rule,
                update_output_activation=self.update_output_activation,
                update_strength_schedule=self.update_strength_schedule,
                k_choices=self.k_choices,
                reset_fraction_schedule=self.reset_fraction_schedule,
                quantize=self.quantize,
                horizontal_flip_prob=self.horizontal_flip_prob,
                vertical_flip_prob=self.vertical_flip_prob,
                latent_pixel_channels=self.latent_pixel_channels,
                latent_lowres_channels=self.latent_lowres_channels,
                latent_lowres_scales=self.latent_lowres_scales,
                latent_global_channels=self.latent_global_channels,
                spectral_latent_channels=self.spectral_latent_channels,
                spectral_alpha_min=self.spectral_alpha_min,
                spectral_alpha_max=self.spectral_alpha_max,
                init_source=self.init_source, seed=seed,
            )
            local._ensure_state(image_size, device)
            net = local._make_net(image_size, device, seed)
            local._buffer.refresh(
                net,
                local._refresh_steps(),
                1.0,
                0.0,
                random_color_fraction,
                fourier_fraction,
                update_strength,
                local.update_rule,
                local.update_output_activation,
                local._latent_sampler(image_size, device),
                seed,
            )
            images = local._buffer.sample(batch_size, seed, self.horizontal_flip_prob, self.vertical_flip_prob)
            images = local._apply_output_mixture(images, image_size, device, seed)
            return validate_images(images, batch_size, image_size)

        self._ensure_state(image_size, device)
        if not refresh:
            assert self._buffer is not None
            images = self._buffer.sample(batch_size, horizontal_flip_prob=self.horizontal_flip_prob, vertical_flip_prob=self.vertical_flip_prob)
            images = self._apply_output_mixture(images, image_size, device, self.seed + self._calls + 3_000_003)
            return validate_images(images, batch_size, image_size)

        net_seed = self.seed + self._calls if self.mode == "random_generator_per_batch" else self.seed
        net = self._make_net(image_size, device, net_seed)
        assert self._buffer is not None
        self._buffer.refresh(
            net,
            self._refresh_steps(),
            refresh_fraction,
            reset_fraction,
            random_color_fraction,
            fourier_fraction,
            update_strength,
            self.update_rule,
            self.update_output_activation,
            self._latent_sampler(image_size, device),
            self.seed + self._calls,
        )
        self._calls += 1
        self._images_seen += int(batch_size)
        images = self._buffer.sample(batch_size, horizontal_flip_prob=self.horizontal_flip_prob, vertical_flip_prob=self.vertical_flip_prob)
        images = self._apply_output_mixture(images, image_size, device, self.seed + self._calls + 3_000_003)
        return validate_images(images, batch_size, image_size)
