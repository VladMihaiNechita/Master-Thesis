from __future__ import annotations

from typing import Any

from src.irc_vit.config import DEFAULT_GENERATOR_NAME
from src.irc_vit.generators.blur import GaussianBlurNoise
from src.irc_vit.generators.fourier import FourierTexture
from src.irc_vit.generators.irc_conv import IRCConvGenerator
from src.irc_vit.generators.noise import IIDRGBNoise
from src.irc_vit.generators.shapes import SimpleShapes


def build_generator(config: dict[str, Any]):
    name = config.get("name", DEFAULT_GENERATOR_NAME)
    params = dict(config.get("params", {}))
    if name == "iid_rgb_noise":
        return IIDRGBNoise(**params)
    if name == "gaussian_blur_noise":
        return GaussianBlurNoise(**params)
    if name == "fourier_texture":
        return FourierTexture(**params)
    if name == "simple_shapes":
        return SimpleShapes(**params)
    if name == "irc_conv":
        return IRCConvGenerator(**params)
    raise ValueError(f"Unknown generator: {name}")


__all__ = [
    "build_generator",
    "IIDRGBNoise",
    "GaussianBlurNoise",
    "FourierTexture",
    "SimpleShapes",
    "IRCConvGenerator",
]
