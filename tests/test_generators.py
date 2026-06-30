from __future__ import annotations

import torch

from src.irc_vit.generators import build_generator
from src.irc_vit.generators.controls import patch_shuffle, phase_randomize, pixel_shuffle


def assert_images(images: torch.Tensor, batch_size: int = 4, image_size: int = 32) -> None:
    assert images.shape == (batch_size, 3, image_size, image_size)
    assert images.dtype == torch.float32
    assert not images.requires_grad
    assert float(images.min()) >= 0.0
    assert float(images.max()) <= 1.0


def test_generator_contract_and_seed_determinism() -> None:
    configs = [
        {"name": "iid_rgb_noise", "params": {}},
        {"name": "gaussian_blur_noise", "params": {"sigma": 1.5}},
        {"name": "fourier_texture", "params": {"components": 8}},
        {"name": "simple_shapes", "params": {"shape_count_max": 2}},
        {"name": "irc_conv", "params": {"k": 2, "width": 8, "depth": 3, "buffer_size": 16}},
    ]
    for cfg in configs:
        gen = build_generator(cfg)
        a = gen.generate(4, 32, "cpu", seed=123)
        b = gen.generate(4, 32, "cpu", seed=123)
        assert_images(a)
        assert torch.allclose(a, b), cfg["name"]


def test_irc_conv_fourier_injection_contract() -> None:
    gen = build_generator({
        "name": "irc_conv",
        "params": {
            "k": 1,
            "width": 4,
            "depth": 3,
            "buffer_size": 16,
            "refresh_fraction": 0.25,
            "reset_fraction": 0.0,
            "fourier_fraction": 0.25,
        },
    })
    images = gen.generate(4, 32, "cpu", seed=321)
    assert_images(images)


def test_irc_conv_random_color_init_and_injection_contract() -> None:
    gen = build_generator({
        "name": "irc_conv",
        "params": {
            "k": 1,
            "width": 4,
            "depth": 3,
            "buffer_size": 16,
            "refresh_fraction": 0.25,
            "reset_fraction": 0.0,
            "fourier_fraction": 0.0,
            "init_random_color_fraction": 0.25,
            "random_color_fraction": 0.25,
            "quantize": True,
        },
    })
    gen._ensure_state(32, torch.device("cpu"))
    assert gen._buffer is not None
    spatial_std = gen._buffer.images.flatten(2).std(dim=2)
    solid_count = int((spatial_std == 0).all(dim=1).sum().item())
    assert solid_count >= 4
    images = gen.generate(4, 32, "cpu", seed=321)
    assert_images(images)
    assert torch.allclose(images, (images * 255.0).round() / 255.0)


def test_irc_conv_state_roundtrip_preserves_buffer_and_progress() -> None:
    cfg = {
        "name": "irc_conv",
        "params": {
            "k": 1,
            "width": 4,
            "depth": 3,
            "buffer_size": 16,
            "refresh_fraction": 0.25,
            "reset_fraction": 0.0,
            "fourier_fraction": 0.0,
            "quantize": True,
            "seed": 1234,
        },
    }
    gen = build_generator(cfg)
    gen.generate(4, 32, "cpu")
    assert gen._buffer is not None

    restored = build_generator(cfg)
    restored.load_state_dict(gen.state_dict(), device="cpu")

    assert restored._calls == gen._calls
    assert restored._images_seen == gen._images_seen
    assert restored._buffer is not None
    assert torch.equal(
        (restored._buffer.images * 255.0).round().to(torch.uint8),
        (gen._buffer.images * 255.0).round().to(torch.uint8),
    )

    torch.manual_seed(99)
    images = gen.generate(4, 32, "cpu", refresh=False)
    torch.manual_seed(99)
    restored_images = restored.generate(4, 32, "cpu", refresh=False)
    assert torch.allclose(images, restored_images)


def test_irc_conv_quantized_flip_contract() -> None:
    gen = build_generator({
        "name": "irc_conv",
        "params": {
            "k": 1,
            "width": 4,
            "depth": 3,
            "buffer_size": 16,
            "refresh_fraction": 0.25,
            "reset_fraction": 0.0,
            "fourier_fraction": 0.25,
            "quantize": True,
            "horizontal_flip_prob": 1.0,
        },
    })
    images = gen.generate(4, 32, "cpu", seed=456)
    assert_images(images)
    assert torch.allclose(images, (images * 255.0).round() / 255.0)


def test_irc_conv_output_mixture_and_variable_k_contract() -> None:
    gen = build_generator({
        "name": "irc_conv",
        "params": {
            "k": 2,
            "k_choices": [1, 2, 3],
            "width": 4,
            "depth": 3,
            "buffer_size": 16,
            "refresh_fraction": 0.25,
            "reset_fraction": 0.0,
            "fourier_fraction": 0.0,
            "output_fourier_fraction": 0.25,
            "output_noise_fraction": 0.25,
            "update_strength": 0.15,
            "quantize": True,
        },
    })
    images = gen.generate(8, 32, "cpu", seed=654)
    assert_images(images, batch_size=8)
    assert torch.allclose(images, (images * 255.0).round() / 255.0)
    assert gen.k_choices == (1, 2, 3)


def test_irc_conv_latent_update_contract_and_determinism() -> None:
    cfg = {
        "name": "irc_conv",
        "params": {
            "k": 2,
            "width": 6,
            "depth": 3,
            "kernel_size": 5,
            "dilation": 1,
            "weight_std_min": 0.04,
            "weight_std_max": 0.12,
            "buffer_size": 16,
            "refresh_fraction": 0.25,
            "reset_fraction": 0.0,
            "fourier_fraction": 0.0,
            "latent_pixel_channels": 2,
            "latent_lowres_channels": 1,
            "latent_lowres_scales": [4, 8],
            "latent_global_channels": 1,
            "spectral_latent_channels": 2,
            "spectral_alpha_min": 0.7,
            "spectral_alpha_max": 2.0,
            "horizontal_flip_prob": 0.5,
            "vertical_flip_prob": 0.5,
            "quantize": True,
        },
    }
    gen_a = build_generator(cfg)
    gen_b = build_generator(cfg)
    images_a = gen_a.generate(4, 32, "cpu", seed=2468)
    images_b = gen_b.generate(4, 32, "cpu", seed=2468)
    assert_images(images_a)
    assert torch.allclose(images_a, images_b)
    assert torch.allclose(images_a, (images_a * 255.0).round() / 255.0)


def test_irc_conv_multi_scale_contract_and_determinism() -> None:
    cfg = {
        "name": "irc_conv",
        "params": {
            "k": 2,
            "width": 7,
            "depth": 3,
            "multi_scale": True,
            "buffer_size": 16,
            "refresh_fraction": 0.25,
            "reset_fraction": 0.0,
            "fourier_fraction": 0.05,
            "quantize": True,
        },
    }
    gen_a = build_generator(cfg)
    gen_b = build_generator(cfg)
    images_a = gen_a.generate(4, 32, "cpu", seed=1357)
    images_b = gen_b.generate(4, 32, "cpu", seed=1357)
    assert_images(images_a)
    assert torch.allclose(images_a, images_b)
    assert torch.allclose(images_a, (images_a * 255.0).round() / 255.0)
    gen_a._ensure_state(32, torch.device("cpu"))
    net = gen_a._make_net(32, torch.device("cpu"), 1357)
    scales = [
        getattr(branch, "_irc_weight_std_scale", None)
        for module in net.modules()
        if module.__class__.__name__ == "MultiScaleRandomConv"
        for branch in module.branches
    ]
    assert {round(float(scale), 2) for scale in scales} == {0.6, 1.0, 3.0}


def test_irc_conv_tanh01_update_output_activation_contract() -> None:
    cfg = {
        "name": "irc_conv",
        "params": {
            "k": 2,
            "width": 4,
            "depth": 3,
            "buffer_size": 16,
            "refresh_fraction": 0.25,
            "reset_fraction": 0.0,
            "fourier_fraction": 0.05,
            "update_output_activation": "tanh01",
            "quantize": True,
        },
    }
    gen_a = build_generator(cfg)
    gen_b = build_generator(cfg)
    images_a = gen_a.generate(4, 32, "cpu", seed=8642)
    images_b = gen_b.generate(4, 32, "cpu", seed=8642)
    assert_images(images_a)
    assert torch.allclose(images_a, images_b)
    assert torch.allclose(images_a, (images_a * 255.0).round() / 255.0)


def test_irc_conv_residual_update_rule_contract_and_determinism() -> None:
    cfg = {
        "name": "irc_conv",
        "params": {
            "k": 2,
            "width": 4,
            "depth": 3,
            "buffer_size": 16,
            "refresh_fraction": 0.25,
            "reset_fraction": 0.0,
            "fourier_fraction": 0.05,
            "update_rule": "residual",
            "update_strength": 0.15,
            "quantize": True,
        },
    }
    gen_a = build_generator(cfg)
    gen_b = build_generator(cfg)
    images_a = gen_a.generate(4, 32, "cpu", seed=9753)
    images_b = gen_b.generate(4, 32, "cpu", seed=9753)
    assert_images(images_a)
    assert torch.allclose(images_a, images_b)
    assert torch.allclose(images_a, (images_a * 255.0).round() / 255.0)


def test_irc_conv_fourier_switch_step() -> None:
    gen = build_generator({
        "name": "irc_conv",
        "params": {
            "k": 1,
            "width": 4,
            "depth": 3,
            "buffer_size": 16,
            "fourier_fraction": 0.25,
            "fourier_fraction_after": 0.05,
            "fourier_fraction_switch_step": 2,
        },
    })
    assert gen._current_fourier_fraction() == 0.25
    gen.generate(4, 32, "cpu")
    assert gen._current_fourier_fraction() == 0.25
    gen.generate(4, 32, "cpu")
    assert gen._current_fourier_fraction() == 0.05
    gen.set_step(0)
    assert gen._current_fourier_fraction() == 0.25
    gen.set_step(2)
    images = gen.generate(4, 32, "cpu", seed=789)
    assert_images(images)
    assert gen._current_fourier_fraction() == 0.05


def test_irc_conv_late_update_and_reset_switches() -> None:
    gen = build_generator({
        "name": "irc_conv",
        "params": {
            "k": 1,
            "width": 4,
            "depth": 3,
            "buffer_size": 16,
            "reset_fraction": 0.001,
            "reset_fraction_after": 0.005,
            "reset_fraction_switch_step": 2,
            "update_strength": 0.3,
            "update_strength_after": 0.15,
            "update_strength_switch_step": 2,
        },
    })
    assert gen._current_update_strength() == 0.3
    assert gen._current_reset_fraction() == 0.001
    gen.set_step(2)
    assert gen._current_update_strength() == 0.15
    assert gen._current_reset_fraction() == 0.005
    images = gen.generate(4, 32, "cpu", seed=987)
    assert_images(images)


def test_irc_conv_linear_schedules() -> None:
    gen = build_generator({
        "name": "irc_conv",
        "params": {
            "k": 1,
            "width": 4,
            "depth": 3,
            "buffer_size": 16,
            "fourier_fraction": 0.05,
            "fourier_fraction_after": 0.05,
            "fourier_fraction_switch_step": 10,
            "fourier_fraction_schedule": "linear",
            "reset_fraction": 0.01,
            "reset_fraction_after": 0.0001,
            "reset_fraction_switch_step": 10,
            "reset_fraction_schedule": "linear",
        },
    })
    assert abs(gen._current_fourier_fraction() - 0.05) < 1e-8
    assert abs(gen._current_reset_fraction() - 0.01) < 1e-8
    gen.set_step(5)
    assert abs(gen._current_fourier_fraction() - 0.05) < 1e-8
    assert abs(gen._current_reset_fraction() - 0.00505) < 1e-8
    gen.set_step(10)
    assert abs(gen._current_fourier_fraction() - 0.05) < 1e-8
    assert abs(gen._current_reset_fraction() - 0.0001) < 1e-8
    images = gen.generate(4, 32, "cpu", seed=4321)
    assert_images(images)


def test_irc_conv_image_based_linear_schedules() -> None:
    gen = build_generator({
        "name": "irc_conv",
        "params": {
            "k": 1,
            "width": 4,
            "depth": 3,
            "buffer_size": 16,
            "reset_fraction": 0.001,
            "reset_fraction_after": 0.0001,
            "reset_fraction_switch_images": 100,
            "reset_fraction_schedule": "linear",
            "refresh_fraction": 0.25,
            "refresh_fraction_after": 0.05,
            "refresh_fraction_switch_images": 100,
            "refresh_fraction_schedule": "linear",
            "random_color_fraction": 0.01,
            "random_color_fraction_after": 0.001,
            "random_color_fraction_switch_images": 100,
            "random_color_fraction_schedule": "linear",
            "update_strength": 0.3,
            "update_strength_after": 0.1,
            "update_strength_switch_images": 100,
            "update_strength_schedule": "linear",
        },
    })
    assert abs(gen._current_reset_fraction() - 0.001) < 1e-8
    assert abs(gen._current_refresh_fraction() - 0.25) < 1e-8
    assert abs(gen._current_random_color_fraction() - 0.01) < 1e-8
    assert abs(gen._current_update_strength() - 0.3) < 1e-8
    gen.set_progress(step=99, images_seen=50)
    assert abs(gen._current_reset_fraction() - 0.00055) < 1e-8
    assert abs(gen._current_refresh_fraction() - 0.15) < 1e-8
    assert abs(gen._current_random_color_fraction() - 0.0055) < 1e-8
    assert abs(gen._current_update_strength() - 0.2) < 1e-8
    gen.set_progress(step=0, images_seen=100)
    assert abs(gen._current_reset_fraction() - 0.0001) < 1e-8
    assert abs(gen._current_refresh_fraction() - 0.05) < 1e-8
    assert abs(gen._current_random_color_fraction() - 0.001) < 1e-8
    assert abs(gen._current_update_strength() - 0.1) < 1e-8
    images = gen.generate(4, 32, "cpu")
    assert_images(images)
    assert gen._images_seen == 104


def test_irc_conv_refresh_false() -> None:
    gen = build_generator({
        "name": "irc_conv",
        "params": {
            "k": 1,
            "width": 4,
            "depth": 3,
            "buffer_size": 16,
        },
    })
    # Run once to initialize state and calls count
    gen.generate(4, 32, "cpu")
    calls_before = gen._calls
    images_seen_before = gen._images_seen
    # Now generate with refresh=False
    images = gen.generate(4, 32, "cpu", refresh=False)
    # Check that calls is NOT incremented
    assert gen._calls == calls_before
    assert gen._images_seen == images_seen_before
    assert_images(images)


def test_control_transforms_keep_shape_and_range() -> None:
    x = torch.rand(2, 3, 32, 32)
    for y in (pixel_shuffle(x, seed=0), patch_shuffle(x, patch_size=8, seed=0), phase_randomize(x, seed=0)):
        assert y.shape == x.shape
        assert float(y.min()) >= 0.0
        assert float(y.max()) <= 1.0
