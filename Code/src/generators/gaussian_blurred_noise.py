import math

import torch
import torch.nn.functional as F


_kernel_cache = {}


def _get_gaussian_kernels(sigma, image_channels, device, dtype):
    key = (sigma, image_channels, device, dtype)
    if key not in _kernel_cache:
        radius = math.ceil(3 * sigma)
        coordinates = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        kernel = torch.exp(-(coordinates ** 2) / (2 * sigma ** 2))
        kernel = kernel / kernel.sum()
        horizontal = kernel.view(1, 1, 1, -1).repeat(image_channels, 1, 1, 1)
        vertical = kernel.view(1, 1, -1, 1).repeat(image_channels, 1, 1, 1)
        _kernel_cache[key] = radius, horizontal, vertical
    return _kernel_cache[key]


def generate_gaussian_blurred_noise(batch_size, image_size, sigma=2.0):
    image_channels = 3
    dtype = torch.float32
    device = "cuda"

    radius, horizontal_kernel, vertical_kernel = _get_gaussian_kernels(sigma, image_channels, device, dtype)

    images = torch.rand(batch_size, image_channels, image_size, image_size, device=device, dtype=dtype)

    images = F.pad(images, (radius, radius, 0, 0), mode="reflect")
    images = F.conv2d(images, horizontal_kernel, groups=image_channels)

    images = F.pad(images, (0, 0, radius, radius), mode="reflect")
    images = F.conv2d(images, vertical_kernel, groups=image_channels)

    return images
