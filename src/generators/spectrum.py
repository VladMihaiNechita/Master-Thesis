# TOO SLOW for a big 100M run
import torch


_frequency_grid_cache = {}


def _get_frequency_grid(image_size, device, dtype):
    key = (image_size, device, dtype)
    if key not in _frequency_grid_cache:
        freq_y = torch.fft.fftfreq(image_size, device=device, dtype=dtype).abs().view(1, image_size, 1)
        freq_x = torch.fft.rfftfreq(image_size, device=device, dtype=dtype).view(1, 1, -1)
        _frequency_grid_cache[key] = freq_x, freq_y
    return _frequency_grid_cache[key]


def generate_spectrum_batch(batch_size, image_size, slope_range=(0.5, 3.5)):
    # Spectrum generator from "Learning to See by Looking at Noise":
    # random noise whose Fourier magnitude follows a natural-image-like power law.

    image_channels = 3
    dtype = torch.float32
    device = 'cuda'

    # Stage 1: start from random image noise.
    images = torch.rand(batch_size, image_channels, image_size, image_size, device=device, dtype=dtype)

    # Stage 2: sample the horizontal and vertical power-law slopes.
    slope_min, slope_max = slope_range
    slope_x = torch.empty(batch_size, 1, 1, device=device, dtype=dtype).uniform_(slope_min, slope_max)
    slope_y = torch.empty(batch_size, 1, 1, device=device, dtype=dtype).uniform_(slope_min, slope_max)

    """
    # A bit worse performance, but more similar to the paper.
    # Stage 2: sample one base slope plus the small official anisotropy offset.
    slope_min, slope_max = slope_range
    slope = torch.empty(batch_size, 1, 1, device=device, dtype=dtype).uniform_(slope_min, slope_max)
    slope_offset = torch.randn(batch_size, 1, 1, device=device, dtype=dtype) * abs(slope_max - slope_min) / 15 / 4
    slope_x = slope + slope_offset
    slope_y = slope - slope_offset
    """

    # Stage 3: reuse the constant real-FFT frequency grids.
    freq_x, freq_y = _get_frequency_grid(image_size, device, dtype)

    # Stage 4: create the target Fourier magnitude 1 / (|fx|^a + |fy|^b).
    frequency = 1.0e-16
    frequency = frequency + freq_x ** slope_x
    frequency = frequency + freq_y ** slope_y

    magnitude = 1.0 / frequency
    magnitude[:, 0, 0] = 0.0
    magnitude = magnitude.unsqueeze(1)
    del frequency

    # Stage 5: keep the random phase, but replace the Fourier magnitude in the half spectrum.
    mean = images.mean(dim=(-2, -1), keepdim=True)
    std = images.std(dim=(-2, -1), keepdim=True).clamp_min(1.0e-6)

    spectrum = torch.fft.rfft2(images - mean)
    spectrum.div_(spectrum.abs().clamp_min_(1.0e-12))
    spectrum.mul_(magnitude)
    images = torch.fft.irfft2(spectrum, s=(image_size, image_size))

    # Stage 6: restore the original per-channel mean and standard deviation.
    images = images - images.mean(dim=(-2, -1), keepdim=True)
    images = images / images.std(dim=(-2, -1), keepdim=True).clamp_min(1.0e-6)
    images = images * std + mean

    # Stage 7: mix color channels with a random orthogonal matrix.
    q, r = torch.linalg.qr(torch.randn(batch_size, image_channels, image_channels, device='cuda', dtype=dtype))
    sign = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    q = q * sign.unsqueeze(-2)
    images = torch.einsum("bij,bjhw->bihw", q, images)

    # Stage 8: normalize each generated image to [0, 1].
    x_min = images.amin(dim=(1, 2, 3), keepdim=True)
    x_max = images.amax(dim=(1, 2, 3), keepdim=True)
    images = (images - x_min) / (x_max - x_min).clamp_min(1.0e-6)

    return images
