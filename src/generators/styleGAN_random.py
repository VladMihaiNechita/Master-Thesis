import math

import torch
import torch.nn.functional as F


def _filter_2d(x, resample_filter, padding, gain=1.0):
    padding_x0, padding_x1, padding_y0, padding_y1 = padding
    x = F.pad(x, (padding_x0, padding_x1, padding_y0, padding_y1))

    weight = (resample_filter * gain).to(x.dtype)
    weight = weight.view(1, 1, *weight.shape).repeat(x.shape[1], 1, 1, 1)
    return F.conv2d(x, weight, groups=x.shape[1])


def _upsample_2d(x, resample_filter):
    # StyleGAN2 inserts zeros and applies its [1, 3, 3, 1] low-pass filter.
    batch_size, channels, height, width = x.shape
    upsampled = x.new_zeros(batch_size, channels, height * 2, width * 2)
    upsampled[:, :, ::2, ::2] = x
    return _filter_2d(upsampled, resample_filter, padding=(2, 1, 2, 1), gain=4.0)


def _modulated_conv2d(x, weight, styles, demodulate=True, up=False, resample_filter=None):
    batch_size, in_channels, height, width = x.shape
    out_channels, _, kernel_height, kernel_width = weight.shape

    per_image_weight = weight.unsqueeze(0) * styles.view(batch_size, 1, in_channels, 1, 1)
    if demodulate:
        demodulation = torch.rsqrt(per_image_weight.square().sum(dim=(2, 3, 4)) + 1.0e-8)
        per_image_weight = per_image_weight * demodulation.view(batch_size, out_channels, 1, 1, 1)

    x = x.reshape(1, batch_size * in_channels, height, width)

    if up:
        # This is the grouped transpose-convolution path used by StyleGAN2.
        per_image_weight = per_image_weight.transpose(1, 2)
        per_image_weight = per_image_weight.reshape(batch_size * in_channels, out_channels, kernel_height, kernel_width)
        x = F.conv_transpose2d(x, per_image_weight.flip((2, 3)), stride=2, groups=batch_size)
        x = x.reshape(batch_size, out_channels, x.shape[-2], x.shape[-1])
        return _filter_2d(x, resample_filter, padding=(1, 1, 1, 1), gain=4.0)

    per_image_weight = per_image_weight.reshape(batch_size * out_channels, in_channels, kernel_height, kernel_width)
    x = F.conv2d(x, per_image_weight, padding=kernel_width // 2, groups=batch_size)
    return x.reshape(batch_size, out_channels, height, width)


class StyleGANRandomGenerator:
    def __init__(self, image_size, latent_size=512, max_channels=512, device=None):
        # Untrained StyleGAN2 synthesis network used for StyleGAN-Random in
        # "Learning to See by Looking at Noise".
        self.image_size = image_size
        self.latent_size = latent_size
        self.max_channels = max_channels
        self.dtype = torch.float32
        self.device = torch.device("cuda" if device is None else device)

        # StyleGAN2 operates at powers of two; MAE receives the final resized image.
        self.internal_size = max(4, 2 ** math.ceil(math.log2(image_size)))
        self.block_resolutions = [2 ** power for power in range(2, int(math.log2(self.internal_size)) + 1)]
        self.channels = {resolution: min(32768 // resolution, max_channels) for resolution in self.block_resolutions}

        # One style per convolution plus the final ToRGB style, as in StyleGAN2.
        self.num_ws = 2 * len(self.block_resolutions)
        self.reset()

    def reset(self):
        channels_4 = self.channels[4]
        self.constant = torch.randn(1, channels_4, 4, 4, device=self.device, dtype=self.dtype)

        filter_1d = torch.tensor([1, 3, 3, 1], device=self.device, dtype=torch.float32)
        self.resample_filter = torch.outer(filter_1d, filter_1d)
        self.resample_filter /= self.resample_filter.sum()

        self.conv_specs = []
        for resolution in self.block_resolutions:
            out_channels = self.channels[resolution]
            if resolution == 4:
                self.conv_specs.append((out_channels, out_channels, resolution, False))
            else:
                in_channels = self.channels[resolution // 2]
                self.conv_specs.append((in_channels, out_channels, resolution, True))
                self.conv_specs.append((out_channels, out_channels, resolution, False))

        self.conv_weights = []
        self.conv_biases = []
        self.affine_weights = []
        self.affine_biases = []

        for in_channels, out_channels, _, _ in self.conv_specs:
            self.conv_weights.append(torch.randn(out_channels, in_channels, 3, 3, device=self.device, dtype=self.dtype))
            self.conv_biases.append(torch.zeros(out_channels, device=self.device, dtype=self.dtype))
            self.affine_weights.append(torch.randn(self.latent_size, in_channels, device=self.device, dtype=self.dtype)
                                        / math.sqrt(self.latent_size))
            self.affine_biases.append(torch.ones(in_channels, device=self.device, dtype=self.dtype))

        self.to_rgb_weights = {}
        self.to_rgb_biases = {}
        self.to_rgb_affine_weights = {}
        self.to_rgb_affine_biases = {}

        for resolution in self.block_resolutions:
            channels = self.channels[resolution]
            self.to_rgb_weights[resolution] = torch.randn(3, channels, 1, 1, device=self.device, dtype=self.dtype)
            self.to_rgb_biases[resolution] = torch.zeros(3, device=self.device, dtype=self.dtype)
            self.to_rgb_affine_weights[resolution] = (torch.randn(self.latent_size, channels, device=self.device, dtype=self.dtype)
                                                       / math.sqrt(self.latent_size))
            self.to_rgb_affine_biases[resolution] = torch.ones(channels, device=self.device, dtype=self.dtype)

    def generate(self, batch_size, chunk_size=16, reset_every=None):
        images = []
        generated_since_reset = 0
        activation_gain = math.sqrt(2.0)

        with torch.inference_mode():
            for chunk_start in range(0, batch_size, chunk_size):
                if reset_every is not None and generated_since_reset >= reset_every:
                    self.reset()
                    generated_since_reset = 0

                current_batch_size = min(chunk_size, batch_size - chunk_start)

                # The paper bypasses StyleGAN2's mapping network and samples W directly.
                ws = torch.randn(current_batch_size, self.num_ws, self.latent_size, device=self.device, dtype=self.dtype)
                ws = F.leaky_relu(ws, negative_slope=0.2)

                x = self.constant.repeat(current_batch_size, 1, 1, 1)
                image = None
                conv_index = 0
                w_index = 0

                for resolution in self.block_resolutions:
                    num_block_convs = 1 if resolution == 4 else 2

                    for _ in range(num_block_convs):
                        in_channels, _, _, up = self.conv_specs[conv_index]
                        style = (ws[:, w_index] @ self.affine_weights[conv_index] + self.affine_biases[conv_index])
                        x = _modulated_conv2d(x, self.conv_weights[conv_index], style, up=up, resample_filter=self.resample_filter)

                        bias = self.conv_biases[conv_index].view(1, -1, 1, 1)
                        x = F.relu(x + bias) * activation_gain
                        conv_index += 1
                        w_index += 1

                    # ToRGB consumes the next style, which the following block reuses.
                    channels = self.channels[resolution]
                    rgb_style = ws[:, w_index] @ self.to_rgb_affine_weights[resolution] + self.to_rgb_affine_biases[resolution]
                    rgb_style /= math.sqrt(channels)
                    rgb = _modulated_conv2d(x, self.to_rgb_weights[resolution], rgb_style, demodulate=False)
                    rgb = rgb + self.to_rgb_biases[resolution].view(1, -1, 1, 1)

                    if image is None:
                        image = rgb
                    else:
                        image = _upsample_2d(image, self.resample_filter) + rgb

                if image.shape[-1] != self.image_size:
                    image = F.interpolate(image, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)

                # Match the paper's per-image, per-channel color normalization.
                mean = image.mean(dim=(-2, -1), keepdim=True)
                std = image.std(dim=(-2, -1), keepdim=True).clamp_min(1.0e-6)
                image = (image - mean) / std

                target_mean = torch.empty(current_batch_size, 3, 1, 1, device=self.device, dtype=self.dtype)
                target_mean[:, 0].normal_(0.483, 0.145)
                target_mean[:, 1].normal_(0.455, 0.142)
                target_mean[:, 2].normal_(0.401, 0.161)

                target_std = torch.empty(current_batch_size, 3, 1, 1, device=self.device, dtype=self.dtype)
                target_std[:, 0].normal_(0.219, 0.063)
                target_std[:, 1].normal_(0.213, 0.062)
                target_std[:, 2].normal_(0.213, 0.069)

                image = (image * target_std + target_mean).clamp(0, 1)
                images.append(image)
                generated_since_reset += current_batch_size

        return torch.cat(images, dim=0)


def generate_stylegan_random_batch(batch_size, image_size, latent_size=512, max_channels=512, chunk_size=16, 
                                   reset_every=400, device=None):
    
    generator = StyleGANRandomGenerator(image_size=image_size, latent_size=latent_size, 
                                        max_channels=max_channels, device=device)
    return generator.generate(batch_size=batch_size, chunk_size=chunk_size, reset_every=reset_every)