import random

import torch
from torch import nn
from torch.nn import functional as F


class SpatialProgram:
    """A short random program made from spatial image operations."""

    operations = ("crop_resize", "blur", "identity",
                  "identity", "identity", "block_crossover")

    def __init__(self, random_seed):
        generator = random.Random(random_seed)
        self.program = []

        for _ in range(100):
            operation = generator.choice(self.operations)
            setting = self._sample_setting(operation, generator)
            self.program.append((operation, setting))
            if generator.random() < 0.5:
                break

        self.uses_crossover = any(operation == "block_crossover"
                                  for operation, _ in self.program)

    @staticmethod
    def _sample_setting(operation, generator):
        if operation == "blur":
            return generator.choice((3, 5, 9))
        if operation == "block_crossover":
            return generator.choice((2, 4, 8))
        return None

    @staticmethod
    def _fresh_sources(images, generator):
        batch_size = images.size(0)
        fresh = torch.rand(batch_size, device=images.device, dtype=images.dtype,
                           generator=generator) < 0.05
        fresh_indices = torch.nonzero(fresh, as_tuple=True)[0]
        fresh_count = fresh_indices.numel()

        noise_mask = torch.rand(fresh_count, 1, 1, 1, device=images.device,
                                dtype=images.dtype, generator=generator) < 0.5
        noise = torch.rand((fresh_count, *images.shape[1:]), device=images.device,
                           dtype=images.dtype, generator=generator)
        colors = torch.rand(fresh_count, 3, 1, 1, device=images.device,
                            dtype=images.dtype, generator=generator)
        images[fresh_indices] = torch.where(noise_mask, noise, colors)
        return images, fresh

    @staticmethod
    def _crop_and_resize(images, generator):
        batch_size = images.size(0)
        scales = 0.5 + 0.4 * torch.rand(batch_size, device=images.device,
                                       dtype=images.dtype, generator=generator)
        translations = (
            2.0 * torch.rand(batch_size, 2, device=images.device,
                             dtype=images.dtype, generator=generator) - 1.0
        ) * (1.0 - scales).unsqueeze(1)

        theta = torch.zeros(batch_size, 2, 3, device=images.device, dtype=images.dtype)
        theta[:, 0, 0] = scales
        theta[:, 1, 1] = scales
        theta[:, :, 2] = translations
        grid = F.affine_grid(theta, images.shape, align_corners=False)
        return F.grid_sample(images, grid, mode="bilinear", padding_mode="reflection",
                             align_corners=False)

    @staticmethod
    def _block_crossover(first, second, generator, setting):
        batch_size = first.size(0)
        field = torch.rand(batch_size, 1, setting, setting, device=first.device,
                           dtype=first.dtype, generator=generator)
        field = F.interpolate(field, size=first.shape[-2:], mode="bicubic",
                              align_corners=False)
        threshold = 0.35 + 0.3 * torch.rand(
            batch_size, 1, 1, 1, device=first.device, dtype=first.dtype,
            generator=generator,
        )
        mask = (field > threshold).to(first.dtype)
        return mask * first + (1.0 - mask) * second

    def __call__(self, first, second, random_seed):
        generator = torch.Generator(device=first.device).manual_seed(random_seed)
        images, fresh = self._fresh_sources(first, generator)

        for operation, setting in self.program:
            if operation == "crop_resize":
                images = self._crop_and_resize(images, generator)
            elif operation == "blur":
                padding = setting // 2
                images = F.avg_pool2d(
                    F.pad(images, (padding,) * 4, mode="reflect"),
                    kernel_size=setting,
                    stride=1,
                )
            elif operation == "block_crossover":
                images = self._block_crossover(images, second, generator, setting)

        return images, fresh


class ResidualRandomConvolution(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 3, kernel_size=3, padding=1, padding_mode="reflect"),
        )

        for layer in self.layers:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, a=0.2, nonlinearity="leaky_relu")
                nn.init.zeros_(layer.bias)

    def forward(self, images):
        change = torch.tanh(self.layers(images - 0.5))
        return torch.clamp(images + 0.1 * change, 0.0, 1.0)


class IRCGenerator:
    """Independent IRC generator with adaptive depth-based sampling."""

    def __init__(self, image_size, buffer_size=40_000, source_batch_size=2_000,
                 chunk_size=200, buffer=None, buffer_device="cpu", depths=None,
                 candidate_pool_size=20_000):
        self.image_size = image_size
        self.buffer_size = buffer_size
        self.source_batch_size = source_batch_size
        self.chunk_size = chunk_size
        self.device = "cuda"
        self.buffer_device = buffer_device
        self.candidate_pool_size = candidate_pool_size
        self.min_depth = 0

        if buffer is None:
            self.buffer = torch.empty(
                buffer_size, 3, image_size, image_size,
                device=buffer_device, dtype=torch.uint8,
            ).random_(0, 256)
            colors = torch.empty(
                buffer_size // 2, 3, 1, 1,
                device=buffer_device, dtype=torch.uint8,
            ).random_(0, 256)
            self.buffer[buffer_size // 2:] = colors.expand(
                -1, -1, image_size, image_size,
            )
        else:
            self.buffer = buffer.to(buffer_device)

        if depths is None:
            self.depths = torch.zeros(buffer_size, device=buffer_device, dtype=torch.int32)
        else:
            self.depths = depths.to(device=buffer_device, dtype=torch.int32)

    def _replace_sources(self):
        reset_count = round(self.buffer_size * 0.0075)
        reset_indices = torch.randperm(
            self.buffer_size, device=self.buffer_device,
        )[:reset_count]
        constant_count = reset_count // 2

        colors = torch.empty(
            constant_count, 3, 1, 1,
            device=self.buffer_device, dtype=torch.uint8,
        ).random_(0, 256)
        self.buffer[reset_indices[:constant_count]] = colors.expand(
            -1, -1, self.image_size, self.image_size,
        )

        random_indices = reset_indices[constant_count:]
        self.buffer[random_indices] = torch.empty(
            len(random_indices), 3, self.image_size, self.image_size,
            device=self.buffer_device, dtype=torch.uint8,
        ).random_(0, 256)
        self.depths[reset_indices] = 0

    def _eligible_indices(self, min_depth, max_depth):
        eligible = self.depths >= min_depth
        if max_depth is not None:
            eligible &= self.depths <= max_depth
        return torch.nonzero(eligible, as_tuple=True)[0]

    def sample_indices(self, batch_size, min_depth=0, max_depth=None):
        """Sample buffer indices from an inclusive depth range."""
        eligible_indices = self._eligible_indices(min_depth, max_depth)
        if eligible_indices.numel() < batch_size:
            raise ValueError(
                f"Requested {batch_size} images, but only {eligible_indices.numel()} "
                "are available in that depth range."
            )
        order = torch.randperm(
            eligible_indices.numel(), device=self.buffer_device,
        )[:batch_size]
        return eligible_indices[order]

    def _select_min_depth(self):
        # The candidate-pool-th largest depth is the largest valid threshold.
        rank = self.buffer_size - self.candidate_pool_size + 1
        return int(torch.kthvalue(self.depths, rank).values.item())

    @torch.no_grad()
    def _update_buffer(self):
        self._replace_sources()
        update_indices = torch.randperm(
            self.buffer_size, device=self.buffer_device,
        )[:self.source_batch_size]
        partner_indices = torch.randperm(
            self.buffer_size, device=self.buffer_device,
        )[:self.source_batch_size]

        for start in range(0, self.source_batch_size, self.chunk_size):
            program_seed = int(torch.randint(0, 2**31 - 1, ()).item())
            program = SpatialProgram(program_seed)
            # Give every image chunk an independent residual transformation.
            convolution = ResidualRandomConvolution().to(self.device).eval()
            indices = update_indices[start:start + self.chunk_size]
            partners = partner_indices[start:start + self.chunk_size]

            images = self.buffer[indices].to(self.device).float().div_(255.0)
            second_images = self.buffer[partners].to(self.device).float().div_(255.0)
            images, fresh = program(images, second_images, program_seed)

            source_depths = self.depths[indices].clone()
            source_depths[fresh.to(self.buffer_device)] = 0
            if program.uses_crossover:
                source_depths = torch.maximum(source_depths, self.depths[partners])

            images = convolution(images)
            self.buffer[indices] = images.mul(255.0).round().to(
                device=self.buffer_device, dtype=torch.uint8,
            )
            self.depths[indices] = source_depths + 1

        del images, second_images, program, convolution

    @torch.no_grad()
    def prepare_batch(self, batch_size):
        self._update_buffer()
        self.min_depth = self._select_min_depth()
        return self.sample_indices(batch_size, min_depth=self.min_depth)

    @torch.no_grad()
    def sample(self, batch_indices):
        return self.buffer[batch_indices].to(self.device).float().div_(255.0)

    def sample_by_depth(self, batch_size, min_depth=0, max_depth=None):
        return self.sample(self.sample_indices(batch_size, min_depth, max_depth))

    def generate(self, batch_size):
        return self.sample(self.prepare_batch(batch_size))

    def sample_buffer(self):
        indices = torch.linspace(
            0, self.buffer_size - 1, steps=16,
            device=self.buffer_device, dtype=torch.long,
        )
        return (self.buffer[indices].float() / 255.0).cpu()
