import random

import torch
from torch import nn
from torch.nn import functional as F


class SpatialProgram:
    """A short random program made only from spatial image operations."""

    # Repeating identity three times gives it 1/2 probability and every active
    # operation 1/6 probability, matching the best IRC8 experiment.
    operations = ("crop_resize", "blur", "identity",
                  "identity", "identity", "block_crossover")

    def __init__(self, random_seed):
        # This RNG fixes the operation sequence and its settings for the whole program.
        generator = random.Random(random_seed)
        self.program = []

        # The 50% stopping chance keeps programs short (two operations on average).
        for _ in range(100):
            operation = generator.choice(self.operations)
            setting = self._sample_setting(operation, generator)
            self.program.append((operation, setting))
            if generator.random() < 0.5:
                break

    @staticmethod
    def _sample_setting(operation, generator):
        # These settings are sampled once, then reused whenever the program runs.
        if operation == "blur":
            return generator.choice((3, 5, 9))
        if operation == "block_crossover":
            return generator.choice((2, 4, 8))
        return None

    @staticmethod
    def _fresh_sources(images, generator):
        batch_size = images.size(0)

        # Independently replace 5% of the images with pixel noise or a solid RGB color.
        fresh_mask = torch.rand(batch_size, device=images.device, dtype=images.dtype, generator=generator) < 0.05
        fresh_indices = torch.nonzero(fresh_mask, as_tuple=True)[0]
        fresh_count = fresh_indices.numel()

        # Shape [N, 1, 1, 1] selects one source type for every replacement image.
        noise_mask = torch.rand(fresh_count, 1, 1, 1, device=images.device, dtype=images.dtype, generator=generator) < 0.5
        noise = torch.rand((fresh_count, *images.shape[1:]), device=images.device, dtype=images.dtype, generator=generator)
        colors = torch.rand(fresh_count, 3, 1, 1, device=images.device, dtype=images.dtype, generator=generator)
        images[fresh_indices] = torch.where(noise_mask, noise, colors)
        return images

    @staticmethod
    def _crop_and_resize(images, generator):
        batch_size = images.size(0)
        # Sample a 50-90% crop and a valid center position independently per image.
        scales = 0.5 + 0.4 * torch.rand(batch_size, device=images.device, dtype=images.dtype, generator=generator)
        translations = (2.0 * torch.rand(batch_size, 2, device=images.device, dtype=images.dtype, generator=generator) - 1.0) * (1.0 - scales).unsqueeze(1)

        # affine_grid maps the output pixels into the selected input region.
        theta = torch.zeros(batch_size, 2, 3, device=images.device, dtype=images.dtype)
        theta[:, 0, 0] = scales
        theta[:, 1, 1] = scales
        theta[:, :, 2] = translations
        grid = F.affine_grid(theta, images.shape, align_corners=False)
        return F.grid_sample(images, grid, mode="bilinear", padding_mode="reflection", align_corners=False)

    @staticmethod
    def _block_crossover(first, second, generator, setting=None):
        batch_size = first.size(0)

        # Smoothly enlarge a small random field to form coherent image regions.
        field = torch.rand(batch_size, 1, setting, setting, device=first.device,
                           dtype=first.dtype, generator=generator)
        field = F.interpolate(field, size=first.shape[-2:], mode="bicubic", align_corners=False)
        threshold = 0.35 + 0.3 * torch.rand(batch_size, 1, 1, 1, device=first.device,
                                            dtype=first.dtype, generator=generator)
        mask = (field > threshold).to(first.dtype)
        return mask * first + (1.0 - mask) * second

    def __call__(self, first, second, random_seed):
        # This RNG provides fresh per-image crops, replacements, and crossover masks.
        generator = torch.Generator(device=first.device).manual_seed(random_seed)
        images = self._fresh_sources(first, generator)

        for operation, setting in self.program:
            if operation == "crop_resize":
                images = self._crop_and_resize(images, generator)
            elif operation == "blur":
                # Reflect-pad before stride-1 pooling to keep the original dimensions.
                padding = setting // 2
                images = F.avg_pool2d(F.pad(images, (padding,) * 4, mode="reflect"), kernel_size=setting, stride=1)
            elif operation == "block_crossover":
                images = self._block_crossover(images, second, generator, setting)

        return images


class ResidualRandomConvolution(nn.Module):  # 883 Parameters
    def __init__(self):
        super().__init__()
        # padding = 1, padding_mode="reflect" -> b | a b c d | c
        # padding = 2, padding_mode="reflect" -> b a | a b c d | d c
        self.layers = nn.Sequential(
            # (B, 3, H, W)
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1, padding_mode="reflect"),
            # (B, 16, H, W)
            nn.LeakyReLU(0.2),
            # (B, 16, H, W)
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding=1, padding_mode="reflect"),
            # (B, 3, H, W)
        )

        for layer in self.layers:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, a=0.2, nonlinearity="leaky_relu")
                nn.init.zeros_(layer.bias)

    def forward(self, images):
        # IRC images are represented in [0, 1].
        change = torch.tanh(self.layers(images - 0.5))
        return torch.clamp(images + 0.1 * change, 0.0, 1.0)


class IRCGenerator:
    """The connected-crossover, crop-resize, and blur IRC generator."""

    def __init__(self, image_size, buffer_size=20_000, source_batch_size=1_000, chunk_size=200, buffer=None):
        self.image_size = image_size
        self.buffer_size = buffer_size
        self.source_batch_size = source_batch_size
        self.chunk_size = chunk_size
        self.device = "cuda"

        if buffer is None:
            self.buffer = torch.empty(buffer_size, 3, image_size, image_size, dtype=torch.uint8).random_(0, 256)
            colors = torch.empty(buffer_size // 2, 3, 1, 1, dtype=torch.uint8).random_(0, 256)
            self.buffer[buffer_size // 2 :] = colors.expand(-1, -1, image_size, image_size)
        else:
            self.buffer = buffer

    def _replace_sources(self):
        reset_count = round(self.buffer_size * 0.0075)
        reset_indices = torch.randperm(self.buffer_size)[:reset_count]
        constant_count = reset_count // 2

        colors = torch.empty(constant_count, 3, 1, 1, dtype=torch.uint8).random_(0, 256)
        self.buffer[reset_indices[:constant_count]] = colors.expand(-1, -1, self.image_size, self.image_size)
        random_indices = reset_indices[constant_count:]
        self.buffer[random_indices] = torch.empty(len(random_indices), 3, self.image_size, self.image_size, dtype=torch.uint8).random_(0, 256)

    @torch.no_grad()
    def generate(self, batch_size):
        # Replace some buffer images
        self._replace_sources()

        # Constructing the spatial program
        program_seed = int(torch.randint(0, 2**31 - 1, ()).item())
        program = SpatialProgram(program_seed)

        # Constructing the random CNN
        convolution = ResidualRandomConvolution().to(self.device).eval()
        
        # Sample random indices for the source images and their partners
        update_indices = torch.randperm(self.buffer_size)[: self.source_batch_size]
        partner_indices = torch.randperm(self.buffer_size)[: self.source_batch_size]

        for chunk_number, start in enumerate(range(0, self.source_batch_size, self.chunk_size)):
            indices = update_indices[start : start + self.chunk_size]
            partners = partner_indices[start : start + self.chunk_size]
            # Transfer compact uint8 images before converting them on the GPU.
            images = self.buffer[indices].to(self.device).float().div_(255.0)
            second_images = self.buffer[partners].to(self.device).float().div_(255.0)

            # Reuse one sampled program and convolution, with fresh masks per chunk.
            images = program(images, second_images, program_seed + chunk_number)
            images = convolution(images)
            self.buffer[indices] = images.mul(255.0).round().to(torch.uint8).cpu()

        # Drop temporary tensors while keeping their CUDA memory available for reuse.
        del images, second_images, program, convolution

        batch_indices = torch.randperm(self.buffer_size)[:batch_size]
        return self.buffer[batch_indices].to(self.device).float().div_(255.0)

    def sample_buffer(self):
        indices = torch.linspace(0, self.buffer_size - 1, steps=16, dtype=torch.long)
        return self.buffer[indices].float() / 255.0
