import math

import torch


class IJEPAMaskSampler:
    def __init__(self, config):
        self.grid_size = config["image_size"] // config["patch_size"]
        self.num_patches = self.grid_size ** 2
        self.context_mask_scale = config["context_mask_scale"]
        self.target_mask_scale = config["target_mask_scale"]
        self.target_aspect_ratio = config["target_aspect_ratio"]
        self.num_target_blocks = config["num_target_blocks"]
        self.min_context_patches = config["min_context_patches"]
        self.max_block_size = (
            self.grid_size if config.get("allow_full_grid_blocks", True) else self.grid_size - 1
        )

    def _sample_block_size(self, scale, aspect_ratio):
        area_scale = torch.empty(()).uniform_(*scale).item()
        ratio = torch.empty(()).uniform_(*aspect_ratio).item()
        area = self.num_patches * area_scale

        # Match I-JEPA by keeping sampled blocks below the full patch grid.
        height = min(self.max_block_size, max(1, round(math.sqrt(area * ratio))))
        width = min(self.max_block_size, max(1, round(math.sqrt(area / ratio))))
        return height, width

    def _sample_blocks(self, batch_size, block_size, device):
        height, width = block_size
        top = torch.randint(0, self.grid_size - height + 1, (batch_size, 1), device=device)
        left = torch.randint(0, self.grid_size - width + 1, (batch_size, 1), device=device)

        rows = torch.arange(height, device=device).unsqueeze(1) * self.grid_size
        columns = torch.arange(width, device=device)
        relative_indices = (rows + columns).reshape(1, -1)
        return top * self.grid_size + left + relative_indices

    def __call__(self, batch_size, device):
        target_size = self._sample_block_size(self.target_mask_scale, self.target_aspect_ratio)
        context_size = self._sample_block_size(self.context_mask_scale, (1.0, 1.0))

        target_indices = [self._sample_blocks(batch_size, target_size, device)
                          for _ in range(self.num_target_blocks)]

        # Sample a large context block and remove every target patch from it.
        while True:
            context_block = self._sample_blocks(batch_size, context_size, device)
            context_mask = torch.zeros(batch_size, self.num_patches, dtype=torch.bool, device=device)
            context_mask.scatter_(1, context_block, True)
            for target in target_indices:
                context_mask.scatter_(1, target, False)

            context_lengths = context_mask.sum(dim=1)
            if context_lengths.min().item() >= self.min_context_patches:
                break

        # Transformers require one sequence length, so keep the batch-wide minimum.
        context_length = context_lengths.min().item()
        scores = torch.rand(batch_size, self.num_patches, device=device)
        scores.masked_fill_(~context_mask, 2.0)
        context_indices = scores.argsort(dim=1)[:, :context_length]
        context_indices = context_indices.sort(dim=1).values

        return context_indices, target_indices
