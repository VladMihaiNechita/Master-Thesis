import sys
from importlib import import_module
from pathlib import Path

sys.dont_write_bytecode = True

import torch
from torchvision.utils import save_image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.util import set_seed


SNAPSHOT_STEPS = (8, 32, 64, 128, 256, 512, 1024)
SAMPLE_COUNT = 4000


def main(generator_name):
    set_seed()
    generator_class = import_module(f"src.generators.{generator_name}").IRCGenerator
    generator = generator_class(image_size=224, buffer_device="cuda")

    sample_generator = torch.Generator(device=generator.buffer_device).manual_seed(1729)
    sample_indices = torch.randperm(generator.buffer_size, device=generator.buffer_device,
                                    generator=sample_generator)[:SAMPLE_COUNT]
    output_directory = PROJECT_ROOT / "Generator_Images" / f"{generator_name}_visual_test_Z"
    output_directory.mkdir(parents=True, exist_ok=True)

    for step in range(max(SNAPSHOT_STEPS) + 1):
        if step in SNAPSHOT_STEPS:
            print(f"Step {step} / {max(SNAPSHOT_STEPS)}")

        if step in SNAPSHOT_STEPS:
            images = generator.sample_by_depth(SAMPLE_COUNT, min_depth=1).cpu()
            save_image(images[:100], output_directory / f"step_{step:03d}_{4000*step:08d}.png", nrow=10)
        if step < max(SNAPSHOT_STEPS):
            generator.prepare_batch(1)

    print(f"Saved visual test to {output_directory}")

main(generator_name="irc_new")
