from datetime import datetime
from pathlib import Path
import math
import random

import numpy as np
import torch
import wandb
from torchvision.utils import make_grid

# from ..evaluation_util import log_pending_eval_results
from ....generators.gaussian_blurred_noise import generate_gaussian_blurred_noise
from ....generators.irc import IRCGenerator
from ....generators.irc_new import IRCGenerator as IRCNewGenerator
from ....generators.real_video import generate_from_real_video
from ....generators.spectrum import generate_spectrum_batch
from ....generators.styleGAN_random import generate_stylegan_random_batch
from ....models import MaskedAutoencoderViT


PROJECT_ROOT = Path(__file__).resolve().parents[4]
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINTS_DIR.mkdir(exist_ok=True)


def get_resume_checkpoint_path(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    return checkpoint_path.with_name(f"{checkpoint_path.stem}_resume{checkpoint_path.suffix}")


def save_checkpoint(model, optimizer, generator_name, generator_object, config, 
                    step, images_seen):
    
    # Evaluation only needs the model and its experiment metadata.
    checkpoint = {
        "config": config,
        "model": model.state_dict(),
        "images_seen": images_seen,
        "wandb_run_id": wandb.run.id,
    }
    resume_checkpoint = {
        "optimizer": optimizer.state_dict(),
        "generator_buffer": generator_object.buffer if generator_object is not None else None,
        "generator_depths": getattr(generator_object, "depths", None),
        "step": step,
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_state": torch.cuda.get_rng_state_all(),
    }
    checkpoint_path = (CHECKPOINTS_DIR / f"model_{config['size']}_generator_{generator_name}_images_seen_{images_seen}.pth")
    resume_checkpoint_path = get_resume_checkpoint_path(checkpoint_path)
    temporary_resume_checkpoint_path = resume_checkpoint_path.with_suffix(".pth.tmp")
    torch.save(resume_checkpoint, temporary_resume_checkpoint_path)
    temporary_resume_checkpoint_path.replace(resume_checkpoint_path)

    temporary_checkpoint_path = checkpoint_path.with_suffix(".pth.tmp")
    torch.save(checkpoint, temporary_checkpoint_path)
    temporary_checkpoint_path.replace(checkpoint_path)  # Publish only the fully written checkpoint.
    print(f"Checkpoint saved at {checkpoint_path} and {resume_checkpoint_path} after {images_seen} images.")

    # Show generator images
    if generator_object is not None:
        with torch.random.fork_rng():
            if hasattr(generator_object, "depths"):
                batch_indices = generator_object.sample_indices(
                    config["train_batch_size"], min_depth=generator_object.min_depth)
            else:
                batch_indices = torch.randperm(
                    generator_object.buffer_size, device=generator_object.buffer_device
                )[:config["train_batch_size"]]
            checkpoint_images = generator_object.sample(batch_indices[:100]).cpu()
        buffer_grid = make_grid(checkpoint_images, nrow=10)
        wandb.log({"images/buffer": wandb.Image(buffer_grid), "images_seen": images_seen})


def main(config, checkpoint_path=None, run_only_one_checkpoint=False):
    start_step = 0
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        resume_checkpoint_path = get_resume_checkpoint_path(checkpoint_path)
        if resume_checkpoint_path.exists():
            checkpoint.update(torch.load(resume_checkpoint_path, map_location="cpu", weights_only=False))
        config = checkpoint["config"]
        start_step = checkpoint["step"]

    number_of_pretraining_images = config["number_of_pretraining_images"]
    checkpoint_every = config["checkpoint_every"]
    generator_name = config["generator_name"]

    train_batch_size = config["train_batch_size"]
    train_mini_batch_size = config["train_mini_batch_size"]
    accumulation_steps = train_batch_size // train_mini_batch_size
    base_learning_rate = config["base_learning_rate"]
    actual_learning_rate = base_learning_rate * train_batch_size / 256
    betas = config["betas"]
    weight_decay = config["weight_decay"]
    train_mask_ratio = config["train_mask_ratio"]

    model = MaskedAutoencoderViT(config).to("cuda").train()
    # Match MAE: do not apply weight decay to biases and LayerNorm parameters.
    decay_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad and parameter.ndim > 1]
    no_decay_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad and parameter.ndim == 1]
    optimizer = torch.optim.AdamW([{"params": decay_parameters, "weight_decay": weight_decay}, 
                                   {"params": no_decay_parameters, "weight_decay": 0.0}], 
                                   lr=actual_learning_rate, betas=betas, fused=True)
    if checkpoint_path is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = actual_learning_rate

    # Keep the original model for checkpoint compatibility and compile only its training forwards.
    compiled_model = torch.compile(model)

    generator_object = None
    if generator_name in ["irc", "irc_new"]:
        buffer_device = config["irc_buffer_device"]
        generator_class = IRCGenerator if generator_name == "irc" else IRCNewGenerator
        if checkpoint_path is not None:
            generator_buffer = checkpoint["generator_buffer"]
            generator_arguments = {
                "buffer_size": generator_buffer.size(0),
                "buffer": generator_buffer,
                "buffer_device": buffer_device,
            }
            if generator_name == "irc_new":
                generator_arguments["depths"] = checkpoint.get("generator_depths")
            generator_object = generator_class(config["image_size"], **generator_arguments)
        else:
            generator_object = generator_class(config["image_size"], buffer_device=buffer_device)

    # WandB setup
    wandb.login()
    if checkpoint_path is not None:
        wandb.init(project="universal-pretraining-for-images", id=checkpoint["wandb_run_id"], 
                   resume="must", allow_val_change=True, config={**config})
    else:
        wandb.init(project="universal-pretraining-for-images", 
                name=f"{config['size']} {number_of_pretraining_images} {generator_name} {datetime.now():%Y-%m-%d %H-%M-%S}", 
                config={**config})

    wandb.define_metric("images_seen", hidden=True)
    wandb.define_metric("train/*", step_metric="images_seen")
    wandb.define_metric("eval_validation_accuracy/*", step_metric="images_seen")
    wandb.define_metric("eval_validation_reconstruction_loss/*", step_metric="images_seen")
    wandb.define_metric("eval_validation_means/*", step_metric="images_seen")
    wandb.define_metric("eval_test_accuracy/*", step_metric="images_seen")
    wandb.define_metric("eval_test_reconstruction_loss/*", step_metric="images_seen")
    wandb.define_metric("eval_test_means/*", step_metric="images_seen")
    if generator_object is not None:
        wandb.define_metric("images/*", step_metric="images_seen")

    if checkpoint_path is not None:
        random.setstate(checkpoint["python_random_state"])
        np.random.set_state(checkpoint["numpy_random_state"])
        torch.set_rng_state(checkpoint["torch_random_state"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_random_state"])
        del checkpoint

    # Initial checkpoint
    if checkpoint_path is None:
        save_checkpoint(model, optimizer, generator_name, generator_object, config, 
                        step=0, images_seen=0)

    # Calculate the number of steps
    total_steps = number_of_pretraining_images // train_batch_size
    checkpoint_steps = checkpoint_every // train_batch_size
    warmup_images = number_of_pretraining_images * 0.05
    loss_sum = torch.zeros((), device="cuda")

    # Training loop
    for step in range(start_step, total_steps):
        # Linearly warm up for 5% of the images, then cosine decay to zero.
        images_seen_before_step = step * train_batch_size
        if images_seen_before_step < warmup_images:
            learning_rate = actual_learning_rate * (images_seen_before_step + train_batch_size) / warmup_images
        else:
            decay_progress = ((images_seen_before_step - warmup_images) /
                              (number_of_pretraining_images - warmup_images))
            learning_rate = actual_learning_rate * 0.5 * (1 + math.cos(math.pi * decay_progress))
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate

        if generator_name in ["irc", "irc_new"]:
            irc_batch_indices = generator_object.prepare_batch(train_batch_size)

        # Average the mini-batch gradients so one optimizer step still represents train_batch_size images.
        for mini_step in range(accumulation_steps):
            if generator_name == "gaussian_blurred_noise":
                images = generate_gaussian_blurred_noise(train_mini_batch_size, config["image_size"])
            elif generator_name == "spectrum":
                images = generate_spectrum_batch(train_mini_batch_size, config["image_size"])
            elif generator_name == "styleGAN-random":
                images = generate_stylegan_random_batch(train_mini_batch_size, config["image_size"])
            elif generator_name == "real_video":
                mini_batch_index = step * accumulation_steps + mini_step
                images = generate_from_real_video(train_mini_batch_size, config["image_size"], mini_batch_index)
            elif generator_name in ["irc", "irc_new"]:
                mini_batch_start = mini_step * train_mini_batch_size
                mini_batch_indices = irc_batch_indices[mini_batch_start:mini_batch_start + train_mini_batch_size]
                images = generator_object.sample(mini_batch_indices)
            else:
                raise ValueError(f"Unknown generator: {generator_name}")

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, _, _ = compiled_model(images, train_mask_ratio)
            (loss / accumulation_steps).backward()
            loss_sum.add_(loss.detach() / accumulation_steps)
            del images

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Average on the GPU and synchronize with the CPU only once every ten steps.
        if (step + 1) % 10 == 0:
            wandb.log({"train/loss": (loss_sum / 10).item(), "train/learning_rate": learning_rate,
                       "images_seen": (step + 1) * train_batch_size})
            loss_sum.zero_()
        # log_pending_eval_results(PROJECT_ROOT, wandb.run.id)

        if (step + 1) % checkpoint_steps == 0:
            save_checkpoint(model, optimizer, generator_name, generator_object, config, 
                            step=step + 1, images_seen=(step + 1) * train_batch_size)
            if run_only_one_checkpoint:
                print("Stopping after saving one checkpoint as requested.")
                break

    # log_pending_eval_results(PROJECT_ROOT, wandb.run.id)
    wandb.finish()
