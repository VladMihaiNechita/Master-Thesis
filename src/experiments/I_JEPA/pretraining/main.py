"""I-JEPA pretraining experiment entry point."""

from contextlib import nullcontext
from datetime import datetime
import math
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from torch.nn.parallel import DistributedDataParallel
from torchvision.utils import make_grid

from ....generators.gaussian_blurred_noise import generate_gaussian_blurred_noise
from ....generators.irc import IRCGenerator
from ....generators.irc_new import IRCGenerator as IRCNewGenerator
from ....generators.real_video import generate_from_real_video
from ....generators.spectrum import generate_spectrum_batch
from ....generators.styleGAN_random import generate_stylegan_random_batch
from ....models import IJEPA
from .masking import IJEPAMaskSampler


PROJECT_ROOT = Path(__file__).resolve().parents[4]
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINTS_DIR.mkdir(exist_ok=True)


def get_resume_checkpoint_path(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    return checkpoint_path.with_name(f"{checkpoint_path.stem}_resume{checkpoint_path.suffix}")


def save_checkpoint(model, optimizer, generator_name, generator_object, config,
                    step, images_seen):
    # Keep evaluation weights separate from the larger exact-resume state, as in MAE.
    checkpoint = {
        "config": config,
        "encoder": model.encoder.state_dict(),
        "predictor": model.predictor.state_dict(),
        "target_encoder": model.target_encoder.state_dict(),
        "step": step,
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

    checkpoint_path = (
        CHECKPOINTS_DIR /
        f"ijepa_model_{config['size']}_generator_{generator_name}_images_seen_{images_seen}.pth"
    )
    resume_checkpoint_path = get_resume_checkpoint_path(checkpoint_path)

    # Publish each checkpoint only after torch.save has completed successfully.
    temporary_resume_path = resume_checkpoint_path.with_suffix(".pth.tmp")
    torch.save(resume_checkpoint, temporary_resume_path)
    temporary_resume_path.replace(resume_checkpoint_path)

    temporary_checkpoint_path = checkpoint_path.with_suffix(".pth.tmp")
    torch.save(checkpoint, temporary_checkpoint_path)
    temporary_checkpoint_path.replace(checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path} and {resume_checkpoint_path} "
          f"after {images_seen} images.")

    if generator_object is not None:
        with torch.random.fork_rng():
            if hasattr(generator_object, "depths"):
                batch_indices = generator_object.sample_indices(
                    config["train_batch_size"], min_depth=generator_object.min_depth
                )
            else:
                batch_indices = torch.randperm(
                    generator_object.buffer_size, device=generator_object.buffer_device
                )[:config["train_batch_size"]]
            checkpoint_images = generator_object.sample(batch_indices[:100]).cpu()

        image_grid = make_grid(checkpoint_images, nrow=10)
        wandb.log({"images/buffer": wandb.Image(image_grid), "images_seen": images_seen})


def get_learning_rate(step, total_steps, start_learning_rate, peak_learning_rate,
                      final_learning_rate, warmup_fraction):
    warmup_steps = int(total_steps * warmup_fraction)
    completed_steps = step + 1
    if completed_steps < warmup_steps:
        warmup_progress = completed_steps / max(1, warmup_steps)
        return start_learning_rate + warmup_progress * (
            peak_learning_rate - start_learning_rate
        )

    decay_progress = (completed_steps - warmup_steps) / max(1, total_steps - warmup_steps)
    return final_learning_rate + (peak_learning_rate - final_learning_rate) * 0.5 * (
        1.0 + math.cos(math.pi * decay_progress)
    )


def get_legacy_learning_rate(step, total_steps, peak_learning_rate, warmup_fraction):
    warmup_steps = total_steps * warmup_fraction
    if step < warmup_steps:
        return peak_learning_rate * (step + 1) / warmup_steps

    decay_progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return peak_learning_rate * 0.5 * (1.0 + math.cos(math.pi * decay_progress))


def get_weight_decay(step, total_steps, start_weight_decay, final_weight_decay):
    progress = (step + 1) / total_steps
    return final_weight_decay + (start_weight_decay - final_weight_decay) * 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )


def get_ema_momentum(step, total_steps, ema_range):
    start_momentum, end_momentum = ema_range
    return start_momentum + step * (end_momentum - start_momentum) / total_steps


@torch.no_grad()
def update_target_encoder(model, momentum):
    for encoder_parameter, target_parameter in zip(
        model.encoder.parameters(), model.target_encoder.parameters()
    ):
        target_parameter.mul_(momentum).add_(encoder_parameter, alpha=1.0 - momentum)


def generate_image_batches(config, generator_object, step, accumulation_steps,
                           rank=0, world_size=1):
    generator_name = config["generator_name"]
    mini_batch_size = config["train_mini_batch_size"]

    if generator_name in ["irc", "irc_new"]:
        batch_indices = generator_object.prepare_batch(config["train_batch_size"])
        base_batch_size, extra_images = divmod(config["train_batch_size"], world_size)
        valid_batch_size = base_batch_size + int(rank < extra_images)
        local_batch_size = math.ceil(config["train_batch_size"] / world_size)
        start = rank * base_batch_size + min(rank, extra_images)
        batch_indices = batch_indices[start:start + valid_batch_size]
        if valid_batch_size < local_batch_size:
            batch_indices = torch.cat([
                batch_indices,
                batch_indices[-1:].expand(local_batch_size - valid_batch_size),
            ])

    for mini_step in range(accumulation_steps):
        if generator_name == "gaussian_blurred_noise":
            images = generate_gaussian_blurred_noise(mini_batch_size, config["image_size"])
        elif generator_name == "spectrum":
            images = generate_spectrum_batch(mini_batch_size, config["image_size"])
        elif generator_name == "styleGAN-random":
            images = generate_stylegan_random_batch(mini_batch_size, config["image_size"])
        elif generator_name == "real_video":
            mini_batch_index = step * accumulation_steps + mini_step
            images = generate_from_real_video(mini_batch_size, config["image_size"], mini_batch_index)
        elif generator_name in ["irc", "irc_new"]:
            start = mini_step * mini_batch_size
            images = generator_object.sample(batch_indices[start:start + mini_batch_size])
        else:
            raise ValueError(f"Unknown generator: {generator_name}")

        if config.get("channels_last", False):
            images = images.contiguous(memory_format=torch.channels_last)
        yield images


def get_gradient_norm(parameters):
    gradient_norms = [parameter.grad.detach().norm(2) for parameter in parameters
                      if parameter.grad is not None]
    return torch.stack(gradient_norms).norm(2)


def train_step(model, training_model, optimizer, mask_sampler, image_batches,
               accumulation_steps, step, total_steps, ema_range,
               valid_local_batch_size, global_batch_size, world_size):
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    context_patches = 0.0
    target_patches = 0.0

    for mini_step, images in enumerate(image_batches):
        valid_mini_batch_size = min(
            images.shape[0],
            max(0, valid_local_batch_size - mini_step * images.shape[0]),
        )
        context_indices, target_indices = mask_sampler(images.shape[0], images.device)
        context_patches += context_indices.shape[1] / accumulation_steps
        target_patches += target_indices[0].shape[1] / accumulation_steps
        sync_context = (
            training_model.no_sync()
            if hasattr(training_model, "no_sync") and mini_step + 1 < accumulation_steps
            else nullcontext()
        )
        with sync_context:
            with torch.autocast(device_type=images.device.type, dtype=torch.bfloat16,
                                enabled=images.is_cuda):
                predictions, targets = training_model(images, context_indices, target_indices)
                prediction_shape = predictions.shape[1:]
                predictions = predictions.reshape(
                    len(target_indices), images.shape[0], *prediction_shape
                )[:, :valid_mini_batch_size].reshape(-1, *prediction_shape)
                targets = targets.reshape(
                    len(target_indices), images.shape[0], *prediction_shape
                )[:, :valid_mini_batch_size].reshape(-1, *prediction_shape)
                loss = F.smooth_l1_loss(
                    predictions,
                    targets,
                )

            # DDP averages ranks equally, so weight uneven final ranks by sample count.
            loss_weight = world_size * valid_mini_batch_size / global_batch_size
            (loss * loss_weight).backward()
        loss_sum = loss_sum + loss.detach() * loss_weight

    encoder_gradient_norm = get_gradient_norm(model.encoder.parameters())
    predictor_gradient_norm = get_gradient_norm(model.predictor.parameters())
    optimizer.step()

    # Update the teacher once per effective batch, after the online optimizer step.
    momentum = get_ema_momentum(step, total_steps, ema_range)
    update_target_encoder(model, momentum)
    return (loss_sum, momentum, encoder_gradient_norm, predictor_gradient_norm,
            context_patches, target_patches)


def main(config, checkpoint_path=None, run_only_one_checkpoint=False):
    world_size = int(os.environ.get("SLURM_NTASKS", "1"))
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    local_rank = int(os.environ.get("SLURM_LOCALID", "0"))
    distributed = world_size > 1
    if distributed:
        os.environ.setdefault("RANK", str(rank))
        os.environ.setdefault("WORLD_SIZE", str(world_size))
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl")

    start_step = 0
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint.update(torch.load(
            get_resume_checkpoint_path(checkpoint_path), map_location="cpu", weights_only=False
        ))
        config = checkpoint["config"]
        start_step = checkpoint["step"]

    train_batch_size = config["train_batch_size"]
    local_batch_size = math.ceil(train_batch_size / world_size)
    valid_local_batch_size = train_batch_size // world_size + int(
        rank < train_batch_size % world_size
    )
    accumulation_steps = local_batch_size // config["train_mini_batch_size"]
    total_steps = config["number_of_pretraining_images"] // train_batch_size
    checkpoint_steps = config["checkpoint_every"] // train_batch_size
    explicit_learning_rate_schedule = "peak_learning_rate" in config
    if explicit_learning_rate_schedule:
        start_learning_rate = config["start_learning_rate"]
        peak_learning_rate = config["peak_learning_rate"]
        final_learning_rate = config["final_learning_rate"]
    else:
        # Preserve the original schedule when resuming an older checkpoint.
        start_learning_rate = 0.0
        peak_learning_rate = config["base_learning_rate"] * train_batch_size / 256
        final_learning_rate = 0.0
    explicit_weight_decay_schedule = "final_weight_decay" in config

    model = IJEPA(config).to("cuda").train()
    if config.get("channels_last", False):
        model.to(memory_format=torch.channels_last)
    decay_parameters = [parameter for parameter in model.parameters()
                        if parameter.requires_grad and parameter.ndim > 1]
    no_decay_parameters = [parameter for parameter in model.parameters()
                           if parameter.requires_grad and parameter.ndim == 1]
    optimizer = torch.optim.AdamW(
        [{"params": decay_parameters, "weight_decay": config["weight_decay"]},
         {"params": no_decay_parameters, "weight_decay": 0.0}],
        lr=peak_learning_rate,
        betas=config["betas"],
        fused=True,
    )
    if checkpoint_path is not None:
        model.encoder.load_state_dict(checkpoint["encoder"])
        model.predictor.load_state_dict(checkpoint["predictor"])
        model.target_encoder.load_state_dict(checkpoint["target_encoder"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if config.get("channels_last", False):
            # Fused AdamW requires restored moments to match 4-D parameter layouts.
            for parameter, state in optimizer.state.items():
                if parameter.ndim == 4:
                    for name, value in state.items():
                        if torch.is_tensor(value) and value.ndim == 4:
                            state[name] = value.contiguous(
                                memory_format=torch.channels_last
                            )

    training_model = (
        DistributedDataParallel(model, device_ids=[local_rank]) if distributed else model
    )

    mask_sampler = IJEPAMaskSampler(config)

    generator_object = None
    if config["generator_name"] in ["irc", "irc_new"]:
        generator_class = IRCGenerator if config["generator_name"] == "irc" else IRCNewGenerator
        generator_arguments = {"buffer_device": config["irc_buffer_device"]}
        if "irc_source_batch_size" in config:
            generator_arguments["source_batch_size"] = config["irc_source_batch_size"]
        if checkpoint_path is None:
            generator_object = generator_class(config["image_size"], **generator_arguments)
        else:
            generator_arguments.update({
                "buffer_size": checkpoint["generator_buffer"].size(0),
                "buffer": checkpoint["generator_buffer"],
            })
            if config["generator_name"] == "irc_new":
                generator_arguments["depths"] = checkpoint["generator_depths"]
            generator_object = generator_class(config["image_size"], **generator_arguments)

    if rank == 0:
        wandb.login()
        if checkpoint_path is None:
            wandb.init(
                project="universal-pretraining-for-images",
                name=(f"I-JEPA {config['size']} {config['number_of_pretraining_images']} "
                      f"{config['generator_name']} {datetime.now():%Y-%m-%d %H-%M-%S}"),
                config={**config, "distributed_world_size": world_size},
            )
        else:
            wandb.init(
                project="universal-pretraining-for-images",
                id=checkpoint["wandb_run_id"],
                resume="must",
                allow_val_change=True,
                config={**config, "distributed_world_size": world_size},
            )

        wandb.define_metric("images_seen", hidden=True)
        wandb.define_metric("train/*", step_metric="images_seen")
        if generator_object is not None:
            wandb.define_metric("images/*", step_metric="images_seen")

    if checkpoint_path is not None:
        # Model, generator and WandB setup may consume randomness, so restore RNG last.
        random.setstate(checkpoint["python_random_state"])
        np.random.set_state(checkpoint["numpy_random_state"])
        torch.set_rng_state(checkpoint["torch_random_state"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_random_state"])
        del checkpoint

    if checkpoint_path is None and rank == 0:
        save_checkpoint(
            model, optimizer, config["generator_name"], generator_object, config,
            step=0, images_seen=0
        )

    loss_sum = torch.zeros((), device="cuda")
    encoder_gradient_norm_sum = torch.zeros((), device="cuda")
    predictor_gradient_norm_sum = torch.zeros((), device="cuda")
    context_patches_sum = 0.0
    target_patches_sum = 0.0
    last_checkpoint_step = start_step

    for step in range(start_step, total_steps):
        if explicit_learning_rate_schedule:
            learning_rate = get_learning_rate(
                step, total_steps, start_learning_rate, peak_learning_rate,
                final_learning_rate, config["warmup_fraction"]
            )
        else:
            learning_rate = get_legacy_learning_rate(
                step, total_steps, peak_learning_rate, config["warmup_fraction"]
            )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate
        if explicit_weight_decay_schedule:
            weight_decay = get_weight_decay(
                step, total_steps, config["weight_decay"], config["final_weight_decay"]
            )
            optimizer.param_groups[0]["weight_decay"] = weight_decay
        else:
            weight_decay = optimizer.param_groups[0]["weight_decay"]

        image_batches = generate_image_batches(
            config, generator_object, step, accumulation_steps, rank, world_size
        )
        (loss, momentum, encoder_gradient_norm, predictor_gradient_norm,
         context_patches, target_patches) = train_step(
            model, training_model, optimizer, mask_sampler, image_batches, accumulation_steps,
            step, total_steps, config["ema"], valid_local_batch_size,
            train_batch_size, world_size
        )
        loss_sum.add_(loss)
        encoder_gradient_norm_sum.add_(encoder_gradient_norm)
        predictor_gradient_norm_sum.add_(predictor_gradient_norm)
        context_patches_sum += context_patches
        target_patches_sum += target_patches

        if (step + 1) % 10 == 0:
            if distributed:
                for value in (loss_sum, encoder_gradient_norm_sum,
                              predictor_gradient_norm_sum):
                    dist.reduce(value, dst=0)
            if rank == 0:
                wandb.log({
                    "train/loss": (loss_sum / (10 * world_size)).item(),
                    "train/learning_rate": learning_rate,
                    "train/weight_decay": weight_decay,
                    "train/ema_momentum": momentum,
                    "train/encoder_gradient_norm": (
                        encoder_gradient_norm_sum / (10 * world_size)
                    ).item(),
                    "train/predictor_gradient_norm": (
                        predictor_gradient_norm_sum / (10 * world_size)
                    ).item(),
                    "train/context_patches": context_patches_sum / 10,
                    "train/target_patches_per_block": target_patches_sum / 10,
                    "images_seen": (step + 1) * train_batch_size,
                })
            loss_sum.zero_()
            encoder_gradient_norm_sum.zero_()
            predictor_gradient_norm_sum.zero_()
            context_patches_sum = 0.0
            target_patches_sum = 0.0

        if (step + 1) % checkpoint_steps == 0:
            last_checkpoint_step = step + 1
            if distributed:
                dist.barrier()
            if rank == 0:
                save_checkpoint(
                    model, optimizer, config["generator_name"], generator_object, config,
                    step=step + 1, images_seen=(step + 1) * train_batch_size
                )
            if distributed:
                dist.barrier()
            if run_only_one_checkpoint:
                print("Stopping after saving one checkpoint as requested.")
                break
    else:
        if last_checkpoint_step != total_steps and rank == 0:
            save_checkpoint(
                model, optimizer, config["generator_name"], generator_object, config,
                step=total_steps, images_seen=total_steps * train_batch_size
            )

    if rank == 0:
        wandb.finish()
    if distributed:
        dist.destroy_process_group()
