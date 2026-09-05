from datetime import datetime
from pathlib import Path
import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from .data_transforms import (IMAGENET_MEAN, IMAGENET_STD, BatchMixupCutmix, DeiTRandomErasing, build_training_transform)
from ....models import FineTuningModel


PROJECT_ROOT = Path(__file__).resolve().parents[4]
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINTS_DIR.mkdir(exist_ok=True)


def get_resume_checkpoint_path(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    return checkpoint_path.with_name(f"{checkpoint_path.stem}_resume{checkpoint_path.suffix}")


def adapt_patch_projection_to_normalized_input(projection, mean, std):
    """Preserve a raw-input projection when fine-tuning receives normalized images."""
    mean = projection.weight.new_tensor(mean).view(1, -1, 1, 1)
    std = projection.weight.new_tensor(std).view(1, -1, 1, 1)

    with torch.no_grad():
        # For z = (x - mean) / std, these parameters make W' z + b' = W x + b.
        # Update the bias before scaling the weight because it needs the original W.
        projection.bias.add_((projection.weight * mean).sum(dim=(1, 2, 3)))
        projection.weight.mul_(std)


def build_optimizer_parameter_groups(model, layer_decay):
    num_blocks = len(model.encoder.blocks)
    parameter_groups = {}

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if name.startswith("encoder.blocks."):
            layer_id = int(name.split(".")[2]) + 1
        elif name.startswith("encoder."):
            layer_id = 0
        else:
            layer_id = num_blocks + 1

        weight_decay = 0.05 if parameter.ndim > 1 and name != "encoder.cls_token" else 0.0
        group_key = (layer_id, weight_decay)
        if group_key not in parameter_groups:
            # The head uses the full rate; earlier encoder layers use progressively smaller rates.
            parameter_groups[group_key] = {
                "params": [],
                "weight_decay": weight_decay,
                "lr_scale": layer_decay ** (num_blocks + 1 - layer_id),
            }
        parameter_groups[group_key]["params"].append(parameter)

    return list(parameter_groups.values())


class CUDAPrefetcher:
    def __init__(self, loader, mean, std, random_erasing):
        self.iterator = iter(loader)
        self.mean = mean
        self.std = std
        self.random_erasing = random_erasing
        self.stream = torch.cuda.Stream()
        self.stream.wait_stream(torch.cuda.current_stream())
        self._preload()

    def _preload(self):
        try:
            images, targets = next(self.iterator)
        except StopIteration:
            self.next_images = None
            return

        # Transfer compact uint8 images, then prepare the next batch on CUDA.
        with torch.cuda.stream(self.stream):
            images = images.to("cuda", non_blocking=True).float().div_(255)
            images.sub_(self.mean).div_(self.std)
            self.next_images = self.random_erasing(images)
            self.next_targets = targets.to("cuda", non_blocking=True)

    def __iter__(self):
        return self

    def __next__(self):
        if self.next_images is None:
            raise StopIteration

        current_stream = torch.cuda.current_stream()
        current_stream.wait_stream(self.stream)
        images = self.next_images
        targets = self.next_targets
        images.record_stream(current_stream)
        targets.record_stream(current_stream)
        self._preload()
        return images, targets


def save_checkpoint(model, optimizer, config, fine_tuning_config,
                    pretraining_checkpoint_path, dataset, classes, epoch):
    # Testing only needs the model and its experiment metadata.
    checkpoint = {
        "config": config,
        "fine_tuning_config": fine_tuning_config,
        "model": model.state_dict(),
        "pretraining_checkpoint_path": pretraining_checkpoint_path,
        "dataset": dataset,
        "classes": classes,
        "epoch": epoch,
        "wandb_run_id": wandb.run.id,
    }
    resume_checkpoint = {
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_state": torch.cuda.get_rng_state_all(),
    }

    checkpoint_path = (CHECKPOINTS_DIR / f"fine_tuned_{pretraining_checkpoint_path.stem}_{dataset}"
                       f"_layer_decay_{fine_tuning_config['fine_tuning_layer_decay']}_epoch_{epoch}.pth")
    resume_checkpoint_path = get_resume_checkpoint_path(checkpoint_path)

    temporary_resume_checkpoint_path = resume_checkpoint_path.with_suffix(".pth.tmp")
    torch.save(resume_checkpoint, temporary_resume_checkpoint_path)
    temporary_resume_checkpoint_path.replace(resume_checkpoint_path)

    temporary_checkpoint_path = checkpoint_path.with_suffix(".pth.tmp")
    torch.save(checkpoint, temporary_checkpoint_path)
    temporary_checkpoint_path.replace(checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path} and {resume_checkpoint_path} after epoch {epoch}.")


def main(fine_tuning_config, fine_tuning_checkpoint_path=None,
         dataset="imagenet100-cmc", run_only_one_checkpoint=False):
    
    start_epoch = 0
    if fine_tuning_checkpoint_path is not None:
        checkpoint = torch.load(fine_tuning_checkpoint_path, map_location="cpu", weights_only=False)
        resume_checkpoint_path = get_resume_checkpoint_path(fine_tuning_checkpoint_path)
        if resume_checkpoint_path.exists():
            checkpoint.update(torch.load(resume_checkpoint_path, map_location="cpu", weights_only=False))
        config = checkpoint["config"]
        fine_tuning_config = checkpoint["fine_tuning_config"]
        pretraining_checkpoint_path = Path(checkpoint["pretraining_checkpoint_path"])
        dataset = checkpoint["dataset"]
        start_epoch = checkpoint["epoch"]
    else:
        pretraining_checkpoint_path = Path(fine_tuning_config["pretraining_checkpoint_path"])
        pretraining_checkpoint_path = CHECKPOINTS_DIR / pretraining_checkpoint_path
        checkpoint = torch.load(pretraining_checkpoint_path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]

    image_size = config["image_size"]
    train_batch_size = fine_tuning_config["fine_tuning_train_batch_size"]
    nr_of_epochs = fine_tuning_config["fine_tuning_epochs"]
    checkpoint_every_n_epochs = fine_tuning_config["checkpoint_every_n_epochs"]
    base_learning_rate = fine_tuning_config["fine_tuning_base_learning_rate"]
    actual_learning_rate = base_learning_rate * train_batch_size / 256
    warmup_epochs = fine_tuning_config["fine_tuning_warmup_epochs"]
    min_learning_rate = fine_tuning_config["fine_tuning_min_learning_rate"]
    layer_decay = fine_tuning_config["fine_tuning_layer_decay"]

    transform = build_training_transform(image_size)
    datasets_dir = Path(os.environ.get("DATASETS_DIR", PROJECT_ROOT / "datasets"))
    train_dataset_path = datasets_dir / dataset / "train"
    train_dataset = ImageFolder(train_dataset_path, transform=transform)
    mixup_cutmix = BatchMixupCutmix(num_classes=len(train_dataset.classes))
    random_erasing = DeiTRandomErasing()
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, 
                              num_workers=8, pin_memory=True, drop_last=True, 
                              persistent_workers=True, prefetch_factor=1, multiprocessing_context="spawn")
    normalization_mean = torch.tensor(IMAGENET_MEAN, device="cuda").view(1, -1, 1, 1)
    normalization_std = torch.tensor(IMAGENET_STD, device="cuda").view(1, -1, 1, 1)

    model = FineTuningModel(config, num_classes=len(train_dataset.classes))

    if fine_tuning_checkpoint_path is not None:
        model.load_state_dict(checkpoint["model"])
    else:
        encoder_state = {name.removeprefix("encoder."): parameter 
                         for name, parameter in checkpoint["model"].items() 
                         if name.startswith("encoder.") and not name.startswith("encoder.norm.")}
        model.encoder.load_state_dict(encoder_state, strict=True)
        if not pretraining_checkpoint_path.stem.endswith("images_seen_0"):
            # IRC pretraining used raw [0, 1] images. Adapt only once, when loading
            # that checkpoint, so normalized images produce the same patch embeddings.
            adapt_patch_projection_to_normalized_input(model.encoder.projection, IMAGENET_MEAN, IMAGENET_STD)
    model = model.to("cuda").train()

    # Keep the original model for checkpoint compatibility and compile only its training forwards.
    compiled_model = model if os.environ.get("DISABLE_COMPILE") == "1" else torch.compile(model)

    parameter_groups = build_optimizer_parameter_groups(model, layer_decay)
    optimizer = torch.optim.AdamW(parameter_groups,
                                  lr=actual_learning_rate, betas=(0.9, 0.999), fused=True)

    # WandB setup
    wandb.login()
    wandb_config = {**config, **fine_tuning_config, "dataset": dataset}
    if fine_tuning_checkpoint_path is not None:
        wandb.init(project="universal-pretraining-for-images", id=checkpoint["wandb_run_id"],
                   resume="must", allow_val_change=True, config=wandb_config)
    else:
        wandb.init(project="universal-pretraining-for-images",
                   name=(f"fine-tuning {config['size']} {pretraining_checkpoint_path.stem} "
                         f"{dataset} {datetime.now():%Y-%m-%d %H-%M-%S}"),
                   config=wandb_config)

    wandb.define_metric("epoch", hidden=True)
    wandb.define_metric("train/*", step_metric="epoch")

    if fine_tuning_checkpoint_path is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        random.setstate(checkpoint["python_random_state"])
        np.random.set_state(checkpoint["numpy_random_state"])
        torch.set_rng_state(checkpoint["torch_random_state"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_random_state"])
    del checkpoint

    for epoch in range(start_epoch, nr_of_epochs):
        loss_sum = torch.zeros((), device="cuda")
        images_seen = 0

        # Starts new workers each epoch, or resets the existing ones when they are persistent.
        train_iterator = CUDAPrefetcher(train_loader, normalization_mean, normalization_std, random_erasing)
        for batch_index, (images, targets) in enumerate(train_iterator):
            epoch_progress = epoch + batch_index / len(train_loader)
            if epoch_progress < warmup_epochs:
                learning_rate = actual_learning_rate * epoch_progress / warmup_epochs
            else:
                decay_progress = ((epoch_progress - warmup_epochs) / (nr_of_epochs - warmup_epochs))
                learning_rate = min_learning_rate + (actual_learning_rate - min_learning_rate) * 0.5 * (1 + math.cos(math.pi * decay_progress))

            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = learning_rate * parameter_group["lr_scale"]


            images, targets = mixup_cutmix(images, targets)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = compiled_model(images)
                loss = torch.sum(-targets * F.log_softmax(logits, dim=-1), dim=-1).mean()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            loss_sum.add_(loss.detach() * images.shape[0])
            images_seen += images.shape[0]

        average_training_loss = (loss_sum / images_seen).item()
        print(f"Epoch {epoch + 1}/{nr_of_epochs}, "f"average training loss: {average_training_loss:.4f}")
        wandb.log({"train/loss": average_training_loss, "train/learning_rate": learning_rate, "epoch": epoch + 1})

        if (epoch + 1) % checkpoint_every_n_epochs == 0:
            save_checkpoint(model, optimizer, config, fine_tuning_config,
                            pretraining_checkpoint_path, dataset, train_dataset.classes,
                            epoch=epoch + 1)
            if run_only_one_checkpoint:
                break

    wandb.finish()
