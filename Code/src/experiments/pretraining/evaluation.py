from pathlib import Path
from time import perf_counter

import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from ...util import set_seed
from ..evaluation_util import write_eval_result
from ...model import MaskedAutoencoderViT, VisionTransformerClassifierHead


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def start_cuda_timer():
    # Synchronize only at section boundaries so asynchronous CUDA work is timed accurately.
    torch.cuda.synchronize()
    return perf_counter()


def stop_cuda_timer(start_time):
    torch.cuda.synchronize()
    return perf_counter() - start_time


def get_datasets(image_size, dataset_name):
    train_path = PROJECT_ROOT / "datasets" / dataset_name / "train"
    test_path = PROJECT_ROOT / "datasets" / dataset_name / "test"

    if dataset_name in ["cifar10", "tiny-imagenet-200", "eurosat", "cifar100", "resisc45", 
                        "line-field-orientation", "shape-color-objects", "texture-mosaic"]:
        transform = transforms.Compose([transforms.Resize((image_size, image_size)), 
                                        transforms.ToTensor()])
    else:
        transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(image_size), 
                                        transforms.ToTensor()])

    train_dataset = datasets.ImageFolder(train_path, transform=transform)
    test_dataset = datasets.ImageFolder(test_path, transform=transform)

    return train_dataset, test_dataset, len(train_dataset.classes)


def evaluate_reconstruction(model, test_loader, evaluate_reconstructin_mask_ratio=0.75):
    model.to('cuda')
    model.eval()
    total_loss = 0.0
    total_images = 0
    cached_images = None
    cached_labels = torch.empty(len(test_loader.dataset), dtype=torch.long)
    cache_start = 0

    with torch.inference_mode():
        for images, labels in test_loader:
            if cached_images is None:
                cached_images = torch.empty((len(test_loader.dataset), *images.shape[1:]), dtype=images.dtype)
            cache_end = cache_start + images.shape[0]
            cached_images[cache_start:cache_end].copy_(images)
            cached_labels[cache_start:cache_end].copy_(labels)
            cache_start = cache_end

            images = images.to('cuda', non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss, _, _ = model(images, mask_ratio=evaluate_reconstructin_mask_ratio)
            total_loss += loss.item() * images.shape[0]
            total_images += images.shape[0]

    return total_loss / total_images, TensorDataset(cached_images, cached_labels)


def evaluate_classification(model, config, train_loader, test_loader, num_classes, batch_size):  
    classifier = VisionTransformerClassifierHead(config, num_classes)
    timings = {}

    # Keep the MAE on the GPU to avoid repeated transfers between datasets.
    # Full FP32 MAE parameter memory (parameter count * 4 bytes):
    # ViT-Tiny:  6,070,336 * 4 = 24.3 MB
    # ViT-Small: 25,171,840 * 4 = 100.7 MB
    # ViT-Base:  111,907,840 * 4 = 447.6 MB
    # ViT-Large: 329,541,888 * 4 = 1,318.2 MB (1.23 GiB)
    # model.decoder.to('cpu')
    # torch.cuda.empty_cache()

    # Feature extraction
    def cache_features(loader):
        model.encoder.eval()
        features = torch.empty(len(loader.dataset), config["encoder_hidden_size"], device='cuda')
        labels = torch.empty(len(loader.dataset), dtype=torch.long, device='cuda')
        start = 0

        with torch.inference_mode():
            for images, batch_labels in loader:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    batch_features = model.encoder.forward_no_mask(images.to('cuda', non_blocking=True))
                    # batch_features = batch_features[:, 0]  # CLS token
                    batch_features = batch_features[:, 1:].mean(dim=1)  # Mean of all tokens
                end = start + batch_features.shape[0]
                features[start:end] = batch_features
                labels[start:end] = batch_labels.to('cuda', non_blocking=True)
                start = end

        return features, labels

    start_time = start_cuda_timer()
    train_features, train_labels = cache_features(train_loader)
    timings["train_feature_extraction"] = stop_cuda_timer(start_time)

    start_time = start_cuda_timer()
    test_features, test_labels = cache_features(test_loader)
    timings["test_feature_extraction"] = stop_cuda_timer(start_time)

    # Keep the encoder on the GPU for the same reason described above.
    # model.encoder.to('cpu')
    # torch.cuda.empty_cache()

    # Keep the FP32 feature cache on the GPU instead of copying every batch for 100 epochs.
    # Largest training cache (100,000 images * hidden size * 4 bytes):
    # ViT-Tiny:  100,000 * 192 * 4 = 76.8 MB
    # ViT-Small: 100,000 * 384 * 4 = 153.6 MB
    # ViT-Base:  100,000 * 768 * 4 = 307.2 MB
    # ViT-Large: 100,000 * 1,024 * 4 = 409.6 MB

    # Training
    classifier.to('cuda')
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(classifier.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=10)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=90)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler], milestones=[10])

    classifier.train()
    start_time = start_cuda_timer()
    for _ in range(100):
        indices = torch.randperm(train_features.shape[0], device='cuda')
        shuffled_features = train_features[indices]
        shuffled_labels = train_labels[indices]
        for start in range(0, train_features.shape[0], batch_size):
            features = shuffled_features[start:start + batch_size]
            labels = shuffled_labels[start:start + batch_size]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = classifier(features)
                loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        del features, labels, shuffled_features, shuffled_labels
    timings["linear_probe_training"] = stop_cuda_timer(start_time)

    # Evaluation
    classifier.eval()
    correct = 0
    total = 0
    start_time = start_cuda_timer()
    with torch.inference_mode():
        for start in range(0, test_features.shape[0], batch_size):
            features = test_features[start:start + batch_size]
            labels = test_labels[start:start + batch_size]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                predictions = classifier(features).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.shape[0]
    timings["linear_probe_evaluation"] = stop_cuda_timer(start_time)

    del classifier
    start_time = start_cuda_timer()
    torch.cuda.empty_cache()
    timings["empty_cache"] = stop_cuda_timer(start_time)
    return correct / total, timings


def evaluate_datasets(model, config, dataset_names, 
                      reconstruction_batch_sizes, feature_batch_sizes, linear_probe_batch_size,
                      metric_group):
    
    metrics = {}
    reconstruction_losses = []
    accuracies = []

    for dataset_name in dataset_names:
        dataset_start_time = perf_counter()
        set_seed()

        start_time = perf_counter()
        train_dataset, test_dataset, num_classes = get_datasets(config["image_size"], dataset_name)
        timings = {"dataset_indexing": perf_counter() - start_time}

        reconstruction_loader = DataLoader(test_dataset, batch_size=reconstruction_batch_sizes[dataset_name], shuffle=False,
                                           num_workers=8, pin_memory=True)
        start_time = start_cuda_timer()
        reconstruction_loss, cached_test_dataset = evaluate_reconstruction(model, reconstruction_loader)
        timings["reconstruction"] = stop_cuda_timer(start_time)
        metrics[f"{metric_group}_reconstruction_loss/{dataset_name}_loss"] = reconstruction_loss
        reconstruction_losses.append(reconstruction_loss)

        train_loader = DataLoader(train_dataset, batch_size=feature_batch_sizes[dataset_name], shuffle=False,
                                  num_workers=8, pin_memory=True)
        test_loader = DataLoader(cached_test_dataset, batch_size=feature_batch_sizes[dataset_name], shuffle=False,
                                 num_workers=8, pin_memory=True)
        accuracy, classification_timings = evaluate_classification(
            model, config, train_loader, test_loader, num_classes, linear_probe_batch_size
        )
        timings.update(classification_timings)
        metrics[f"{metric_group}_accuracy/{dataset_name}_accuracy"] = accuracy
        accuracies.append(accuracy)
        del reconstruction_loader, train_loader, test_loader, cached_test_dataset

        timings["dataset_total"] = perf_counter() - dataset_start_time
        timing_text = " ".join(f"{name}={seconds:.2f}s" for name, seconds in timings.items())
        print(f"Timing: dataset={dataset_name} {timing_text}", flush=True)

        # Yield progress after each dataset so it can be saved immediately.
        metrics[f"{metric_group}_means/accuracy"] = sum(accuracies) / len(accuracies)
        metrics[f"{metric_group}_means/reconstruction_loss"] = sum(reconstruction_losses) / len(reconstruction_losses)
        yield dataset_name, metrics
        
        print(f"Dataset: {dataset_name}, Reconstruction Loss: {reconstruction_loss:.4f}, Accuracy: {accuracy:.4f}", flush=True)


def main(checkpoint_path, datasets_to_evaluate,
         reconstruction_batch_sizes, feature_batch_sizes, linear_probe_batch_size,
         metric_group, result_suffix=""):
    
    checkpoint_path = Path(checkpoint_path)

    start_time = start_cuda_timer()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    images_seen = checkpoint["images_seen"]
    wandb_run_id = checkpoint["wandb_run_id"]

    model = MaskedAutoencoderViT(config).to('cuda')
    model.load_state_dict(checkpoint["model"])
    del checkpoint  # This is on CPU, but it is big
    print(f"Timing: checkpoint_loading={stop_cuda_timer(start_time):.2f}s", flush=True)

    for dataset_name, metrics in evaluate_datasets(model, config, datasets_to_evaluate,
                                                   reconstruction_batch_sizes, feature_batch_sizes, linear_probe_batch_size,
                                                   metric_group=metric_group):
        start_time = perf_counter()
        write_eval_result(PROJECT_ROOT, checkpoint_path, images_seen, metrics, wandb_run_id, result_suffix=result_suffix)
        print(f"Timing: dataset={dataset_name} result_write={perf_counter() - start_time:.2f}s", flush=True)
