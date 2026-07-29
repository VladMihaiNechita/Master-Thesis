from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ...model import FineTuningModel
from ..evaluation_util import write_eval_result


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def get_dataset(image_size, dataset_name):
    test_path = PROJECT_ROOT / "datasets" / dataset_name / "test"
    transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(image_size), transforms.ToTensor()])
    return datasets.ImageFolder(test_path, transform=transform)


def evaluate_classification(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to("cuda", non_blocking=True)
            labels = labels.to("cuda", non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                predictions = model(images).argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.shape[0]

    return correct / total


def main(checkpoint_path, batch_size):
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    dataset_name = checkpoint["dataset"]
    # Older fine-tuning checkpoints stored the epoch only in the filename.
    epoch = checkpoint.get("epoch", int(checkpoint_path.stem.rsplit("_epoch_", 1)[1]))
    wandb_run_id = checkpoint["wandb_run_id"]

    test_dataset = get_dataset(config["image_size"], dataset_name)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=8, pin_memory=True,
                             persistent_workers=True, prefetch_factor=1)

    model = FineTuningModel(config, num_classes=len(checkpoint["classes"])).to("cuda")
    model.load_state_dict(checkpoint["model"])
    del checkpoint

    accuracy = evaluate_classification(model, test_loader)
    print(f"Dataset: {dataset_name}, Accuracy: {accuracy:.4f}", flush=True)
    metrics = {f"eval_test_accuracy/{dataset_name}": accuracy}
    write_eval_result(PROJECT_ROOT, checkpoint_path, epoch, metrics, wandb_run_id,
                      step_name="epoch")
    return accuracy
