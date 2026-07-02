from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_SEED = 1729

DEFAULT_MODEL_NAME = "vit_tiny_patch16_224"
DEFAULT_IMAGE_SIZE = 224
DEFAULT_PATCH_SIZE = 16
DEFAULT_IN_CHANS = 3
DEFAULT_VIT_EMBED_DIM = 192
DEFAULT_VIT_DEPTH = 12
DEFAULT_VIT_NUM_HEADS = 3
DEFAULT_VIT_MLP_RATIO = 4.0
DEFAULT_VIT_DROPOUT = 0.0

DEFAULT_EVAL_LINEAR_EPOCHS = 30
DEFAULT_EVAL_LINEAR_BATCH_SIZE = 8192
DEFAULT_EVAL_LINEAR_LR = 0.1
DEFAULT_EVAL_LINEAR_WEIGHT_DECAY = 1e-4
DEFAULT_EVAL_LINEAR_OPTIMIZER = "torch.optim.SGD"
DEFAULT_EVAL_LINEAR_MOMENTUM = 0.9
DEFAULT_EVAL_LINEAR_LOSS = "cross_entropy"
DEFAULT_EVAL_LINEAR_DEVICE = "auto"
DEFAULT_EVAL_PERCEPTRON_TRAINING = dict(
    epochs=DEFAULT_EVAL_LINEAR_EPOCHS,
    batch_size=DEFAULT_EVAL_LINEAR_BATCH_SIZE,
    lr=DEFAULT_EVAL_LINEAR_LR,
    weight_decay=DEFAULT_EVAL_LINEAR_WEIGHT_DECAY,
    optimizer=DEFAULT_EVAL_LINEAR_OPTIMIZER,
    momentum=DEFAULT_EVAL_LINEAR_MOMENTUM,
    loss=DEFAULT_EVAL_LINEAR_LOSS,
    device=DEFAULT_EVAL_LINEAR_DEVICE,
)

MODEL_PRESETS: dict[str, dict[str, Any]] = {
    "vit_tiny_patch16_224": dict(
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4.0,
        decoder_embed_dim=96,
        decoder_depth=2,
        decoder_num_heads=3,
        perceptron=dict(DEFAULT_EVAL_PERCEPTRON_TRAINING),
    ),
    "vit_small_patch16_224": dict(
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        decoder_embed_dim=192,
        decoder_depth=4,
        decoder_num_heads=6,
        perceptron=dict(DEFAULT_EVAL_PERCEPTRON_TRAINING),
    ),
}

DEFAULT_MASK_RATIO = 0.75
DEFAULT_DECODER_EMBED_DIM = 96
DEFAULT_DECODER_DEPTH = 2
DEFAULT_DECODER_NUM_HEADS = 3

DEFAULT_TRAIN_TARGET_INSTANCES = 10_000_000
DEFAULT_TRAIN_BATCH_SIZE = 1000
DEFAULT_TRAIN_STEPS = DEFAULT_TRAIN_TARGET_INSTANCES // DEFAULT_TRAIN_BATCH_SIZE
DEFAULT_TRAIN_LR = 1e-4
DEFAULT_TRAIN_MIN_LR = 0.0
DEFAULT_TRAIN_WARMUP_STEPS = 500
DEFAULT_TRAIN_WEIGHT_DECAY = 0.05
DEFAULT_TRAIN_BETAS = (0.9, 0.95)
DEFAULT_TRAIN_LOG_EVERY = 20
DEFAULT_TRAIN_VAL_EVERY = 0
DEFAULT_TRAIN_VALIDATION_INTERVAL_INSTANCES = 1_000_000
DEFAULT_TRAIN_VAL_BATCHES = 4
DEFAULT_TRAIN_CHECKPOINT_EVERY = 1000
DEFAULT_TRAIN_PREVIEW_EVERY = 0
DEFAULT_TRAIN_CLIP_GRAD = 1.0

DEFAULT_EVAL_BATCH_SIZE = 256
DEFAULT_EVAL_NUM_WORKERS = 4
SUPPORTED_EVAL_EMBEDDING_MODES = ("cls_mean",)
DEFAULT_EVAL_EMBEDDING_MODES = SUPPORTED_EVAL_EMBEDDING_MODES
SUPPORTED_EVAL_PROBES = ("linear",)
DEFAULT_EVAL_PROBES = ("linear",)
DEFAULT_WANDB_PROJECT = "up-images-vit"
DEFAULT_EVAL_OUTPUT_CSV = "results/eval.csv"
DEFAULT_EVAL_FEATURE_OUTPUT_CSV = "results/eval_from_features.csv"
DEFAULT_EVAL_MERGED_OUTPUT_CSV = "results/eval_merged.csv"
DEFAULT_OUTPUT_DIR = "results/eval"
DEFAULT_PREVIEW_OUTPUT_DIR = "results/previews"
DEFAULT_PREVIEW_BATCH_SIZE = 8
DEFAULT_PREVIEW_GRID_NROW = 4
DEFAULT_PREVIEW_MAX_IMAGES = 16
DEFAULT_RECONSTRUCTION_PREVIEW_MAX_IMAGES = 8
DEFAULT_PREVIEW_IRC_MIN_BUFFER = 128
DEFAULT_PREVIEW_IRC_BUFFER_MULTIPLIER = 4
DEFAULT_PREPROCESS_OUTPUT_DIR = "data/cache"
DEFAULT_PREPROCESS_SHARD_SIZE = 1024

DEFAULT_GENERATOR_NAME = "irc_conv"
DEFAULT_IID_RGB_NOISE_DISTRIBUTION = "uniform"
DEFAULT_GAUSSIAN_BLUR_SIGMA = 2.0
DEFAULT_FOURIER_COMPONENTS = 32
DEFAULT_FOURIER_MIN_FREQ = 1.0
DEFAULT_FOURIER_MAX_FREQ = 12.0
DEFAULT_FOURIER_COLOR_MIXING = True
DEFAULT_SIMPLE_SHAPES_MIN = 1
DEFAULT_SIMPLE_SHAPES_MAX = 3

DEFAULT_RANDOM_CONV_CHANNELS = 3
DEFAULT_RANDOM_CONV_WIDTH = 16
DEFAULT_RANDOM_CONV_DEPTH = 3
DEFAULT_RANDOM_CONV_KERNEL_SIZE = 3
DEFAULT_RANDOM_CONV_DILATION = 1
DEFAULT_RANDOM_CONV_MULTI_SCALE = False
DEFAULT_RANDOM_CONV_ACTIVATION = "gelu"
DEFAULT_RANDOM_CONV_WEIGHT_STD = 0.08
DEFAULT_RANDOM_CONV_WEIGHT_STD_MIN = DEFAULT_RANDOM_CONV_WEIGHT_STD
DEFAULT_RANDOM_CONV_WEIGHT_STD_MAX = DEFAULT_RANDOM_CONV_WEIGHT_STD

DEFAULT_IRC_K = 2
DEFAULT_IRC_MODE = "random_generator_per_batch"
DEFAULT_IRC_BUFFER_SIZE = 10000
DEFAULT_IRC_REFRESH_FRACTION = 0.05
DEFAULT_IRC_REFRESH_FRACTION_AFTER: float | None = None
DEFAULT_IRC_REFRESH_FRACTION_SWITCH_STEP = 0
DEFAULT_IRC_REFRESH_FRACTION_SWITCH_IMAGES: int | None = None
DEFAULT_IRC_REFRESH_FRACTION_SCHEDULE = "step"
DEFAULT_IRC_RESET_FRACTION = 0.001
DEFAULT_IRC_INIT_RANDOM_COLOR_FRACTION = 0.0
DEFAULT_IRC_RANDOM_COLOR_FRACTION = 0.0
DEFAULT_IRC_RANDOM_COLOR_FRACTION_AFTER: float | None = None
DEFAULT_IRC_RANDOM_COLOR_FRACTION_SWITCH_STEP = 0
DEFAULT_IRC_RANDOM_COLOR_FRACTION_SWITCH_IMAGES: int | None = None
DEFAULT_IRC_RANDOM_COLOR_FRACTION_SCHEDULE = "step"
DEFAULT_IRC_FOURIER_FRACTION = 0.05
DEFAULT_IRC_FOURIER_FRACTION_AFTER = 0.02
DEFAULT_IRC_FOURIER_FRACTION_SWITCH_STEP = 5000
DEFAULT_IRC_FOURIER_FRACTION_SWITCH_IMAGES: int | None = None
DEFAULT_IRC_FOURIER_FRACTION_SCHEDULE = "step"
DEFAULT_IRC_OUTPUT_FOURIER_FRACTION = 0.0
DEFAULT_IRC_OUTPUT_NOISE_FRACTION = 0.0
DEFAULT_IRC_UPDATE_STRENGTH = 0.3
DEFAULT_IRC_UPDATE_RULE = "blend"
DEFAULT_IRC_UPDATE_OUTPUT_ACTIVATION = "sigmoid"
DEFAULT_IRC_UPDATE_STRENGTH_AFTER: float | None = None
DEFAULT_IRC_UPDATE_STRENGTH_SWITCH_STEP = 0
DEFAULT_IRC_UPDATE_STRENGTH_SWITCH_IMAGES: int | None = None
DEFAULT_IRC_UPDATE_STRENGTH_SCHEDULE = "step"
DEFAULT_IRC_RESET_FRACTION_AFTER: float | None = None
DEFAULT_IRC_RESET_FRACTION_SWITCH_STEP = 0
DEFAULT_IRC_RESET_FRACTION_SWITCH_IMAGES: int | None = None
DEFAULT_IRC_RESET_FRACTION_SCHEDULE = "step"
DEFAULT_IRC_K_CHOICES: tuple[int, ...] | None = None
DEFAULT_IRC_QUANTIZE = True
DEFAULT_IRC_HORIZONTAL_FLIP_PROB = 0.5
DEFAULT_IRC_VERTICAL_FLIP_PROB = 0.0
DEFAULT_IRC_INIT_SOURCE = "noise"
DEFAULT_IRC_LOCAL_BUFFER_LIMIT = 256
DEFAULT_IRC_LATENT_PIXEL_CHANNELS = 0
DEFAULT_IRC_LATENT_LOWRES_CHANNELS = 0
DEFAULT_IRC_LATENT_LOWRES_SCALES = (14, 28, 56)
DEFAULT_IRC_LATENT_GLOBAL_CHANNELS = 0
DEFAULT_IRC_SPECTRAL_LATENT_CHANNELS = 0
DEFAULT_IRC_SPECTRAL_ALPHA_MIN = 0.7
DEFAULT_IRC_SPECTRAL_ALPHA_MAX = 2.0

DEFAULT_DATA_ROOT = "data"
DEFAULT_TRAIN_SPLIT = "train"
DEFAULT_EVAL_SPLIT = "test"
DEFAULT_EVAL_FRACTION = 0.2
DEFAULT_RESIZE_SHORTER_SIZE = 256
DEFAULT_DTD_PARTITION = 1
DEFAULT_FAKE_DATASET_SIZE = 128
DEFAULT_FAKE_DATASET_CLASSES = 10
DEFAULT_REAL_FAKE_DATASET_SIZE = 1024
DEFAULT_REAL_FAKE_DATASET_CLASSES = 100
DEFAULT_VIDEO_PATH = "data/video/Walking_in_Amsterdam_640x360_60fps.mp4"

FULL_RESIZE_DATASETS = {
    "cifar10",
    "cifar100",
    "tiny_imagenet",
    "tiny-imagenet",
    "tinyimagenet",
    "eurosat",
    "resisc45",
    "resisc_45",
}

CENTER_CROP_DATASETS = {
    "imagenet",
    "imagenet1k",
    "imagenet-1k",
    "imagenet_1k",
    "imagenet_r",
    "imagenet-r",
    "food101",
    "pets",
    "oxford_iiit_pets",
    "oxfordiiitpet",
    "cars",
    "stanford_cars",
    "dtd",
    "sun397",
}

STANDARD_IMAGEFOLDER_ROOTS = {
    "cifar10": "data/cifar10",
    "cifar100": "data/cifar100",
    "tiny_imagenet": "data/tiny-imagenet-200",
    "tiny-imagenet": "data/tiny-imagenet-200",
    "tinyimagenet": "data/tiny-imagenet-200",
    "food101": "data/food101",
    "pets": "data/pets",
    "oxford_iiit_pets": "data/pets",
    "oxfordiiitpet": "data/pets",
    "cars": "data/stanford-cars",
    "stanford_cars": "data/stanford-cars",
    "dtd": "data/dtd",
    "eurosat": "data/eurosat",
    "resisc45": "data/resisc45",
    "resisc_45": "data/resisc45",
    "sun397": "data/sun397",
    "imagenet": "data/imagenet",
    "imagenet1k": "data/imagenet",
    "imagenet-1k": "data/imagenet",
    "imagenet_1k": "data/imagenet",
}

DEFAULT_DATASET_EVAL_LOADER_OVERRIDES = {
    "imagenet": {"batch_size": 128, "num_workers": 0},
    "imagenet1k": {"batch_size": 128, "num_workers": 0},
    "imagenet-1k": {"batch_size": 128, "num_workers": 0},
    "imagenet_1k": {"batch_size": 128, "num_workers": 0},
    "imagenet_r": {"batch_size": 128, "num_workers": 0},
    "imagenet-r": {"batch_size": 128, "num_workers": 0},
}

DEFAULT_VIDEO_BUFFER_SIZE = 10000
DEFAULT_VIDEO_REFRESH_FRACTION = 0.125
DEFAULT_VIDEO_REFRESH_EVERY = 32
DEFAULT_VIDEO_SAMPLE_FPS = 2.0
DEFAULT_VIDEO_RESEEK_EVERY = 512
DEFAULT_VIDEO_BUFFER_SHORTER_SIZE = 256
DEFAULT_VIDEO_HORIZONTAL_FLIP_PROB = 0.5
DEFAULT_VIDEO_COLOR_JITTER_STRENGTH = 0.2
DEFAULT_VIDEO_RANDOM_GRAYSCALE_PROB = 0.05
DEFAULT_VIDEO_TEMPORAL_JITTER = True
DEFAULT_VIDEO_CROP_SCALE = (0.55, 1.0)
DEFAULT_VIDEO_CROP_RATIO = (0.75, 1.3333333333333333)
DEFAULT_VIDEO_DECODE_RETRIES = 8
DEFAULT_VIDEO_FALLBACK_FPS = 30.0

DEFAULT_SLURM_MODULES = (
    "2023",
    "PyTorch/2.1.2-foss-2023a-CUDA-12.1.1",
    "torchvision/0.16.0-foss-2023a-CUDA-12.1.1",
    "SciPy-bundle/2023.07-gfbf-2023a",
    "matplotlib/3.7.2-gfbf-2023a",
    "tqdm/4.66.1-GCCcore-12.3.0",
)
DEFAULT_SLURM_TRAIN_PARTITION = "gpu_h100"
DEFAULT_SLURM_TRAIN_GPUS = 1
DEFAULT_SLURM_TRAIN_CPUS = 16
DEFAULT_SLURM_TRAIN_TIME = "06:00:00"
DEFAULT_SLURM_EXTRACT_PARTITION = "gpu_h100"
DEFAULT_SLURM_EXTRACT_GPUS = 1
DEFAULT_SLURM_EXTRACT_CPUS = 18
DEFAULT_SLURM_EXTRACT_TIME = "08:00:00"
DEFAULT_SLURM_CPU_PARTITION = "genoa"
DEFAULT_SLURM_CPU_CPUS = 64
DEFAULT_SLURM_CPU_TIME = "04:00:00"
DEFAULT_SLURM_EVAL_PARTITION = "gpu_h100"
DEFAULT_SLURM_EVAL_GPUS = 1
DEFAULT_SLURM_EVAL_CPUS = 18
DEFAULT_SLURM_EVAL_TIME = "08:00:00"
DEFAULT_SLURM_MERGE_PARTITION = "genoa"
DEFAULT_SLURM_MERGE_CPUS = 4
DEFAULT_SLURM_MERGE_TIME = "00:30:00"


def normalized_name(name: str) -> str:
    return str(name).lower().replace("-", "_")


def dataset_eval_loader_override(name: str) -> dict[str, int]:
    normalized = normalized_name(name)
    for key, value in DEFAULT_DATASET_EVAL_LOADER_OVERRIDES.items():
        if normalized_name(key) == normalized:
            return dict(value)
    return {}


def eval_perceptron_training_config(eval_cfg: dict[str, Any], model_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    model_cfg = model_cfg or {}
    name = str(model_cfg.get("name", model_cfg.get("preset", DEFAULT_MODEL_NAME)))
    preset = MODEL_PRESETS.get(name, {})
    values = dict(DEFAULT_EVAL_PERCEPTRON_TRAINING)

    for source in (
        preset.get("perceptron", {}),
        model_cfg.get("perceptron", {}),
        eval_cfg.get("perceptron", {}),
        eval_cfg.get("linear_probe", {}),
    ):
        if isinstance(source, dict):
            for key in DEFAULT_EVAL_PERCEPTRON_TRAINING:
                if key in source:
                    values[key] = source[key]

    legacy_keys = {
        "linear_epochs": "epochs",
        "linear_batch_size": "batch_size",
        "linear_lr": "lr",
        "linear_weight_decay": "weight_decay",
        "linear_optimizer": "optimizer",
        "linear_momentum": "momentum",
        "linear_loss": "loss",
        "linear_device": "device",
    }
    for legacy_key, key in legacy_keys.items():
        if legacy_key in eval_cfg:
            values[key] = eval_cfg[legacy_key]
    return values


def validate_eval_probes(raw: Any) -> list[str]:
    if raw is None:
        values = DEFAULT_EVAL_PROBES
    elif isinstance(raw, str):
        values = (raw,)
    else:
        values = raw
    probes = [str(probe) for probe in values]
    unsupported = [probe for probe in probes if probe not in SUPPORTED_EVAL_PROBES]
    if unsupported:
        supported = ", ".join(SUPPORTED_EVAL_PROBES)
        raise ValueError(f"Unsupported eval probe(s): {unsupported}. Supported probes: {supported}")
    return probes


def validate_eval_embedding_modes(raw: Any) -> list[str]:
    if raw is None:
        values = DEFAULT_EVAL_EMBEDDING_MODES
    elif isinstance(raw, str):
        values = (raw,)
    else:
        values = raw
    modes = [str(mode) for mode in values]
    unsupported = [mode for mode in modes if mode not in SUPPORTED_EVAL_EMBEDDING_MODES]
    if unsupported:
        supported = ", ".join(SUPPORTED_EVAL_EMBEDDING_MODES)
        raise ValueError(f"Unsupported eval embedding mode(s): {unsupported}. Supported modes: {supported}")
    return modes


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    data["_config_path"] = str(path)
    return data


def save_config(config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def get_nested(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    cur: Any = config
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur
