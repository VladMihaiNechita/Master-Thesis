import numpy, random, torch, yaml


def load_config(path):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def set_seed(seed=1729):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Unused
def evaluate_batch_sizes(reconstruction_batch_size, feature_batch_size, datasets_to_evaluate):
    # I want to calculate the number of passes for reconstruction and feature extraction based on the batch sizes and the datasets to evaluate.
    # next_reconstruction_batch_size is the smallest batch size that lowers the number of passes for reconstruction
    # next_feature_extraction_batch_size is the smallest batch size that lowers the number of passes for feature extraction
    dataset_sizes = {
        "cifar10": {"train": 50_000, "test": 10_000},
        "cifar100": {"train": 50_000, "test": 10_000},
        "dtd": {"train": 3_760, "test": 1_880},
        "eurosat": {"train": 21_600, "test": 5_400},
        "food101": {"train": 75_750, "test": 25_250},
        "line-field-orientation": {"train": 49_992, "test": 9_996},
        "pets": {"train": 3_680, "test": 3_669},
        "resisc45": {"train": 25_200, "test": 6_300},
        "shape-color-objects": {"train": 50_000, "test": 10_000},
        "stanford-cars": {"train": 8_144, "test": 8_041},
        "sun397-partition1": {"train": 19_850, "test": 19_850},
        "texture-mosaic": {"train": 50_000, "test": 10_000},
        "tiny-imagenet-200": {"train": 100_000, "test": 10_000},
    }

    reconstruction_sizes = [dataset_sizes[name]["test"] for name in datasets_to_evaluate]
    feature_extraction_sizes = [dataset_sizes[name][split]
                                for name in datasets_to_evaluate
                                for split in ("train", "test")]

    def number_of_passes(batch_size, sizes):
        return sum((size + batch_size - 1) // batch_size for size in sizes)

    def next_batch_size(batch_size, sizes):
        current_number_of_passes = number_of_passes(batch_size, sizes)
        for candidate_batch_size in range(batch_size + 1, max(sizes) + 1):
            if number_of_passes(candidate_batch_size, sizes) < current_number_of_passes:
                return candidate_batch_size
        return None

    reconstruction_number_of_passes = number_of_passes(reconstruction_batch_size, reconstruction_sizes)
    next_reconstruction_batch_size = next_batch_size(reconstruction_batch_size, reconstruction_sizes)
    feature_extraction_number_of_passes = number_of_passes(feature_batch_size, feature_extraction_sizes)
    next_feature_extraction_batch_size = next_batch_size(feature_batch_size, feature_extraction_sizes)

    return reconstruction_number_of_passes, next_reconstruction_batch_size, feature_extraction_number_of_passes, next_feature_extraction_batch_size