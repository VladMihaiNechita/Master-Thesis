# Data

This directory is intentionally local-only. Do not commit datasets, archives,
checkpoints, generated samples, or evaluation outputs. The only file that should
be tracked from `data/` is this README.

On Snellius, store data on scratch:

```text
/scratch-shared/vnechita/Master-Thesis/data
/scratch-shared/vnechita/Master-Thesis/data_cache
```

The repository checkout can then symlink the large local-only folders into
`data/`.

## Pretraining Video Source

The real-data pretraining baseline uses a local video instead of CIFAR-100 or
other small classification datasets:

```text
data/video/Walking_in_Amsterdam_640x360_60fps.mp4
```

The video is not committed. The training code decodes a CPU buffer of temporally
strided frames, samples random resized crops from the buffer, applies light color
and flip augmentation, and periodically refreshes part of the buffer with new
frames. It seeks to a random point only once per chunk and then decodes
sequentially, which is much faster than random-seeking every frame. This makes
the source less redundant than consecutive video frames while keeping natural
image statistics.

CIFAR-10, CIFAR-100, and STL10 are evaluation/smoke-test datasets only. Do not
use them as the real-data pretraining source.

All target-model evaluation images are converted to `224x224` for a ViT with
`16x16` patches. Two preprocessing policies are used:

- `full_resize`: resize the full image directly to `224x224` with bicubic
  interpolation and antialiasing. This is used when the original image is small
  or when whole-scene context matters.
- `center_crop`: resize the shorter side to `256`, then take a centered
  `224x224` crop. This is used for normal variable-resolution natural images.

Serious experiment configs evaluate on the full 16-dataset suite below. Use
CIFAR-10 alone only for smoke tests or very early debugging.

## Preprocessed Cache

To avoid repeating PIL decoding and resizing at the beginning of every run,
datasets can be preprocessed once into uint8 tensor shards:

```powershell
.\.venv\Scripts\python.exe -m src.irc_vit.preprocess_data --config configs/default.yaml --out data/cache --datasets cifar10 --overwrite
```

The loader automatically uses the cache when `metadata.json` exists for both the
train and test splits. Otherwise it falls back to the raw dataset.

Local raw benchmark datasets use a consistent ImageFolder-style layout:

```text
data/<dataset>/train/<class>/*
data/<dataset>/test/<class>/*
```

`test/` is the final evaluation split used for paper plots and tables. It may
come from an official dataset test split, an official validation split, or a
deterministic split when no official split is provided. Do not tune
hyperparameters separately on this split.

## Dataset Suite

| Dataset | Role in the paper | Local path | Protocol |
| --- | --- | --- | --- |
| CIFAR-10 | Very simple object-recognition sanity check. 60k color images, 10 classes, original size `32x32`. | `data/cifar10` | Official train/test split. Apply `full_resize`; no crop. |
| CIFAR-100 | Harder low-resolution object/category structure. 60k color images, 100 classes, original size `32x32`. | `data/cifar100` | Official train/test split. Apply `full_resize`; no crop. |
| Tiny-ImageNet | Medium difficulty, ImageNet-like, low-resolution benchmark. 200 classes, `64x64` images. | `data/tiny-imagenet-200` | Official train plus labeled validation split stored internally as `test`. Apply `full_resize`; no crop. |
| ImageNet-1K | Main credibility benchmark. 1,000 classes, about 1.28M train and 50k validation images. | `data/imagenet` | Manual/licensed dataset. Official train plus official validation split stored internally as `test`. Apply `center_crop`. |
| ImageNet-R | OOD/style robustness: art, cartoons, sketches, and renditions. About 30k images over 200 ImageNet classes. | `data/imagenet-r` | Evaluation-only `test` split. Do not train a new probe on ImageNet-R; train on ImageNet-1K and evaluate on overlapping classes. Apply `center_crop`. |
| Food-101 | Fine-grained real-world object recognition. 101 food classes, 101k images. | `data/food101` | Official train/test split. Apply `center_crop`. |
| Oxford-IIIT Pets | Fine-grained animals with pose, scale, and lighting variation. 37 categories, roughly 200 images per class. | `data/pets` | Official `trainval` stored as `train`; official `test` stored as `test`. Apply `center_crop`. |
| Stanford Cars | Fine-grained man-made objects. 196 classes, 16,185 images. | `data/stanford-cars` | Official train/test split. Apply `center_crop`. |
| DTD | Texture representation rather than object semantics. 47 texture categories, 5,640 images. | `data/dtd` | Official partition 1 train/test split. Apply `center_crop`. |
| EuroSAT RGB | Simple remote-sensing / satellite domain-shift benchmark. 10 classes, 27k RGB images, original size `64x64`. | `data/eurosat` | No official split; deterministic class-balanced 80/20 train/test split with seed `1729`. Apply `full_resize`; no crop. |
| ShapeColorObjects | Synthetic shape/color/object sanity benchmark. 8 classes: circle, square, triangle, ellipse, star, ring, cross, polygon. | generated or `data/cache/shape_color_objects` | Generated lazily at `224x224` with 50k train and 10k test examples by default. |
| LineFieldOrientation | Synthetic orientation and frequency benchmark. 12 classes from 4 orientations and 3 line frequencies. | generated or `data/cache/line_field_orientation` | Generated lazily with high-resolution rendering and antialiasing to `224x224`; 49,992 train and 9,996 test examples by default for exact class balance. |
| TextureMosaic | Synthetic local texture and region-structure benchmark. 8 procedural texture classes. | generated or `data/cache/texture_mosaic` | Generated lazily at `224x224` with irregular texture regions; 50k train and 10k test examples by default. |
| CheckerGridField | Synthetic periodic/grid benchmark. 12 classes from 3 grid types and 4 cell sizes. | generated or `data/cache/checker_grid_field` | Generated lazily with high-resolution rendering and antialiasing to `224x224`; 49,992 train and 9,996 test examples by default for exact class balance. |
| RESISC45 | Harder remote-sensing scene-recognition benchmark. 45 classes, 31,500 RGB images, original size `256x256`. | `data/resisc45` | No official split; deterministic class-balanced 80/20 train/test split with seed `1729`. Apply `full_resize` to preserve whole-scene context. |
| SUN397 | Large-scale scene/place recognition. 397 scene categories, about 108k images. | `data/sun397` | No official split found locally; deterministic class-balanced 80/20 train/test split with seed `1729`. Apply `center_crop`. |

The synthetic datasets are deterministic from `(dataset name, split, seed,
index)`, so they can be regenerated on the fly or preprocessed into tensor
shards with `src.irc_vit.preprocess_data`. Their labels cycle through the class
set in order. The 8-class synthetic datasets use exactly 50,000 train and
10,000 test examples; the 12-class datasets use the nearest lower exactly
balanced sizes because 50,000 and 10,000 are not divisible by 12.
The normal dataset API returns `(image, label)` for compatibility with the
linear-probe and preprocessing code. Synthetic datasets also expose
`sample_with_mask(index)` for mask inspection. ShapeColorObjects masks use
`0` for background and `class_id + 1` for visible object shapes; the region
datasets use class ids directly for every pixel region.

## Expected Layout

```text
data/
  README.md
  video/
    Walking_in_Amsterdam_640x360_60fps.mp4
  cache/
    cifar10/
      train/
      test/
  cifar10/
    train/
    test/
  cifar100/
    train/
    test/
  dtd/
    train/
    test/
  eurosat/
    train/
    test/
  food101/
    train/
    test/
  imagenet/
    train/
    test/
  imagenet-r/
    test/
  pets/
    train/
    test/
  resisc45/
    train/
    test/
  stanford-cars/
    train/
    test/
  sun397/
    train/
    test/
  tiny-imagenet-200/
    train/
    test/
```

`ImageNet-1K` must be placed manually because it is licensed. The other datasets
can be downloaded or converted locally, but the resulting files should remain
outside git and Codeberg.
