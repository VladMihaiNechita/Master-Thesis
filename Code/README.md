# IRC ViT

Small research codebase for pretraining a Vision Transformer target model on
synthetic image streams from iterated random computation.

The repository is deliberately compact. Code lives under `src/irc_vit`. Local
datasets can be placed under `data/`, and short local runs create temporary
outputs under `results/` when needed. Keep both paths out of Git; this local
checkout uses `.git/info/exclude` for those ignore rules.

## Layout

```text
configs/default.yaml     tiny local smoke-test config
configs/default_irc.yaml default full IRC generator run
configs/default_video.yaml default full real-video run
src/irc_vit/model.py     ViT encoder and MAE target model
src/irc_vit/train.py     target-model training entry point
src/irc_vit/evaluate.py  frozen-representation evaluation
src/irc_vit/generators/  separate generator implementations
src/irc_vit/data.py      real-data loaders and eval dataset builders
src/irc_vit/utils.py     small shared helpers
tests/                   local CPU-friendly tests
data/                    optional local datasets and raw assets, local-only
results/                 temporary local outputs from smoke runs, local-only
```

Local-only folders kept outside Git:

```text
Papers/
Z_DAS-6/
Z_Snellius/
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Local Checks

These commands may create `results/` temporarily. Delete those outputs after
inspection unless they are intentionally being kept.

```bash
python -m pytest -q
python -m src.irc_vit.preview --out results/previews --image-size 64 --batch-size 8
python -m src.irc_vit.train --config configs/default.yaml
```

Evaluate a trained checkpoint:

```bash
python -m src.irc_vit.evaluate \
  --config configs/default.yaml \
  --checkpoint results/smoke_irc_k4/checkpoint_final.pt \
  --out results/eval.csv
```

Evaluate an untrained baseline:

```bash
python -m src.irc_vit.evaluate \
  --config configs/default.yaml \
  --checkpoint random_init \
  --out results/random_init_eval.csv
```

For large datasets, especially ImageNet-1K and ImageNet-R, use the cached
feature path. First extract frozen features on a GPU, then run the PyTorch
linear probes from those feature tensors:

```bash
python -m src.irc_vit.extract_feature_cache \
  --config configs/eval_all_datasets.yaml \
  --checkpoint results/run/checkpoint_final.pt \
  --out results/run/features \
  --reconstruction-out results/run/reconstruction_eval.csv

python -m src.irc_vit.evaluate_feature_cache \
  --config configs/eval_all_datasets.yaml \
  --checkpoint results/run/checkpoint_final.pt \
  --features results/run/features \
  --out results/run/probe_eval.csv \
  --disable-wandb
```

The feature cache stores one tensor file per checkpoint, dataset split, and
embedding mode. ImageNet-R reuses the ImageNet-1K train feature cache for its
linear probe, then evaluates on the ImageNet-R test features. Merge the
reconstruction and probe CSVs with `src.irc_vit.log_eval_csv_to_wandb` when
logging split jobs back to W&B.

On Snellius, the fastest full-run layout is:

```text
H100 job       train the MAE/ViT target model
H100 jobs      extract cached frozen features, and optionally reconstruction metrics
H100 jobs      run grouped PyTorch linear probes from cached features
CPU jobs       merge each checkpoint's CSV shards and log W&B as soon as it finishes
CPU job        write the final merged CSV without duplicating W&B points
```

Generate this dependency-linked Slurm graph from the cluster checkout with one
of the full default configs, for example:

```bash
python -m src.irc_vit.submit_slurm_pipeline \
  --config configs/default_irc.yaml \
  --group-eval-by-checkpoint \
  --submit
```

If evaluating an existing checkpoint, pass `--checkpoint path/to/checkpoint.pt`.
The command writes scripts under the run's `pipeline/slurm/` directory. Without
`--submit`, it only writes the scripts and prints the `sbatch` commands.

## Config

The project keeps one tiny smoke-test config and two full default-run configs:

```text
configs/default.yaml        local smoke test, deliberately tiny
configs/default_irc.yaml    full IRC generator default
configs/default_video.yaml  full real-video default
```

YAML is used because experiment settings are data, not implementation code. It
is easy to diff, serialize into `results/.../config.yaml`, pass through Slurm,
and log to W&B. Python config files are more flexible, but they can execute
arbitrary code and tend to mix experiment settings with implementation logic.

The main fields are:

```text
model   ViT target-model size and image resolution
mae     masking and decoder hyperparameters
source  generator or real-data pretraining source
train   optimizer, batch size, logging, checkpoint cadence
eval    frozen evaluation settings
wandb   optional W&B logging
```

## Default Full Runs

The default IRC run is `configs/default_irc.yaml`.

```text
source:        irc_conv
target model:  ViT-Tiny/16, 224x224 images
objective:     MAE, mask_ratio=0.75
training:      10M images, batch_size=1000, lr=1e-4
checkpoints:   every 1000 steps = every 1M images
evaluation:    linear probe on cls_mean, reconstruction raw/patch-normalized MAE/MSE
datasets:      all 16 datasets, including synthetic validation datasets
```

The IRC generator defaults are:

```text
k=2, mode=random_generator_per_batch, width=16, depth=3
buffer_size=20000, refresh_fraction=0.025
reset_fraction=0.001 -> 0.0001 linearly over 10M images
fourier_fraction=0.05 -> 0.05 over 10M images
update_strength=0.3, quantize=true, horizontal_flip_prob=0.5
```

The default video run is `configs/default_video.yaml`.

```text
source:        real video
target model:  ViT-Tiny/16, 224x224 images
objective:     MAE, mask_ratio=0.75
training:      20M images, batch_size=6250, lr=1e-4
checkpoints:   every 160 steps = every 1M images
evaluation:    linear probe on cls_mean, reconstruction raw/patch-normalized MAE/MSE
datasets:      all 16 datasets, including synthetic validation datasets
```

The video-source defaults are:

```text
path=data/video/Walking_in_Amsterdam_640x360_60fps.mp4
buffer_size=10000, refresh_fraction=0.125, refresh_every=32
sample_fps=2.0, reseek_every=512, buffer_shorter_size=256
horizontal_flip_prob=0.5, color_jitter_strength=0.2
random_grayscale_prob=0.05, temporal_jitter=true
crop_scale=[0.55, 1.0], crop_ratio=[0.75, 1.3333333333333333]
```

For a full end-to-end Snellius run with eval at every checkpoint, pass the
expected checkpoint paths to the Slurm pipeline before training starts. For
the video default:

```bash
RUN=default_video_20m_all_datasets_vit_tiny_b6250
BASE=/scratch-shared/vnechita/Master-Thesis/results/$RUN
CKPTS=()
for step in $(seq 160 160 3200); do
  CKPTS+=("$BASE/checkpoint_step_$(printf '%07d' "$step").pt")
done

python -m src.irc_vit.submit_slurm_pipeline \
  --config configs/default_video.yaml \
  --checkpoints "${CKPTS[@]}" \
  --group-eval-by-checkpoint \
  --extract-mem 128G \
  --eval-partition gpu_h100 \
  --eval-gpus 1 \
  --eval-cpus 18 \
  --eval-time 08:00:00 \
  --eval-mem 128G \
  --merge-time 00:45:00 \
  --submit
```

For the IRC default:

```bash
RUN=default_irc_10m_all_datasets_vit_tiny_b1000
BASE=/scratch-shared/vnechita/Master-Thesis/results/$RUN
CKPTS=()
for step in $(seq 1000 1000 10000); do
  CKPTS+=("$BASE/checkpoint_step_$(printf '%07d' "$step").pt")
done

python -m src.irc_vit.submit_slurm_pipeline \
  --config configs/default_irc.yaml \
  --checkpoints "${CKPTS[@]}" \
  --group-eval-by-checkpoint \
  --submit
```

The main experiments use `224x224` inputs with a `16x16` ViT patch size. The
same image size is used for synthetic generators and evaluation datasets.
The smoke config is intentionally small; use the two full default configs above
for paper-style runs.

To switch generator, edit:

```yaml
source:
  type: generator
  generator:
    name: fourier_texture
    params:
      components: 32
```

Available generator names:

```text
iid_rgb_noise
gaussian_blur_noise
fourier_texture
simple_shapes
irc_conv
```

For the real-data source baseline, use the local video instead of CIFAR-100 or
other small classification datasets:

```yaml
source:
  type: real
  params:
    name: video
    path: data/video/Walking_in_Amsterdam_640x360_60fps.mp4
    buffer_size: 4096
    refresh_fraction: 0.125
    refresh_every: 32
    sample_fps: 2.0
    reseek_every: 512
    crop_scale: [0.55, 1.0]
```

The video source keeps a CPU buffer of decoded frames, samples random resized
crops from that buffer, and periodically refreshes part of it with temporally
strided frames. It seeks to a random point only once per chunk, then decodes
sequentially for efficiency. This avoids training on near-duplicate consecutive
video frames while keeping the source natural-image-like.

CIFAR-10, CIFAR-100, and STL10 are reserved for evaluation or smoke checks; the
real-data pretraining path rejects them to avoid accidentally training the
target model on a classification benchmark.

## Data

Datasets are local-only. The `data/` directory may not exist after cloning; make
it locally only when needed. Keep only `data/README.md` in git; it documents the
pretraining video, the 16 evaluation datasets, and their preprocessing
protocols. Common paths:

```text
data/cifar10
data/cifar100
data/tiny-imagenet-200
data/imagenet
data/imagenet-r
data/food101
data/pets
data/stanford-cars
data/dtd
data/eurosat
data/resisc45
data/sun397
data/video
```

For repeated evaluation, preprocess datasets once into tensor shards:

```powershell
.\.venv\Scripts\python.exe -m src.irc_vit.preprocess_data --config configs/default.yaml --out data/cache --datasets cifar10 --overwrite
```

On Snellius, keep both raw data and `data/cache` on scratch, for example under
`/scratch-shared/vnechita/Master-Thesis/`, and symlink the needed subfolders
from the repository checkout.

Evaluation datasets use two preprocessing policies:

```text
full_resize  resize the full image directly to 224x224 with bicubic antialiasing
center_crop  resize shorter side to 256, then center crop 224x224
```

The default benchmark suite is:

```text
CIFAR-10       full_resize, data/cifar10
CIFAR-100      full_resize, data/cifar100
Tiny-ImageNet  full_resize, data/tiny-imagenet-200
ImageNet-1K    center_crop, data/imagenet
ImageNet-R     center_crop, data/imagenet-r; probe trained on ImageNet-1K
Food-101       center_crop, data/food101
Oxford Pets    center_crop, data/pets
Stanford Cars  center_crop, data/stanford-cars
DTD            center_crop, data/dtd
EuroSAT RGB    full_resize, data/eurosat
ShapeColorObjects generated synthetic validation dataset
LineFieldOrientation generated synthetic validation dataset
TextureMosaic generated synthetic validation dataset
CheckerGridField generated synthetic validation dataset
RESISC45       full_resize, data/resisc45
SUN397         center_crop, data/sun397
```

ImageNet-1K is not auto-downloaded; put the licensed dataset under
`data/imagenet` before running ImageNet-1K or ImageNet-R evaluations.

Do not commit datasets, checkpoints, W&B cache files, or generated results.

## Cluster Runs

For longer experiments, clone this repository on Snellius or DAS-6 and install
the requirements in a fresh environment. The Codeberg copy intentionally does
not include private helper folders such as `Z_Snellius/` or `Z_DAS-6/`.

On Snellius, prefer H100 GPU nodes for GPU training jobs and choose batch sizes
that use the node efficiently. Use grouped cached-feature evaluation for
paper-scale runs so PyTorch linear probes run on GPU without thousands of small
CPU eval jobs.

W&B credentials are not stored in this repository. Authenticate on the cluster
with `wandb login` or set `WANDB_API_KEY` in the job environment when logging is
enabled.

## Codeberg Scope

Codeberg is used as the minimal cluster-run mirror. Keep only files needed to
run the experiments on Snellius or DAS-6: source code, configs, requirements,
tests or smoke checks, and concise run instructions.

Do not push local agent helpers, private notes, `data/`, `results/`, `Papers/`,
`Z_DAS-6/`, `Z_Snellius/`, checkpoints, W&B files, or credentials to Codeberg.
The Codeberg mirror intentionally does not track `.gitignore` or
`.gitattributes`; keep local ignore rules in `.git/info/exclude`.
