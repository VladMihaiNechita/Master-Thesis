# Universal Pretraining for Images

MAE is the active experiment. The I-JEPA ViT-Tiny experiment is frozen for possible
future use; its commands and artifact references are in the
[I-JEPA README](src/experiments/I_JEPA/README.md). The `ijepa-final` Git tag preserves
the complete source snapshot, including shared model and generator code.

## Setup

Use a project virtual environment and a CUDA-capable PyTorch installation.
The training and evaluation entry points require an NVIDIA GPU; their full
workflows are intended for the configured DAS-6 environment.

```sh
python -m venv .venv
# Activate .venv for your shell before running the following commands.
python -m pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
wandb login
```

The dependency versions were recorded from the existing environment; a fresh
installation and full GPU run were not validated as part of the freeze.
Use `python -B` to avoid leaving Python bytecode caches.

Datasets use ImageFolder layout: `datasets/<dataset>/train/<class>/` and
`datasets/<dataset>/test/<class>/`. Dataset preparation and weights are external
to this repository. Store checkpoints in `checkpoints/`.

## MAE

Run from the repository root:

```sh
python -B src/run.py --experiment=MAE --mode=pretraining_main
python -B src/run.py --experiment=MAE --mode=pretraining_evaluation --checkpoint_path=model_tiny_generator_irc_new_images_seen_100000000.pth
python -B src/run.py --experiment=MAE --mode=fine_tuning_main --pretraining_checkpoint_path=model_tiny_generator_irc_new_images_seen_100000000.pth --fine_tuning_dataset=imagenet1k-simclr-10pct
python -B src/run.py --experiment=MAE --mode=fine_tuning_evaluation --checkpoint_path=fine_tuned_model_tiny_generator_irc_new_images_seen_100000000_imagenet1k-simclr-10pct_layer_decay_0.95_epoch_300.pth
```

These filenames describe outputs of the checked-in configuration; use your actual
checkpoint names. Configs are under `src/experiments/MAE/`. Model size is selected
by `size` in `src/run.py` (default: `tiny`). The `--gpu_type` argument is currently
informational and does not change batch sizes.

Evaluation writes JSON files to `eval_results/`. Pretraining evaluation saves
progress after each dataset and marks the final result `complete: true` for W&B
upload. Intermediate pretraining checkpoints and their resume files are deleted
after successful evaluation by default; turn off
`delete_checkpoints_after_evaluation` in the evaluation YAML to retain them.
Initial and configured final checkpoint filenames are retained automatically.

See [DAS-6 helpers](Z_DAS-6/README.md) for launching and result synchronization.
Credentials, checkpoints, datasets, and generated results are intentionally ignored
by Git. Keep separate backups of artifacts needed to reproduce experiments.
