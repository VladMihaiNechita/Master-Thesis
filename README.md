# Universal Pre-training for Images

Code for my master's thesis, [Universal Pre-training for Images](Universal%20Pre-training%20for%20Images.pdf).
The question is whether a Vision Transformer can learn useful visual features from
random computation, without real images during pre-training.

The default experiment trains a ViT-Tiny with MAE for 100 million image
presentations. The IRC generator starts with pixel noise and solid colors, then
repeatedly applies crop-and-resize, blur, crossover, and random residual CNNs to
images in a persistent buffer.

The thesis reports frozen linear probes, reconstruction, and fine-tuning on
ImageNet-100 and a 10% ImageNet-1K training subset. It also includes an
[I-JEPA comparison](src/experiments/I_JEPA/README.md).

## Setup

An NVIDIA GPU is required. From the repository root:

```sh
python -m venv .venv
# Activate .venv for your shell.
python -m pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
wandb login
```

The pinned versions record the experiment environment. Datasets and checkpoints
are not included. Put datasets in ImageFolder layout:
`datasets/<dataset>/train/<class>/` and `datasets/<dataset>/test/<class>/`.
For the ImageNet experiments, `test/` contains the corresponding validation images.
Checkpoints go in `checkpoints/`.

## Run MAE

```sh
# Pre-train, then evaluate a saved checkpoint.
python -B src/run.py --experiment=MAE --mode=pretraining_main
python -B src/run.py --experiment=MAE --mode=pretraining_evaluation --checkpoint_path=model_tiny_generator_irc_new_images_seen_100000000.pth

# Fine-tune on the ImageNet-1K 10% training subset, then evaluate.
python -B src/run.py --experiment=MAE --mode=fine_tuning_main --pretraining_checkpoint_path=model_tiny_generator_irc_new_images_seen_100000000.pth --fine_tuning_dataset=imagenet1k-simclr-10pct --fine_tuning_layer_decay=1.0
python -B src/run.py --experiment=MAE --mode=fine_tuning_evaluation --checkpoint_path=fine_tuned_model_tiny_generator_irc_new_images_seen_100000000_imagenet1k-simclr-10pct_layer_decay_1.0_epoch_300.pth
```

These commands use layer decay 1.0, as in the thesis; the fine-tuning YAML defaults
to 0.95. Use `--fine_tuning_dataset=imagenet100-cmc` for ImageNet-100 and the
`images_seen_0.pth` checkpoint for the no-pre-training comparison.

Configs are under `src/experiments/MAE/`; `size` in `src/run.py` selects the model
(default: `tiny`). Adjust batch sizes for your GPU. Pre-training evaluation uses
the validation dataset group by default; choose `*test_datasets`, `eval_test`, and
a separate `result_suffix` in the evaluation YAML for the test group.

Evaluation saves JSON files in `eval_results/`. Set
`delete_checkpoints_after_evaluation: false` to keep intermediate checkpoints,
including when evaluating both groups. The default deletes them after evaluation.
