# I-JEPA: frozen ViT-Tiny experiment

This experiment is retained for possible future work. MAE is the active default.
Restore Git tag `ijepa-final` in a separate checkout to recover this version of
I-JEPA together with its shared model, generator, and evaluation code.

## Included pipeline

- Pretraining with a context encoder, predictor, and frozen EMA target encoder.
- Frozen-target-encoder linear probing on the same datasets as MAE.
- Supervised fine-tuning initialized from the target encoder, and final evaluation.
- Optimizer/generator/RNG resume state, W&B training logs, and evaluation JSONs.

The checked-in tiny configuration uses 224-pixel images, 16-pixel patches, a
192-wide/12-layer encoder, and a 128-wide/6-layer predictor. It uses `irc_new`,
four target blocks, a global batch of 4,002 (microbatch 1,334), 100,050,000 images,
and checkpoints every 4,002,000 images. Fine-tuning uses 300 epochs, batch 1,024,
and layer decay 1.0. The YAMLs are the full configuration reference.

These describe the frozen code defaults. Historical runs must be interpreted
using the `config` and `fine_tuning_config` stored in their checkpoints/W&B;
their exact code/config correspondence was not independently verified at freeze.

## Run and resume

Use the project virtual environment from the repository root, with the setup and
dataset layout described in the root README. Checkpoint paths below are relative
to `checkpoints/`; substitute actual filenames when needed.

```sh
python -B src/run.py --experiment=I_JEPA --mode=pretraining_main
python -B src/run.py --experiment=I_JEPA --mode=pretraining_evaluation --checkpoint_path=ijepa_model_tiny_generator_irc_new_images_seen_100050000.pth
python -B src/run.py --experiment=I_JEPA --mode=fine_tuning_main --pretraining_checkpoint_path=ijepa_model_tiny_generator_irc_new_images_seen_100050000.pth --fine_tuning_dataset=imagenet1k-simclr-10pct
python -B src/run.py --experiment=I_JEPA --mode=fine_tuning_evaluation --checkpoint_path=fine_tuned_ijepa_model_tiny_generator_irc_new_images_seen_100050000_imagenet1k-simclr-10pct_layer_decay_1.0_epoch_300.pth
```

To resume either training stage, add `--checkpoint_path=<saved-training-checkpoint.pth>`
to that stage's command. Keep its matching `<stem>_resume.pth` beside it. Resume
loads the saved configuration and W&B run; it continues that run's existing budget.
Resuming the final checkpoint does not extend training beyond the saved budget.
For the random-initialization baseline, fine-tune the `images_seen_0.pth` checkpoint.

Linear-probe evaluation uses the validation dataset list by default. Select
`*test_datasets`, `eval_test`, and a distinct `result_suffix` in
`pretraining/configs/tiny_pretraining_evaluation.yaml` for the held-out dataset list.
Set `delete_checkpoints_after_evaluation: false` when evaluating checkpoints you
want to keep beyond the automatically retained initial/final checkpoints.

## Artifact references

Known DAS-6 project location:
`/var/scratch/gkl505/Universal Pretraining for Images`.
Weights/resume files belong in `checkpoints/`; metrics are in `eval_results/` and
`eval_results_logged/`. W&B project: `universal-pretraining-for-images`.

Keep the initial and final `ijepa_model_tiny_generator_irc_new_images_seen_*.pth`
files (0 and 100050000), the final pretraining `_resume.pth`, the epoch-300
fine-tuned weights and any needed resume files, and the corresponding evaluation
JSONs. Weights and results are not included in Git.

The existing local monitor records these fine-tuning run IDs:

| Dataset | Pretraining images | W&B run ID |
| --- | ---: | --- |
| imagenet100-cmc | 0 | wzc9srl9 |
| imagenet1k-simclr-10pct | 0 | jsa4wfwi |
| imagenet100-cmc | 100050000 | n09hayi4 |
| imagenet1k-simclr-10pct | 100050000 | xj3luvjz |

The pretraining W&B run ID is stored in its checkpoint's `wandb_run_id` field.
Remote artifact presence, backups, and run results could not be checked during
finalization on 2026-09-05 because DAS-6 authentication failed. These are recorded
locations and identifiers, not a verified backup manifest.

## Scope and verification

Only ViT-Tiny configs are supplied. Pretraining runs eagerly; fine-tuning uses
`torch.compile` unless `DISABLE_COMPILE=1`. No pixel reconstruction metric is
provided because the predictor targets embeddings. Fine-tuning reuses MAE's
augmentation module, so use the tag when later MAE changes affect shared code.

The default pretraining batch/microbatch sizes divide correctly for one GPU or
three SLURM ranks. Distributed training and exact distributed resume were not
validated in this freeze. Match the original world size and saved configuration
when resuming. Full GPU training/evaluation was not rerun; syntax and CPU model
forward/backward, mask separation, and target-to-fine-tuning loading were checked.
