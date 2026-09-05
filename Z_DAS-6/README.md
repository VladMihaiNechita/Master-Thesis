# DAS-6 helpers

These scripts retain the project's existing VU/DAS-6 account paths, node groups,
and scheduling conventions. They require access to that environment and the
existing remote `run_job.sh` wrapper at the project root. The wrapper, credentials,
and per-run monitoring state are not published.

Create ignored `Z_DAS-6/0_credentials.json` locally with this structure and your
actual passwords:

```json
{
  "jump_host_password": "YOUR_VU_GATEWAY_PASSWORD",
  "target_password": "YOUR_DAS6_PASSWORD"
}
```

For W&B synchronization, set `WANDB_API_KEY` or create ignored
`Z_DAS-6/0_wandb_api_key.txt`. Authenticate training separately in the remote
environment. Use the project virtual environment for local helper commands.

`python -B Z_DAS-6/main.py` copies local `src` to the configured remote `src`
directory and launches MAE on an available A6000. This replaces remote source;
run it only when that checkout is ready to be updated.

To watch and submit checkpoint evaluations from the repository root:

```sh
python -B -c "import sys; sys.path.insert(0, 'Z_DAS-6'); from small_scripts import evaluate_all_checkpoints_das6; evaluate_all_checkpoints_das6(experiment='MAE', mode='pretraining_evaluation')"
```

Use `experiment='I_JEPA'` for the frozen experiment and
`mode='fine_tuning_evaluation'` for fine-tuned checkpoints. The helper filters
checkpoint prefixes by experiment and stage and excludes resume files. Configured
model size in `src/run.py` must match the checkpoints. The in-memory submitted-job
list resets when the helper restarts.

`python -B Z_DAS-6/sync_on_wandb.py` uploads completed remote evaluation JSONs
locally and moves them into remote `eval_results_logged/`. The `complete` field
lets initial/final checkpoint results upload even when weights are retained.
Older pretraining JSONs without this field still use checkpoint deletion as the
completion signal; reevaluate retained checkpoints to produce marked results.
