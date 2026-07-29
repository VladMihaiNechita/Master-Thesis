import json, wandb
from pathlib import Path


def write_eval_result(project_root, checkpoint_path, step, metrics, wandb_run_id,
                      result_suffix="", step_name="images_seen"):
    results_dir = Path(project_root) / "eval_results"
    results_dir.mkdir(exist_ok=True)

    result = {
        step_name: step,
        "checkpoint": Path(checkpoint_path).name,
        "wandb_run_id": wandb_run_id,
        "metrics": metrics,
    }

    # This is an atomic-write pattern
    # Another process only sees the final .json after writing has completed, so it cannot accidentally read a partially written file
    result_path = results_dir / f"{Path(checkpoint_path).stem}{result_suffix}.json"
    tmp_path = result_path.with_suffix(".json.tmp")

    with tmp_path.open("w") as f:
        json.dump(result, f, indent=2)

    tmp_path.replace(result_path)


def log_pending_eval_results(project_root, wandb_run_id):
    results_dir = Path(project_root) / "eval_results"
    logged_dir = Path(project_root) / "eval_results_logged"
    results_dir.mkdir(exist_ok=True)
    logged_dir.mkdir(exist_ok=True)

    results = []
    for result_path in results_dir.glob("*.json"):
        with result_path.open("r") as f:
            result = json.load(f)
        if result["wandb_run_id"] != wandb_run_id:
            continue
        step_name = "epoch" if "epoch" in result else "images_seen"
        results.append((result[step_name], result_path, result))

    for step, result_path, result in sorted(results):
        step_name = "epoch" if "epoch" in result else "images_seen"
        wandb.log({f"eval_{step_name}": step, **result["metrics"]})
        result_path.replace(logged_dir / result_path.name)


def log_all_pending_eval_results_to_wandb(project_root):
    results_dir = Path(project_root) / "eval_results"
    results_dir.mkdir(exist_ok=True)

    wandb_runs = {}
    for result_path in results_dir.glob("*.json"):
        with result_path.open("r") as f:
            result = json.load(f)
        step_name = "epoch" if "epoch" in result else "images_seen"
        wandb_runs[result["wandb_run_id"]] = step_name

    for wandb_run_id, step_name in sorted(wandb_runs.items()):
        wandb.login()
        wandb.init(project="universal-pretraining-for-images", id=wandb_run_id, resume="must")
        eval_step_name = f"eval_{step_name}"
        wandb.define_metric(step_name, hidden=True)
        wandb.define_metric(eval_step_name, hidden=True)
        wandb.define_metric("train/*", step_metric=step_name)
        wandb.define_metric("eval_test_accuracy/*", step_metric=eval_step_name, overwrite=True)
        if step_name == "images_seen":
            wandb.define_metric("eval_validation_accuracy/*", step_metric=eval_step_name, overwrite=True)
            wandb.define_metric("eval_validation_reconstruction_loss/*", step_metric=eval_step_name, overwrite=True)
            wandb.define_metric("eval_validation_means/*", step_metric=eval_step_name, overwrite=True)
            wandb.define_metric("eval_test_reconstruction_loss/*", step_metric=eval_step_name, overwrite=True)
            wandb.define_metric("eval_test_means/*", step_metric=eval_step_name, overwrite=True)
        log_pending_eval_results(project_root, wandb_run_id)
        wandb.finish()
