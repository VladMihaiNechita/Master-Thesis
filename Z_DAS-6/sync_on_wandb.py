from pathlib import Path
import json
import os
import shlex
import sys
from tempfile import TemporaryDirectory


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if "WANDB_API_KEY" not in os.environ:
    os.environ["WANDB_API_KEY"] = (Path(__file__).resolve().parent / "0_wandb_api_key.txt").read_text().strip()
sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.evaluation_util import log_all_pending_eval_results_to_wandb
from util import connect_to_das6, run_command_on_das6


DAS6_PROJECT_ROOT = "/var/scratch/gkl505/Universal Pretraining for Images"


def main():
    jump, target = connect_to_das6()
    remote_results_dir = f"{DAS6_PROJECT_ROOT}/eval_results"
    remote_logged_dir = f"{DAS6_PROJECT_ROOT}/eval_results_logged"
    run_command_on_das6(target, f"mkdir -p {shlex.quote(remote_results_dir)} {shlex.quote(remote_logged_dir)}")

    sftp = target.open_sftp()
    pending_results = []
    for result_name in sftp.listdir(remote_results_dir):
        if result_name.endswith(".json"):
            with sftp.open(f"{remote_results_dir}/{result_name}") as result_file:
                result = json.load(result_file)
            # Completed evaluations may intentionally retain their checkpoint.
            if result.get("complete", False) or "epoch" in result:
                pending_results.append(result_name)
                continue
            if "complete" in result:
                continue
            # run.py deletes the checkpoint only after every dataset finishes successfully.
            try:
                sftp.stat(f"{DAS6_PROJECT_ROOT}/checkpoints/{result['checkpoint']}")
            except FileNotFoundError:
                pending_results.append(result_name)

    if pending_results:
        # W&B runs locally; only the small result files are temporarily downloaded.
        with TemporaryDirectory() as temporary_directory:
            temporary_project_root = Path(temporary_directory)
            temporary_results_dir = temporary_project_root / "eval_results"
            temporary_results_dir.mkdir()

            for result_name in pending_results:
                sftp.get(f"{remote_results_dir}/{result_name}",
                         str(temporary_results_dir / result_name))

            log_all_pending_eval_results_to_wandb(temporary_project_root)

            for result_name in pending_results:
                sftp.posix_rename(f"{remote_results_dir}/{result_name}",
                                  f"{remote_logged_dir}/{result_name}")

    sftp.close()
    target.close()
    jump.close()


if __name__ == "__main__":
    main()
