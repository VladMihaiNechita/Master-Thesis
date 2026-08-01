import argparse
import os
import sys
import time
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.util import load_config, set_seed
from src.experiments.pretraining.main import get_resume_checkpoint_path, main as pretrain
from src.experiments.pretraining.evaluation import main as evaluate
from src.experiments.fine_tuning.main import main as fine_tune_main
from src.experiments.fine_tuning.evaluation import main as fine_tune_evaluate


def main():
        set_seed()
        start_time = time.time()

        modes = ["pretraining_main", "pretraining_evaluation", "fine_tuning_main", "fine_tuning_evaluation"]
        gpu_types = ["a100", "a6000", "a5000", "a4000", "a2"]

        parser = argparse.ArgumentParser()
        parser.add_argument("--mode", choices=modes, default="pretraining_main")
        parser.add_argument("--gpu_type", choices=gpu_types, default="a6000")
        parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to the checkpoint file for evaluation.")
        args = parser.parse_args()

        mode = args.mode
        gpu_type = args.gpu_type  # Unused

        if args.checkpoint_path is not None:
            checkpoint_path = PROJECT_ROOT / "checkpoints" / args.checkpoint_path
        else:
            checkpoint_path = None
        
        if mode == "pretraining_main":
            config_path = PROJECT_ROOT / "src" / "experiments" / "pretraining" / "configs" / "tiny_pretraining_main.yaml"
            config = load_config(config_path)
            pretrain(config, checkpoint_path=checkpoint_path, run_only_one_checkpoint=False)

        elif mode == "pretraining_evaluation":
            if checkpoint_path is None:
                parser.error("--checkpoint_path is required for evaluation")
        
            config_path = PROJECT_ROOT / "src" / "experiments" / "pretraining" / "configs" / "tiny_pretraining_evaluation.yaml"
            config = load_config(config_path)

            datasets_to_evaluate = config["datasets_to_evaluate"]
            metric_group = config["metric_group"]
            result_suffix = config["result_suffix"]
            delete_checkpoints_after_evaluation = config["delete_checkpoints_after_evaluation"]

            # Map underscore-normalized config keys back to the dataset names used by evaluation.
            reconstruction_batch_sizes = {
                dataset: config[f"reconstruction_batch_size_{dataset.replace('-', '_')}"]
                for dataset in datasets_to_evaluate
            }
            feature_batch_sizes = {
                dataset: config[f"feature_batch_size_{dataset.replace('-', '_')}"]
                for dataset in datasets_to_evaluate
            }
            linear_probe_batch_size = config["linear_probe_batch_size"]

            evaluate(checkpoint_path, datasets_to_evaluate,
                    reconstruction_batch_sizes, feature_batch_sizes, linear_probe_batch_size,
                    metric_group, result_suffix)
        
            if delete_checkpoints_after_evaluation:
                if not checkpoint_path.name.endswith("_0.pth") and not checkpoint_path.name.endswith("_100000000.pth"):
                    get_resume_checkpoint_path(checkpoint_path).unlink(missing_ok=True)
                    checkpoint_path.unlink()

        elif mode == "fine_tuning_main":
            config_path = PROJECT_ROOT / "src" / "experiments" / "fine_tuning" / "configs" / "tiny_fine_tuning_main.yaml"
            fine_tune_main(fine_tuning_config=load_config(config_path),
                           fine_tuning_checkpoint_path=checkpoint_path,
                           dataset="imagenet100-cmc", run_only_one_checkpoint=True)

        elif mode == "fine_tuning_evaluation":
            if checkpoint_path is None:
                parser.error("--checkpoint_path is required for evaluation")

            config_path = PROJECT_ROOT / "src" / "experiments" / "fine_tuning" / "configs" / "tiny_fine_tuning_evaluation.yaml"
            config = load_config(config_path)
            fine_tune_evaluate(checkpoint_path, config["batch_size"])

        print(f"Total time taken: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
