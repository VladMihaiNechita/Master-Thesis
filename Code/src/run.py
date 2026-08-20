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
from src.experiments.MAE.pretraining.main import (
    get_resume_checkpoint_path as get_mae_resume_checkpoint_path,
    main as mae_pretrain,
)
from src.experiments.MAE.pretraining.evaluation import main as mae_evaluate
from src.experiments.I_JEPA.pretraining.main import (
    get_resume_checkpoint_path as get_ijepa_resume_checkpoint_path,
    main as ijepa_pretrain,
)
from src.experiments.I_JEPA.pretraining.evaluation import main as ijepa_evaluate
from src.experiments.MAE.fine_tuning.main import main as mae_fine_tune
from src.experiments.MAE.fine_tuning.evaluation import main as mae_fine_tune_evaluate
from src.experiments.I_JEPA.fine_tuning.main import main as ijepa_fine_tune
from src.experiments.I_JEPA.fine_tuning.evaluation import main as ijepa_fine_tune_evaluate


size = "tiny"  # tiny, small, base, large
run_only_one_checkpoint = False
fine_tuning_dataset = "imagenet1k-simclr-10pct" # imagenet100-cmc, imagenet1k-simclr-10pct

def main():
        set_seed()
        start_time = time.time()

        modes = ["pretraining_main", "pretraining_evaluation", "fine_tuning_main", "fine_tuning_evaluation"]
        gpu_types = ["a100", "a6000", "a5000", "a4000", "a2"]

        parser = argparse.ArgumentParser()
        parser.add_argument("--experiment", choices=["MAE", "I_JEPA"], default="MAE")
        parser.add_argument("--mode", choices=modes, default="pretraining_main")
        parser.add_argument("--gpu_type", choices=gpu_types, default="a6000")
        parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to the checkpoint file for evaluation.")
        parser.add_argument("--fine_tuning_dataset", default=fine_tuning_dataset)
        parser.add_argument("--fine_tuning_layer_decay", type=float, default=None)
        parser.add_argument("--pretraining_checkpoint_path", type=str, default=None)
        args = parser.parse_args()

        mode = args.mode
        experiment = args.experiment
        gpu_type = args.gpu_type  # Unused

        if experiment == "MAE":
            pretrain = mae_pretrain
            evaluate = mae_evaluate
            get_resume_checkpoint_path = get_mae_resume_checkpoint_path
            fine_tune = mae_fine_tune
            fine_tune_evaluate = mae_fine_tune_evaluate
        else:
            pretrain = ijepa_pretrain
            evaluate = ijepa_evaluate
            get_resume_checkpoint_path = get_ijepa_resume_checkpoint_path
            fine_tune = ijepa_fine_tune
            fine_tune_evaluate = ijepa_fine_tune_evaluate

        if args.checkpoint_path is not None:
            checkpoint_path = PROJECT_ROOT / "checkpoints" / args.checkpoint_path
        else:
            checkpoint_path = None
        
        if mode == "pretraining_main":
            config_path = PROJECT_ROOT / "src" / "experiments" / experiment / "pretraining" / "configs" / f"{size}_pretraining_main.yaml"
            config = load_config(config_path)
            pretrain(config, checkpoint_path=checkpoint_path, run_only_one_checkpoint=run_only_one_checkpoint)

        elif mode == "pretraining_evaluation":
            if checkpoint_path is None:
                parser.error("--checkpoint_path is required for evaluation")
        
            config_path = PROJECT_ROOT / "src" / "experiments" / experiment / "pretraining" / "configs" / f"{size}_pretraining_evaluation.yaml"
            config = load_config(config_path)

            datasets_to_evaluate = config["datasets_to_evaluate"]
            metric_group = config["metric_group"]
            result_suffix = config["result_suffix"]
            delete_checkpoints_after_evaluation = config["delete_checkpoints_after_evaluation"]

            # Map underscore-normalized config keys back to the dataset names used by evaluation.
            feature_batch_sizes = {
                dataset: config[f"feature_batch_size_{dataset.replace('-', '_')}"]
                for dataset in datasets_to_evaluate
            }
            linear_probe_batch_size = config["linear_probe_batch_size"]

            if experiment == "MAE":
                reconstruction_batch_sizes = {
                    dataset: config[f"reconstruction_batch_size_{dataset.replace('-', '_')}"]
                    for dataset in datasets_to_evaluate
                }
                evaluate(checkpoint_path, datasets_to_evaluate,
                        reconstruction_batch_sizes, feature_batch_sizes, linear_probe_batch_size,
                        metric_group, result_suffix)
            else:
                evaluate(checkpoint_path, datasets_to_evaluate,
                        feature_batch_sizes, linear_probe_batch_size,
                        metric_group, result_suffix)
        
            if delete_checkpoints_after_evaluation:
                if not checkpoint_path.name.endswith(("_0.pth", "_100000000.pth", "_100050000.pth")):
                    get_resume_checkpoint_path(checkpoint_path).unlink(missing_ok=True)
                    checkpoint_path.unlink()

        elif mode == "fine_tuning_main":
            config_path = PROJECT_ROOT / "src" / "experiments" / experiment / "fine_tuning" / "configs" / f"{size}_fine_tuning_main.yaml"
            fine_tuning_config = load_config(config_path)
            if args.fine_tuning_layer_decay is not None:
                fine_tuning_config["fine_tuning_layer_decay"] = args.fine_tuning_layer_decay
            if args.pretraining_checkpoint_path is not None:
                fine_tuning_config["pretraining_checkpoint_path"] = args.pretraining_checkpoint_path

            fine_tune(fine_tuning_config=fine_tuning_config,
                      fine_tuning_checkpoint_path=checkpoint_path,
                      dataset=args.fine_tuning_dataset, run_only_one_checkpoint=run_only_one_checkpoint)

        elif mode == "fine_tuning_evaluation":
            if checkpoint_path is None:
                parser.error("--checkpoint_path is required for evaluation")

            config_path = PROJECT_ROOT / "src" / "experiments" / experiment / "fine_tuning" / "configs" / f"{size}_fine_tuning_evaluation.yaml"
            config = load_config(config_path)
            fine_tune_evaluate(checkpoint_path, config["batch_size"])

        print(f"Total time taken: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
