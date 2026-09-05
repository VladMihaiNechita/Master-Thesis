from pathlib import Path

from util import connect_to_das6, run_command_on_das6, copy_a_directory_from_local_to_das6, run_file, get_max_time
from small_scripts import wait_for_node_availability, evaluate_all_checkpoints_das6


def main():
    jump, target = connect_to_das6()

    copy_a_directory_from_local_to_das6(target=target,
                                        das6_directory_path="/var/scratch/gkl505/Universal Pretraining for Images/src",
                                        local_directory_path=Path(__file__).resolve().parents[1] / "src")

    run_command_on_das6(target, "sinfo -o '%40N  %20G  %20T  %P'")

    file_arguments=["--experiment=MAE", "--mode=pretraining_main", "--gpu_type=a6000"]
    jump, target, node_name, gpu_type = wait_for_node_availability(jump, target, nodes=["a6000"])
    run_file(target=target, file_to_run="run.py", file_arguments=file_arguments, run_name="pretraining_mae_tiny_irc_new", node_name=node_name, time_limit=get_max_time())
    run_command_on_das6(target, "squeue -u gkl505")

    #jump, target, node_name, gpu_type = wait_for_node_availability(jump, target, nodes=["a100", "a6000"])
    #print(f"Node {node_name} with GPU type {gpu_type} is available for fine-tuning.")
    #for i in range(5, 100, 5):
    #    file_arguments=["--mode=fine_tuning_main", "--gpu_type=a6000", f"--checkpoint_path=fine_tuned_official_model_tiny_generator_irc_images_seen_100000000_imagenet1k-simclr-10pct_layer_decay_1.0_epoch_{i}.pth"]
    #    run_file(target=target, file_to_run="run.py", file_arguments=file_arguments, run_name=f"fine_tuning_100M_1.0_1k", node_name=node_name, time_limit=get_max_time())
    #run_command_on_das6(target, "squeue -u gkl505")

    target.close()
    jump.close()

    #evaluate_all_checkpoints_das6(mode="fine_tuning_evaluation")


if __name__ == "__main__":
    main()
