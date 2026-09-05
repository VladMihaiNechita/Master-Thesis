import paramiko, time
from pathlib import Path

from util import connect_to_das6, run_command_on_das6, best_node_for_evaluation, run_file


def evaluate_all_checkpoints_das6(mode="pretraining_evaluation", experiment="MAE"):
    checkpoints_directory = "/var/scratch/gkl505/Universal Pretraining for Images/checkpoints"
    current_checkpoints = set()
    prefixes = ("ijepa_model_",) if experiment == "I_JEPA" else ("model_", "official_model_")
    if mode == "fine_tuning_evaluation":
        prefixes = tuple(f"fine_tuned_{prefix}" for prefix in prefixes)

    start_time = time.time()
    time_to_check = 10 * 3600  # 10 hours
    try:
        while time.time() - start_time < time_to_check:
            jump = target = sftp = None
            try:
                jump, target = connect_to_das6()
                sftp = target.open_sftp()

                while time.time() - start_time < time_to_check:
                    checkpoints = {name for name in sftp.listdir(checkpoints_directory)
                                   if name.startswith(prefixes) and name.endswith(".pth")
                                   and not name.endswith("_resume.pth")}
                    new_checkpoints = sorted(checkpoints - current_checkpoints)
                    for checkpoint in new_checkpoints:
                        node_name, gpu_type = best_node_for_evaluation(target, ["a5000", "a4000", "a2"])
                        if node_name is not None:
                            file_arguments = [f"--experiment={experiment}", f"--mode={mode}", f"--gpu_type={gpu_type}",
                                              f"--checkpoint_path={checkpoint}"]
                            job_id = run_file(target=target, file_to_run="run.py", file_arguments=file_arguments,
                                              run_name=f"evaluation_{Path(checkpoint).stem}", node_name=node_name, time_limit="00:14:59")
                            if job_id is not None:
                                print(f"Started evaluation for checkpoint {checkpoint} on {node_name} ({gpu_type}).")
                                current_checkpoints.add(checkpoint)
                                run_command_on_das6(target, "squeue -u gkl505")
                            else:
                                print(f"Failed to start evaluation for checkpoint {checkpoint}.")
                                print("I do not know what happend !!! I will try again in 30 seconds.")
                                break
                        else:
                            print("No suitable node found for evaluation. Will try again in 30 seconds.")
                            break
                    time.sleep(30)
            except (OSError, EOFError, paramiko.SSHException) as error:
                print(f"DAS-6 connection interrupted: {error}. Reconnecting in 5 seconds.")
            finally:
                if sftp is not None:
                    sftp.close()
                if target is not None:
                    target.close()
                if jump is not None:
                    jump.close()

            time.sleep(5)
    except KeyboardInterrupt:
        print("Stopped checking DAS-6 checkpoints.")


def wait_for_node_availability(jump, target, nodes=["a6000"], check_interval_in_seconds=10):
    while True:
        try:
            if target is None:
                jump, target = connect_to_das6()

            node_name, gpu_type = best_node_for_evaluation(target, nodes)
            if node_name is not None:
                return jump, target, node_name, gpu_type

            print(f"No suitable node found for evaluation. Will check again in {check_interval_in_seconds} seconds.")
            time.sleep(check_interval_in_seconds)
        except (OSError, EOFError, paramiko.SSHException) as error:
            print(f"DAS-6 connection interrupted: {error}. Reconnecting in 5 seconds.")
            if target is not None:
                target.close()
            if jump is not None:
                jump.close()
            jump = target = None
            time.sleep(5)
