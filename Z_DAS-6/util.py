import json, paramiko, shlex
from datetime import datetime, time, timedelta
from pathlib import Path


a100_nodes = ["node003"]
a6000_nodes = ["node001", "node002", "node004", "node030", "node032", "node028", "node029"]
a5000_nodes = ["node024", "node027", "node014", "node015", "node021"]
a4000_nodes = ["node005", "node006", "node008", "node007", "node009", "node011", "node012", "node013", "node022", "node023", "node025", "node026", "node033", "node034"]
a2_nodes = ["node020", "node016", "node017", "node018", "node019"]
cpu_only_nodes = ["node031"]

node_groups = [("a100", a100_nodes), ("a6000", a6000_nodes),
               ("a5000", a5000_nodes), ("a4000", a4000_nodes),
               ("a2", a2_nodes), ("cpu_only", cpu_only_nodes)]


def get_max_time():
    # Get the maximum time limit for jobs on DAS-6 in the format "HH:MM:SS"
    now = datetime.now()
    if now.weekday() < 5 and time(8) <= now.time() < time(19, 45):
        return "00:14:59"

    deadline = now.replace(hour=8, minute=0, second=0, microsecond=0)
    if deadline <= now:
        deadline += timedelta(days=1)
    while deadline.weekday() >= 5:
        deadline += timedelta(days=1)

    # An idle node can take up to 10 seconds to start the submitted job.
    total_seconds = max(0, int((deadline - now).total_seconds()) - 10)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def connect_to_das6():
    credentials = json.loads(Path(__file__).with_name("0_credentials.json").read_text())
    # This is the public VU SSH gateway that you can reach from your laptop.
    jump_host = "ssh.data.vu.nl"
    jump_username = "gkl505"
    jump_host_password = credentials["jump_host_password"]

    # This is the DAS-6 filesystem/login host that is reachable through the gateway.
    target_host = "fs0.das6.cs.vu.nl"
    target_username = "gkl505"
    target_password = credentials["target_password"]

    # SSH client for the gateway connection from your laptop into the VU network.
    jump = paramiko.SSHClient()
    jump.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    jump.connect(jump_host, username=jump_username, password=jump_host_password, auth_timeout=60)
    jump_transport = jump.get_transport()

    channel = jump_transport.open_channel(kind="direct-tcpip", dest_addr=(target_host, 22), src_addr=("", 0))

    # SSH client for the actual DAS-6 login, using the gateway tunnel below.
    target = paramiko.SSHClient()
    target.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    target.connect(target_host, username=target_username, password=target_password, sock=channel, auth_timeout=60)

    print("Connected to DAS-6 through the VU SSH gateway.")
    return jump, target


def run_command_on_das6(target, command):
    stdin, stdout, stderr = target.exec_command(command)
    output = stdout.read().decode()
    error = stderr.read().decode()
    if output:
        print(output)
    if error:
        print(error)


def is_node_idle(target, node_name):
    # The node must be idle and not already requested by a queued job.
    command = f"sinfo --noheader --nodes={shlex.quote(node_name)} --format=%T"
    stdin, stdout, stderr = target.exec_command(command)
    node_state = stdout.read().decode().strip()
    error = stderr.read().decode().strip()
    if error:
        print(error)
    if error or node_state != "idle":
        return False

    stdin, stdout, stderr = target.exec_command("squeue --noheader --format=%n")
    requested_nodes = stdout.read().decode().split()
    error = stderr.read().decode().strip()
    if error:
        print(error)
    return not error and node_name not in requested_nodes


def best_node_for_evaluation(target, node_types = ["a5000", "a4000", "a2"]):
    # Check the state of each node and return the first idle node found.
    for node_type, nodes in node_groups:
        if node_type in node_types:
            for node in nodes:
                if is_node_idle(target, node):
                    return node, node_type
    return None, None


def set_up(target):
    # Prepare the Python source files on scratch storage.
    run_command_on_das6(target, "mkdir -p /var/scratch/$USER/src")
    run_command_on_das6(target, "cd /var/scratch/$USER/src && wget https://www.python.org/ftp/python/3.13.13/Python-3.13.13.tgz")
    run_command_on_das6(target, "cd /var/scratch/$USER/src && tar -xzf Python-3.13.13.tgz")

    # Configure, compile, and install Python under your home directory.
    run_command_on_das6(target, "cd /var/scratch/$USER/src/Python-3.13.13 && ./configure --prefix=$HOME/.local/python-3.13.13 --with-ensurepip=install")
    run_command_on_das6(target, "cd /var/scratch/$USER/src/Python-3.13.13 && make -j4")
    run_command_on_das6(target, "cd /var/scratch/$USER/src/Python-3.13.13 && make install")

    # Create the virtual environment.
    run_command_on_das6(target, "$HOME/.local/python-3.13.13/bin/python3.13 -m venv /var/scratch/$USER/venv-python-3.13.13")

    # Install PyTorch and other libraries into the virtual environment.
    run_command_on_das6(target, ". /var/scratch/$USER/venv-python-3.13.13/bin/activate && pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126")
    run_command_on_das6(target, ". /var/scratch/$USER/venv-python-3.13.13/bin/activate && pip install pyyaml")
    run_command_on_das6(target, ". /var/scratch/$USER/venv-python-3.13.13/bin/activate && pip install matplotlib")
    run_command_on_das6(target, ". /var/scratch/$USER/venv-python-3.13.13/bin/activate && pip install wandb")
    run_command_on_das6(target, ". /var/scratch/$USER/venv-python-3.13.13/bin/activate && pip install opencv-python")

    print("Setup complete!")


def copy_a_directory_from_local_to_das6(target, das6_directory_path, local_directory_path):
    local_directory = Path(local_directory_path)
    if not local_directory.is_dir():
        raise FileNotFoundError(f"Local directory does not exist: {local_directory}")

    # Recreate the destination so files deleted locally do not remain on DAS-6.
    quoted_directory = shlex.quote(das6_directory_path)
    run_command_on_das6(target, f"rm -rf -- {quoted_directory}")
    run_command_on_das6(target, f"mkdir -p -- {quoted_directory}")

    created_remote_directories = {das6_directory_path}
    sftp = target.open_sftp()

    # Recursively visit everything inside the local directory
    for local_path in local_directory.rglob("*"):
        # Construct the correspoinding path on DAS-6
        relative_path = local_path.relative_to(local_directory).as_posix()
        das6_path = f"{das6_directory_path}/{relative_path}"

        # If the current item is a directory, create it on DAS-6 if it doesn't exist
        if local_path.is_dir():
            if das6_path not in created_remote_directories:
                run_command_on_das6(target, f'mkdir -p "{das6_path}"')
                created_remote_directories.add(das6_path)

        # If the current item is a file, copy it to DAS-6
        else:
            sftp.put(str(local_path), das6_path)

    sftp.close()
    print(f"Copied {local_directory_path} to {das6_directory_path} on DAS-6.")


def run_on_das6_node(target, node_name, time_limit, output_file_path, error_file_path,
                     das6_file_path, file_arguments):
    if node_name in ["node028", "node029"]:
        partition = "fatq"
    else:
        partition = "defq"

    if node_name == "node031":
        gres_arg = ""
        gpu = False
    else:
        gres_arg = "--gres=gpu:1 "
        gpu = True


    job_script_path = "/var/scratch/gkl505/Universal Pretraining for Images/run_job.sh"
    quoted_file_arguments = " ".join(shlex.quote(argument) for argument in file_arguments)

    submit_command = (
        f"sbatch --parsable " # --parsable: return only a machine-readable job ID, such as 1234567.
        f"--partition={shlex.quote(partition)} "
        f"--nodelist={shlex.quote(node_name)} "
        f"{gres_arg}"
        f"--time={shlex.quote(time_limit)} "
        f"--output={shlex.quote(output_file_path)} "
        f"--error={shlex.quote(error_file_path)} "
        f"{shlex.quote(job_script_path)} {shlex.quote(das6_file_path)} {shlex.quote(str(gpu))} "
        f"{quoted_file_arguments}"
    )

    stdin, stdout, stderr = target.exec_command(submit_command)
    job_id = stdout.read().decode().strip()
    error = stderr.read().decode().strip()

    if error:
        print(error)
        return None
    else:
        print(f"Job submitted to {node_name} with time limit {time_limit}. Job id: {job_id}\n\n\n")
        return job_id


def run_file(target, file_to_run, file_arguments, run_name, node_name, time_limit):
    return run_on_das6_node(target, node_name, time_limit,
                            output_file_path=f"/var/scratch/gkl505/Universal Pretraining for Images/{run_name}_output.log",
                            error_file_path=f"/var/scratch/gkl505/Universal Pretraining for Images/{run_name}_error.log",
                            das6_file_path=f"/var/scratch/gkl505/Universal Pretraining for Images/src/{file_to_run}",
                            file_arguments=file_arguments)
