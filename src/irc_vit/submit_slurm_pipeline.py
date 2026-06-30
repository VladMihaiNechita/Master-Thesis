from __future__ import annotations

import argparse
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.irc_vit.config import (
    DEFAULT_EVAL_EMBEDDING_MODES,
    DEFAULT_EVAL_PROBES,
    DEFAULT_SLURM_CPU_CPUS,
    DEFAULT_SLURM_CPU_PARTITION,
    DEFAULT_SLURM_CPU_TIME,
    DEFAULT_SLURM_EVAL_CPUS,
    DEFAULT_SLURM_EVAL_GPUS,
    DEFAULT_SLURM_EVAL_PARTITION,
    DEFAULT_SLURM_EVAL_TIME,
    DEFAULT_SLURM_EXTRACT_CPUS,
    DEFAULT_SLURM_EXTRACT_GPUS,
    DEFAULT_SLURM_EXTRACT_PARTITION,
    DEFAULT_SLURM_EXTRACT_TIME,
    DEFAULT_SLURM_MERGE_CPUS,
    DEFAULT_SLURM_MERGE_PARTITION,
    DEFAULT_SLURM_MERGE_TIME,
    DEFAULT_SLURM_MODULES,
    DEFAULT_SLURM_TRAIN_CPUS,
    DEFAULT_SLURM_TRAIN_GPUS,
    DEFAULT_SLURM_TRAIN_PARTITION,
    DEFAULT_SLURM_TRAIN_TIME,
    load_config,
    validate_eval_embedding_modes,
    validate_eval_probes,
)
from src.irc_vit.synthetic_datasets import dataset_filter_key


@dataclass
class Job:
    name: str
    script: Path
    partition: str
    cpus: int
    time: str
    gpus: int = 0
    mem: str = ""
    account: str = ""


def _argv_without_sbatch_go(argv: list[str] | None) -> list[str] | None:
    if argv and argv[0] == "go":
        return argv[1:]
    return argv


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit a Slurm train/eval pipeline with GPU feature caches")
    parser.add_argument("--config", required=True, help="Training config")
    parser.add_argument("--eval-config", default="", help="Evaluation config; defaults to --config")
    parser.add_argument("--checkpoint", default="", help="Existing checkpoint. If omitted, submit training first.")
    parser.add_argument("--checkpoints", nargs="*", default=None, help="Checkpoint paths to evaluate after training")
    parser.add_argument("--skip-train", action="store_true", help="Do not submit the training job")
    parser.add_argument("--train-resume", default="", help="Checkpoint path used to resume the training job.")
    parser.add_argument("--train-dependency", default="", help="Existing Slurm job IDs that the training job should wait for.")
    parser.add_argument("--work-dir", default="", help="Pipeline outputs and Slurm logs")
    parser.add_argument("--features", default="", help="Feature-cache directory")
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--embedding-modes", nargs="*", default=None)
    parser.add_argument("--feature-dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--skip-reconstruction", action="store_true")
    parser.add_argument(
        "--group-eval-by-checkpoint",
        dest="group_eval_by_checkpoint",
        action="store_true",
        default=True,
        help="Run one GPU cached-feature eval job per checkpoint. This is the default.",
    )
    parser.add_argument(
        "--split-eval-by-dataset",
        dest="group_eval_by_checkpoint",
        action="store_false",
        help="Use the legacy per-dataset CPU eval fanout.",
    )
    parser.add_argument("--linear-device", default="auto", choices=["config", "auto", "cpu", "cuda"], help="Device passed to grouped cached-feature linear probes")
    parser.add_argument("--log-shards-to-wandb", action="store_true")
    parser.add_argument("--disable-wandb-merge", action="store_true")
    parser.add_argument("--submit", action="store_true", help="Actually call sbatch. Default only writes scripts.")

    parser.add_argument("--train-partition", default=DEFAULT_SLURM_TRAIN_PARTITION)
    parser.add_argument("--train-gpus", type=int, default=DEFAULT_SLURM_TRAIN_GPUS)
    parser.add_argument("--train-cpus", type=int, default=DEFAULT_SLURM_TRAIN_CPUS)
    parser.add_argument("--train-time", default=DEFAULT_SLURM_TRAIN_TIME)
    parser.add_argument("--train-mem", default="")

    parser.add_argument("--extract-partition", default=DEFAULT_SLURM_EXTRACT_PARTITION)
    parser.add_argument("--extract-gpus", type=int, default=DEFAULT_SLURM_EXTRACT_GPUS)
    parser.add_argument("--extract-cpus", type=int, default=DEFAULT_SLURM_EXTRACT_CPUS)
    parser.add_argument("--extract-time", default=DEFAULT_SLURM_EXTRACT_TIME)
    parser.add_argument("--extract-mem", default="")

    parser.add_argument("--cpu-partition", default=DEFAULT_SLURM_CPU_PARTITION)
    parser.add_argument("--cpu-cpus", type=int, default=DEFAULT_SLURM_CPU_CPUS)
    parser.add_argument("--cpu-time", default=DEFAULT_SLURM_CPU_TIME)
    parser.add_argument("--cpu-mem", default="")

    parser.add_argument("--eval-partition", default=DEFAULT_SLURM_EVAL_PARTITION)
    parser.add_argument("--eval-gpus", type=int, default=DEFAULT_SLURM_EVAL_GPUS)
    parser.add_argument("--eval-cpus", type=int, default=DEFAULT_SLURM_EVAL_CPUS)
    parser.add_argument("--eval-time", default=DEFAULT_SLURM_EVAL_TIME)
    parser.add_argument("--eval-mem", default="")

    parser.add_argument("--merge-partition", default=DEFAULT_SLURM_MERGE_PARTITION)
    parser.add_argument("--merge-cpus", type=int, default=DEFAULT_SLURM_MERGE_CPUS)
    parser.add_argument("--merge-time", default=DEFAULT_SLURM_MERGE_TIME)
    parser.add_argument("--merge-mem", default="")
    parser.add_argument("--account", default="")
    return parser.parse_args(_argv_without_sbatch_go(argv))


def _q(value: str | Path) -> str:
    return shlex.quote(str(value))


def _module_setup() -> str:
    loads = "\n".join(f"  module load {shlex.quote(module)} >/dev/null 2>&1" for module in DEFAULT_SLURM_MODULES)
    return f"""if command -v module >/dev/null 2>&1; then
  module purge >/dev/null 2>&1 || true
{loads}
fi"""


def write_job_script(job: Job, command: str) -> None:
    job.script.parent.mkdir(parents=True, exist_ok=True)
    gpu_line = f"#SBATCH --gpus={job.gpus}\n" if job.gpus > 0 else ""
    account_line = f"#SBATCH --account={job.account}\n" if job.account else ""
    mem_line = f"#SBATCH --mem={job.mem}\n" if job.mem else ""
    content = f"""#!/bin/bash
#SBATCH --job-name={job.name}
#SBATCH --partition={job.partition}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={job.cpus}
{gpu_line}#SBATCH --time={job.time}
{account_line}{mem_line}#SBATCH --output={job.script.parent / (job.name + ".out")}
#SBATCH --error={job.script.parent / (job.name + ".err")}

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

{_module_setup()}

if [ -f .wandb_env ]; then
  set -a
  source .wandb_env
  set +a
fi

if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

export PYTHONPATH="$PWD:${{PYTHONPATH:-}}"
export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-1}}"
export MKL_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-1}}"
export OPENBLAS_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-1}}"

{command}
"""
    job.script.write_text(content, encoding="utf-8")


def submit_or_print(job: Job, dependency: str, submit: bool) -> str:
    cmd = ["sbatch", "--parsable"]
    if dependency:
        cmd.append(f"--dependency=afterok:{dependency}")
    cmd.append(str(job.script))
    printable = " ".join(shlex.quote(part) for part in cmd)
    if not submit:
        print(printable)
        return f"<{job.name}>"
    result = subprocess.run(cmd, text=True, capture_output=True, check=True)
    job_id = result.stdout.strip().split(";")[0]
    print(f"{job.name}: {job_id}")
    return job_id


def python_module(module: str, args: list[str]) -> str:
    return "python -m " + shlex.quote(module) + " " + " ".join(_q(arg) for arg in args)


def cache_dataset_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name).lower()).strip("_") or "dataset"


def checkpoint_tag(checkpoint: Path) -> str:
    if str(checkpoint).lower() in {"random_init", "none", "no_pretraining"}:
        return "random_init"
    return cache_dataset_name(checkpoint.stem)


def dataset_names(config: dict[str, Any], selected: list[str] | None) -> list[str]:
    names = [str(raw.get("name", "")) for raw in config.get("datasets", [])]
    names = [name for name in names if name]
    if not selected:
        return names
    selected_norm = {dataset_filter_key(name) for name in selected}
    return [name for name in names if dataset_filter_key(name) in selected_norm]


def probes(config: dict[str, Any]) -> list[str]:
    return validate_eval_probes(config.get("eval", {}).get("probes", DEFAULT_EVAL_PROBES))


def eval_device_arg(requested: str, gpus: int) -> str:
    if requested in {"config", "cpu", "cuda"}:
        return requested
    return "cuda" if gpus > 0 else "cpu"


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    train_cfg = load_config(args.config)
    eval_config = args.eval_config or args.config
    eval_cfg = load_config(eval_config)
    validate_eval_embedding_modes(eval_cfg.get("eval", {}).get("embedding_modes", DEFAULT_EVAL_EMBEDDING_MODES))
    if args.embedding_modes is not None:
        args.embedding_modes = validate_eval_embedding_modes(args.embedding_modes)
    run_id = str(train_cfg.get("run_id", Path(args.config).stem))
    train_output = Path(str(train_cfg.get("output_dir", f"results/{run_id}")))
    work_dir = Path(args.work_dir or train_output / "pipeline")
    slurm_dir = work_dir / "slurm"
    feature_base = Path(args.features or work_dir / "features")
    checkpoints = [Path(path) for path in args.checkpoints] if args.checkpoints else [Path(args.checkpoint) if args.checkpoint else train_output / "checkpoint_final.pt"]
    eval_datasets = dataset_names(eval_cfg, args.datasets)
    eval_probes = probes(eval_cfg)
    merge_inputs: list[Path] = []
    eval_jobs: list[tuple[Job, str]] = []
    online_merge_jobs: list[tuple[Job, str]] = []
    extract_job_ids: list[str] = []

    train_job_id = ""
    should_train = not args.skip_train and not (args.checkpoint and not args.checkpoints)
    if should_train:
        train_job = Job(
            name=f"{run_id}-train",
            script=slurm_dir / "01_train.sbatch",
            partition=args.train_partition,
            gpus=args.train_gpus,
            cpus=args.train_cpus,
            time=args.train_time,
            mem=args.train_mem,
            account=args.account,
        )
        train_args = ["--config", args.config]
        if args.train_resume:
            train_args.extend(["--resume", args.train_resume])
        write_job_script(train_job, python_module("src.irc_vit.train", train_args))
        train_job_id = submit_or_print(train_job, args.train_dependency, args.submit)

    for checkpoint in checkpoints:
        ckpt_tag = checkpoint_tag(checkpoint)
        feature_root = feature_base if len(checkpoints) == 1 else feature_base / ckpt_tag
        reconstruction_csv = work_dir / f"eval_reconstruction_{ckpt_tag}.csv"
        extract_args = [
            "--config", eval_config,
            "--checkpoint", str(checkpoint),
            "--out", str(feature_root),
            "--dtype", args.feature_dtype,
            "--overwrite",
        ]
        if args.datasets:
            extract_args.extend(["--datasets", *args.datasets])
        if args.embedding_modes:
            extract_args.extend(["--embedding-modes", *args.embedding_modes])
        if args.skip_reconstruction or not bool(eval_cfg.get("eval", {}).get("reconstruction_mae", False)):
            extract_args.append("--disable-reconstruction")
        else:
            extract_args.extend(["--reconstruction-out", str(reconstruction_csv)])
            merge_inputs.append(reconstruction_csv)

        extract_job = Job(
            name=f"{run_id}-features-{ckpt_tag}",
            script=slurm_dir / f"02_extract_features_{ckpt_tag}.sbatch",
            partition=args.extract_partition,
            gpus=args.extract_gpus,
            cpus=args.extract_cpus,
            time=args.extract_time,
            mem=args.extract_mem,
            account=args.account,
        )
        write_job_script(extract_job, python_module("src.irc_vit.extract_feature_cache", extract_args))
        extract_job_id = submit_or_print(extract_job, train_job_id, args.submit)
        extract_job_ids.append(extract_job_id)

        if args.group_eval_by_checkpoint:
            checkpoint_merge_inputs: list[Path] = []
            if reconstruction_csv in merge_inputs:
                checkpoint_merge_inputs.append(reconstruction_csv)
            checkpoint_merge_dependency = extract_job_id
            if eval_probes:
                out_csv = work_dir / f"eval_features_{ckpt_tag}.csv"
                merge_inputs.append(out_csv)
                checkpoint_merge_inputs.append(out_csv)
                linear_eval_device = eval_device_arg(args.linear_device, args.eval_gpus)
                eval_args = [
                    "--config", eval_config,
                    "--checkpoint", str(checkpoint),
                    "--features", str(feature_root),
                    "--out", str(out_csv),
                    "--linear-device", linear_eval_device,
                    "--overwrite",
                ]
                if args.datasets:
                    eval_args.extend(["--datasets", *args.datasets])
                if args.embedding_modes:
                    eval_args.extend(["--embedding-modes", *args.embedding_modes])
                if not args.log_shards_to_wandb:
                    eval_args.append("--disable-wandb")
                job = Job(
                    name=f"{run_id}-eval-{ckpt_tag}",
                    script=slurm_dir / f"03_eval_{ckpt_tag}.sbatch",
                    partition=args.eval_partition,
                    gpus=args.eval_gpus,
                    cpus=args.eval_cpus,
                    time=args.eval_time,
                    mem=args.eval_mem,
                    account=args.account,
                )
                write_job_script(job, python_module("src.irc_vit.evaluate_feature_cache", eval_args))
                eval_jobs.append((job, extract_job_id))
                checkpoint_merge_dependency = f"eval:{job.name}"
            if checkpoint_merge_inputs:
                checkpoint_merge_args = [
                    "--config", eval_config,
                    "--inputs", *[str(path) for path in checkpoint_merge_inputs],
                    "--out", str(work_dir / f"eval_merged_{ckpt_tag}.csv"),
                ]
                if args.disable_wandb_merge or args.log_shards_to_wandb:
                    checkpoint_merge_args.append("--disable-wandb")
                checkpoint_merge_job = Job(
                    name=f"{run_id}-merge-{ckpt_tag}",
                    script=slurm_dir / f"04_merge_{ckpt_tag}.sbatch",
                    partition=args.merge_partition,
                    gpus=0,
                    cpus=args.merge_cpus,
                    time=args.merge_time,
                    mem=args.merge_mem,
                    account=args.account,
                )
                write_job_script(
                    checkpoint_merge_job,
                    python_module("src.irc_vit.log_eval_csv_to_wandb", checkpoint_merge_args),
                )
                online_merge_jobs.append((checkpoint_merge_job, checkpoint_merge_dependency))
            continue

        for dataset in eval_datasets:
            dataset_key = cache_dataset_name(dataset).replace("-", "_")
            if eval_probes:
                out_csv = work_dir / f"eval_cpu_{ckpt_tag}_{dataset_key}.csv"
                merge_inputs.append(out_csv)
                eval_args = [
                    "--config", eval_config,
                    "--checkpoint", str(checkpoint),
                    "--features", str(feature_root),
                    "--out", str(out_csv),
                    "--datasets", dataset,
                    "--probes", *eval_probes,
                    "--linear-device", "cpu",
                    "--overwrite",
                ]
                if args.embedding_modes:
                    eval_args.extend(["--embedding-modes", *args.embedding_modes])
                if not args.log_shards_to_wandb:
                    eval_args.append("--disable-wandb")
                job = Job(
                    name=f"{run_id}-cpu-{ckpt_tag}-{dataset_key}",
                    script=slurm_dir / f"03_cpu_{ckpt_tag}_{dataset_key}.sbatch",
                    partition=args.cpu_partition,
                    gpus=0,
                    cpus=args.cpu_cpus,
                    time=args.cpu_time,
                    mem=args.cpu_mem,
                    account=args.account,
                )
                write_job_script(job, python_module("src.irc_vit.evaluate_feature_cache", eval_args))
                eval_jobs.append((job, extract_job_id))

    eval_job_ids: list[str] = []
    eval_job_ids_by_name: dict[str, str] = {}
    for job, dep in eval_jobs:
        job_id = submit_or_print(job, dep, args.submit)
        eval_job_ids.append(job_id)
        eval_job_ids_by_name[job.name] = job_id

    online_merge_job_ids: list[str] = []
    previous_online_merge_job_id = ""
    for job, dep in online_merge_jobs:
        if dep.startswith("eval:"):
            dep = eval_job_ids_by_name[dep.removeprefix("eval:")]
        dependencies = [dep]
        if previous_online_merge_job_id:
            dependencies.append(previous_online_merge_job_id)
        job_id = submit_or_print(job, ":".join(dependencies), args.submit)
        online_merge_job_ids.append(job_id)
        previous_online_merge_job_id = job_id

    if not merge_inputs:
        print(f"wrote pipeline scripts under {slurm_dir}")
        print(f"feature cache: {feature_base}")
        print("no eval CSV shards requested; skipping merge job")
        return

    merge_args = [
        "--config", eval_config,
        "--inputs", *[str(path) for path in merge_inputs],
        "--out", str(work_dir / "eval_merged.csv"),
    ]
    if args.disable_wandb_merge or args.log_shards_to_wandb or online_merge_jobs:
        merge_args.append("--disable-wandb")
    merge_job = Job(
        name=f"{run_id}-merge-eval",
        script=slurm_dir / "05_merge_eval.sbatch",
        partition=args.merge_partition,
        gpus=0,
        cpus=args.merge_cpus,
        time=args.merge_time,
        mem=args.merge_mem,
        account=args.account,
    )
    write_job_script(merge_job, python_module("src.irc_vit.log_eval_csv_to_wandb", merge_args))
    final_merge_dependencies = eval_job_ids or extract_job_ids
    submit_or_print(merge_job, ":".join(final_merge_dependencies), args.submit)

    print(f"wrote pipeline scripts under {slurm_dir}")
    print(f"feature cache: {feature_base}")
    print(f"merged eval CSV: {work_dir / 'eval_merged.csv'}")


if __name__ == "__main__":
    main()
