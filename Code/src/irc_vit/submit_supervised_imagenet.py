from __future__ import annotations

import argparse
from pathlib import Path

from src.irc_vit.config import (
    DEFAULT_SLURM_TRAIN_CPUS,
    DEFAULT_SLURM_TRAIN_GPUS,
    DEFAULT_SLURM_TRAIN_PARTITION,
    DEFAULT_SLURM_TRAIN_TIME,
    load_config,
)
from src.irc_vit.submit_slurm_pipeline import Job, python_module, submit_or_print, write_job_script


def _argv_without_sbatch_go(argv: list[str] | None) -> list[str] | None:
    if argv and argv[0] == "go":
        return argv[1:]
    return argv


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit supervised ImageNet ViT training")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--work-dir", default="", help="Directory for Slurm scripts/logs; defaults to output_dir/supervised_slurm")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--partition", default=DEFAULT_SLURM_TRAIN_PARTITION)
    parser.add_argument("--gpus", type=int, default=DEFAULT_SLURM_TRAIN_GPUS)
    parser.add_argument("--cpus", type=int, default=DEFAULT_SLURM_TRAIN_CPUS)
    parser.add_argument("--time", default=DEFAULT_SLURM_TRAIN_TIME)
    parser.add_argument("--mem", default="")
    parser.add_argument("--account", default="")
    return parser.parse_args(_argv_without_sbatch_go(argv))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.gpus != 1:
        raise ValueError("src.irc_vit.supervised_imagenet is single-process; request exactly one GPU.")
    config = load_config(args.config)
    run_id = str(config.get("run_id", Path(args.config).stem))
    output_dir = Path(str(config.get("output_dir", f"results/runs/{run_id}")))
    work_dir = Path(args.work_dir or output_dir / "supervised_slurm")
    job = Job(
        name=f"{run_id}-sup",
        script=work_dir / "01_supervised_imagenet.sbatch",
        partition=args.partition,
        gpus=args.gpus,
        cpus=args.cpus,
        time=args.time,
        mem=args.mem,
        account=args.account,
    )
    module_args = ["--config", args.config]
    if args.resume:
        module_args.extend(["--resume", args.resume])
    write_job_script(job, python_module("src.irc_vit.supervised_imagenet", module_args))
    job_id = submit_or_print(job, "", args.submit)
    print(f"wrote supervised script: {job.script}")
    print(f"job: {job_id}")


if __name__ == "__main__":
    main()
