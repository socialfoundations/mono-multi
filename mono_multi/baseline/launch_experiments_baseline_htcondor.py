#!/usr/bin/env python3
"""Launch htcondor jobs for all ACS benchmark experiments."""
import argparse
from pathlib import Path
from pprint import pprint

from folktexts._io import load_json, save_json
from folktexts.cli._utils import get_or_create_results_dir

from folktexts.cli.experiments import Experiment, launch_experiment_job
import logging

from mono_multi.setup import ACS_TASKS, TABLESHIFT_TASKS, SIPP_TASKS
from mono_multi.baseline import BASELINES

TASKS = ACS_TASKS + TABLESHIFT_TASKS + SIPP_TASKS

################
# Useful paths #
################
ROOT_DIR = Path("/fast/groups/sf")
# ROOT_DIR = Path("~").expanduser().resolve()               # on local machine

# data directory
ACS_DATA_DIR = ROOT_DIR / "data"
TABLESHIFT_DATA_DIR = Path("/fast/mgorecki/monoculture/data")


##################
# Global configs #
##################
JOB_CPUS = 4
JOB_GPUS = 1
JOB_MEMORY_GB = 60
JOB_BID = 500


# Function that defines common settings among all LLM-as-clf experiments
def make_base_clf_experiment(
    executable_path: str,
    model_name: str,
    task: str,
    results_dir: str,
    env_vars_str: str = "",
    **kwargs,
) -> Experiment:
    """Create an experiment object to run."""

    # Split experiment and job kwargs
    job_kwargs = {key: val for key, val in kwargs.items() if key.startswith("job_")}
    experiment_kwargs = {
        key: val for key, val in kwargs.items() if key not in job_kwargs
    }

    # Set default job kwargs
    job_kwargs.setdefault("job_cpus", JOB_CPUS)
    job_kwargs.setdefault("job_gpus", JOB_GPUS)
    job_kwargs.setdefault("job_memory_gb", JOB_MEMORY_GB)
    job_kwargs.setdefault("job_gpu_memory_gb", 35)
    job_kwargs.setdefault("job_bid", JOB_BID)

    experiment_kwargs.setdefault(
        "data_dir",
        (
            ACS_DATA_DIR.as_posix()
            if task in ACS_TASKS
            else TABLESHIFT_DATA_DIR.as_posix()
        ),
    )

    results_dir = get_or_create_results_dir(
        model_name=model_name,
        task_name=task,
        results_root_dir=results_dir,
    )

    # Define experiment
    exp = Experiment(
        executable_path=executable_path,
        env_vars=env_vars_str,
        kwargs=dict(
            model=model_name,
            task=task,
            results_dir=results_dir.as_posix(),
            **experiment_kwargs,
        ),
        **job_kwargs,
    )

    save_json(
        obj=exp.to_dict(),
        path=Path(results_dir) / f"experiment.{exp.hash()}.json",
        overwrite=True,
    )
    logging.info(f"Created experiment.{exp.hash()}.json at {results_dir.as_posix()}")
    print(f"Saving experiment json to {results_dir.as_posix()}")

    return exp


def setup_arg_parser() -> argparse.ArgumentParser:
    # Init parser
    parser = argparse.ArgumentParser(
        description="Launch experiments to evaluate baseline classifiers."
    )

    parser.add_argument(
        "--executable-path",
        type=str,
        help="[string] Path to the executable script to run.",
        required=True,
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        help="[string] Directory under which results will be saved.",
        required=True,
    )

    parser.add_argument(
        "--model",
        type=str,
        help="[string] Model name (sklearn classifier) - can provide multiple!",
        required=False,
        action="append",
    )

    parser.add_argument(
        "--task",
        type=str,
        help="[string] ACS task name to run experiments on - can provide multiple!",
        required=False,
        action="append",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Construct folder structure and print experiments without launching them.",
        default=False,
    )

    parser.add_argument(
        "--experiment-json",
        type=str,
        help="[string] Path to an experiment JSON file to load. Will override all other args.",
        required=False,
    )

    parser.add_argument(
        "--environment",
        type=str,
        help=(
            "[string] String defining environment variables to be passed to "
            "launched jobs, in the form 'VAR1=val1;VAR2=val2;...'."
        ),
        required=False,
    )

    return parser


def main():
    # Parse command-line arguments
    parser = setup_arg_parser()
    args, extra_kwargs = parser.parse_known_args()

    # Parse extra kwargs
    from folktexts.cli._utils import cmd_line_args_to_kwargs

    extra_kwargs = cmd_line_args_to_kwargs(extra_kwargs)

    # Prepare command-line arguments
    models = args.model or BASELINES
    tasks = args.task or TASKS
    executable_path = Path(args.executable_path).resolve()
    if not executable_path.exists() or not executable_path.is_file():
        raise FileNotFoundError(f"Executable script not found at '{executable_path}'.")

    # Set-up results directory
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)

    # Load experiment from JSON file if provided
    if args.experiment_json:
        print(f"Launching job for experiment at '{args.experiment_json}'...")
        exp = Experiment(**load_json(args.experiment_json))
        all_experiments = [exp]

    # Otherwise, run all experiments planned
    else:
        all_experiments = [
            make_base_clf_experiment(
                executable_path=executable_path.as_posix(),
                model_name=model,
                task=task,
                results_dir=args.results_dir,
                env_vars_str=args.environment,
                **extra_kwargs,
            )
            for model in models
            for task in tasks
        ]

    # Log total number of experiments
    print(f"Launching {len(all_experiments)} experiment(s)...\n")
    for i, exp in enumerate(all_experiments):
        cluster_id = launch_experiment_job(exp).cluster() if not args.dry_run else None
        print(f"{i:2}. cluster-id={cluster_id}")
        pprint(exp.to_dict(), indent=4)


if __name__ == "__main__":
    main()
