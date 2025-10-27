#!/usr/bin/env python3
"""Runs the calibration benchmark with a baseline model from the command line.

usage:
    - general:
        run_acs_benchmark_baseline.py
            [-h]
            --model MODEL
            --results-dir RESULTS_DIR
            --data-dir DATA_DIR
            [--task TASK]
            [--subsampling SUBSAMPLING]
            [--seed SEED]
            [--clf-params [CLF_PARAMS ...]]
            [--use-feature-subset USE_FEATURE_SUBSET]
            [--use-population-filter USE_POPULATION_FILTER]
            [--logger-level {DEBUG,INFO,WARNING,ERROR,CRITICAL}]
    - example: python baseline/run_benchmark_baseline.py
                --model GBM
                --results-dir '../results/baselines/'
                --data-dir '../data/'
                --clf-params "learning_rate=0.2 max_depth=5 max_iter=200"

"""
import json
import logging
import sys
from argparse import ArgumentParser
from pathlib import Path

from folktexts._utils import ParseDict

from mono_multi.baseline import BASELINES
from mono_multi.setup import ACS_TASKS, TABLESHIFT_TASKS, SIPP_TASKS

TASKS = ACS_TASKS + TABLESHIFT_TASKS + SIPP_TASKS

DEFAULT_SEED = 42


def is_float(element: any) -> bool:
    # If you expect None to be passed:
    if element is None:
        return False
    try:
        float(element)
        return True
    except ValueError:
        return False


def is_int(element: any) -> bool:
    # If you expect None to be passed:
    if element is None:
        return False
    try:
        int(element)
        return True
    except ValueError:
        return False


def setup_arg_parser() -> ArgumentParser:

    # Init parser
    parser = ArgumentParser(
        description="Benchmark risk scores produced by a language model on ACS data."
    )

    # Define a custom argument type for a list of strings
    def list_of_strings(arg):
        return arg.split(",")

    # List of command-line arguments, with type and helper string
    cli_args = [
        (
            "--model",
            str,
            f"[str] Baseline model name, one of {BASELINES}",
            True,
        ),
        (
            "--results-dir",
            str,
            "[str] Directory under which this experiment's results will be saved",
        ),
        ("--data-dir", str, "[str] Root folder to find datasets on"),
        (
            "--task",
            str,
            f"[str] Name of the task to run the experiment on, one of {TASKS}",
            True,
        ),
        (
            "--subsampling",
            float,
            "[float] Which fraction of the dataset to use (if omitted will use all data)",
            False,
        ),
        (
            "--seed",
            int,
            "[int] Random seed -- to set for reproducibility",
            False,
            DEFAULT_SEED,
        ),
    ]

    for arg in cli_args:
        parser.add_argument(
            arg[0],
            type=arg[1],
            help=arg[2],
            required=(arg[3] if len(arg) > 3 else True),  # NOTE: required by default
            default=(arg[4] if len(arg) > 4 else None),  # default value if provided
        )

    # Add special arguments (e.g., boolean flags or multiple-choice args)
    parser.add_argument(
        "--clf-params",
        # type=list_of_strings,
        help="[str] Optional hyperparameters for the baseline model",
        nargs="*",
        action=ParseDict,
        required=False,
        default={},
    )

    # Optionally, receive a list of features to use (subset of original list)
    parser.add_argument(
        "--use-feature-subset",
        type=list_of_strings,
        help="[str] Optional subset of features to use for prediction, comma separated",
        required=False,
    )

    parser.add_argument(
        "--use-population-filter",
        type=list_of_strings,
        help=(
            "[str] Optional population filter for this benchmark; must follow "
            "the format 'column_name=value' to filter the dataset by a specific value."
        ),
        required=False,
    )

    parser.add_argument(
        "--logger-level",
        type=str,
        help="[str] The logging level to use for the experiment",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        required=False,
        default="WARNING",
    )

    return parser


def main():
    """Prepare and launch the LLM-as-classifier experiment using ACS data."""

    # Setup parser and process cmd-line args
    parser = setup_arg_parser()
    args = parser.parse_args()

    logging.getLogger().setLevel(level=args.logger_level)
    pretty_args_str = json.dumps(vars(args), indent=4, sort_keys=True)
    logging.info(f"Current python executable: '{sys.executable}'")
    logging.info(f"Received the following cmd-line args: {pretty_args_str}")

    task = args.task
    model = args.model
    assert task in TASKS, f"Unknown task name: {task}, must be one of {TASKS}"
    assert (
        model in BASELINES
    ), f"Unknown model name: {model}, must be one of {BASELINES}"

    # Parse population filter if provided
    population_filter_dict = None
    if args.use_population_filter:
        from folktexts.cli._utils import cmd_line_args_to_kwargs

        population_filter_dict = cmd_line_args_to_kwargs(args.use_population_filter)

    # Load model
    print("Model parameters passed: ", args.clf_params)
    # model = BASELINES[args.model](**args.clf_params)

    # Fill ACS Benchmark config
    from monoculture.baseline.benchmark import BenchmarkBaselineConfig

    config = BenchmarkBaselineConfig(
        feature_subset=args.use_feature_subset or None,
        population_filter=population_filter_dict,
        seed=args.seed,
    )

    # Create ACS Benchmark object
    from monoculture.baseline.benchmark import BenchmarkBaseline

    if task in ACS_TASKS:
        bench = BenchmarkBaseline.make_acs_benchmark(
            task_name=task,
            model=model,
            clf_params=args.clf_params,
            # using auto-tokenizer, TODO: check if baseline
            data_dir=args.data_dir,
            config=config,
            subsampling=args.subsampling,
        )
    elif task in TABLESHIFT_TASKS:
        bench = BenchmarkBaseline.make_tableshift_benchmark(
            task_name=task,
            model=model,
            clf_params=args.clf_params,
            # using auto-tokenizer, TODO: check if baseline
            data_dir=args.data_dir,
            config=config,
            subsampling=args.subsampling,
        )
    elif task in SIPP_TASKS:
        bench = BenchmarkBaseline.make_sipp_benchmark(
            task_name=task,
            model=model,
            clf_params=args.clf_params,
            # using auto-tokenizer, TODO: check if baseline
            data_dir=args.data_dir,
            config=config,
            subsampling=args.subsampling,
        )
    else:
        raise ValueError(f"Task {task} not implemented.")

    # Set-up results directory
    from folktexts.cli._utils import get_or_create_results_dir

    results_dir = get_or_create_results_dir(
        model_name=Path(args.model).name,
        task_name=args.task,
        results_root_dir=args.results_dir,
    )
    logging.info(f"Saving results to {results_dir.as_posix()}")

    # Run benchmark
    bench.run(results_root_dir=results_dir)
    bench.save_results()

    # Save results
    import pprint

    pprint.pprint(bench.results, indent=4, sort_dicts=True)

    # Finish
    from folktexts._utils import get_current_timestamp

    print(f"\nFinished experiment successfully at {get_current_timestamp()}\n")


if __name__ == "__main__":
    main()
