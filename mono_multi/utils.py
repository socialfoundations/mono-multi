import os
import re
from .setup import (
    LLM_MODELS,
    ACS_TASKS,
    TABLESHIFT_TASKS,
    SIPP_TASKS,
    RESULTS_CSV_SAME_PROMPT,
    RESULTS_CSV_VARY_PROMPT,
    model_families,
    variations,
    variations_defaults,
    map_feature_order_to_short,
    map_example_order_to_short,
)
import folktexts
from folktexts._io import load_json
from folktexts.llm_utils import get_model_size_B
from folktexts.acs import ACSTaskMetadata, ACSDataset
from folktexts.ts import TableshiftBRFSSTaskMetadata, TableshiftBRFSSDataset
from folktexts.sipp import SIPPTaskMetadata, SIPPDataset
from .setup import developer_map
import matplotlib.ticker as mticker

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from functools import partial
from typing import Union, Iterable, Optional, Tuple, List
from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------
# Data Loading
# ---------------------

_RESULT_COLUMNS = [
    "task",
    "model",
    "is_inst",
    "threshold_fitted",
    "threshold",
    "threshold_obj",
    "accuracy",
    "balanced_accuracy",
    "bench_hash",
    "num_shots",
    "prompt_format",
    "prompt_connector",
    "prompt_granularity",
    "prompt_feature_order",
    "prompt_example_order",
    "prompt_example_composition",
    "eval_results_path",
    "predictions_path",
    "correct_order_bias",
]


def create_result_df(
    root_dir: str | Path,
    subfolders: List[str] | str,
    tasks: List[str] | str = ACS_TASKS,
    save_path: str | Path | None = None,
    save_every: int = 500,  # intermediate save frequency
    # add_baselines: list = [],
) -> pd.DataFrame:
    """
    Creates a pandas DataFrame with relevant metadata and paths from result files.

    Parameters
    ----------
    root_dir : str or Path
        Root directory where results are gathered from.
    tasks : list, optional
        List of tasks to include in the DataFrame (default is ACS_TASKS).
    save_path : str or Path, optional
        If provided, the DataFrame will be saved to this path.

    Returns
    -------
    pd.DataFrame
        DataFrame with the following columns:
            - task: str, task name
            - model: str, model name
            - is_inst: int, whether the model is instruction-finetuned
            - bench_hash: str, hash of the benchmarking result
            - num_shots: int, number of shots used for few-shot prompting
            - prompt_format: str, format used for prompts if specified (if None: bullet)
            - prompt_connector: str, connector used for prompts if specified (if None: is)
            - prompt_granularity: str, low/original granularity used for the prompt if specified (if None: orginal)
            - prompt_feature_order: str/list, order in which features are presented in prompt (if None: orginal order)
            - eval_results_path: str, path to the evaluation results file
            - predictions_path: str, path to the predictions file
    """

    # Normalize args
    if isinstance(subfolders, str):
        subfolders = [subfolders]
    if isinstance(tasks, str):
        tasks = [tasks]
    if not isinstance(subfolders, (list, tuple)):
        raise ValueError("subfolders must be list, tuple, or str")
    if not isinstance(tasks, (list, tuple)):
        raise ValueError("tasks must be list, tuple, or str")

    root_dir = Path(root_dir)
    if save_path and isinstance(save_path, str):
        save_path = Path(save_path)
        if save_path.exists():
            raise FileExistsError(
                f"{save_path} already exists. Set a different path or delete the existing file."
            )
    # file name pattern
    pattern_json = r"^results.bench-(?P<hash>\d+)(?:-acc)?[.]json$"

    results_all = []
    processed_count = 0
    for task, folder in product(tasks, subfolders):
        logging.debug(f"Processing task={task}, folder={folder}")
        # find results files
        bench_files = find_files(root_dir / folder, pattern_json, dir_pattern=task)
        for file_path in bench_files:
            file_path = Path(file_path)
            logging.debug(f"File: {file_path}")

            # Extract model name
            model_name = file_path.parent.parent.name.replace("model-", "").replace(
                f"_task-{task}", ""
            )

            if task == "ACSIncome" and model_name.endswith("PovertyRatio"):
                # name of tasks overlaps
                continue
            if key_to_model(model_name) not in LLM_MODELS:
                logging.warning(f"Skipping unknown model: {model_name}")
                continue

            # Parse Results
            bench_hash = file_path.parent.name.split("_bench-")[1]
            parsed = parse_results_dict(load_json(file_path))

            # Build row
            row = _build_result_row(task, model_name, bench_hash, file_path, parsed)
            results_all.append(row)

            # Fitting overwrites the non-fitted results if t=0.5 (hash unchanged), save duplicate
            if row["threshold_fitted"] and parsed.get("threshold") == 0.5:
                tmp = row.copy()
                tmp["threshold_fitted"] = 0
                results_all.append(tmp)

            # Save intermediate
            processed_count += 1
            if save_path and save_every and processed_count % save_every == 0:
                pd.DataFrame(results_all, columns=_RESULT_COLUMNS).to_csv(
                    save_path, index=False
                )
                logging.info(
                    f"Intermediate save after {processed_count} files → {save_path}"
                )

    # Final Dataframe
    df = pd.DataFrame(
        results_all,
        columns=_RESULT_COLUMNS,
    )
    df = df[~((df["prompt_example_order"].astype(bool)) & (df["num_shots"] == 0))]
    logging.info(f"Final DataFrame shape: {df.shape}")
    # Validate Variations
    _validate_variations(df)

    if save_path:
        print(f"Saving dataframe to {save_path}")
        df.to_csv(save_path, index=False)
    return df


def create_result_df_parallelized(
    root_dir: str | Path,
    subfolders: List[str] | str,
    tasks: List[str] | str = ACS_TASKS,
    save_path: str | Path | None = None,
    save_every: int = 10,  # intermediate save frequency
    max_workers: int = 8,
    batch_size: int = 50,  # number of files per worker batch
    # add_baselines: list = [],
) -> pd.DataFrame:
    """
    Creates a pandas DataFrame with relevant metadata and paths from result files.

    Parameters
    ----------
    root_dir : str or Path
        Root directory where results are gathered from.
    tasks : list, optional
        List of tasks to include in the DataFrame (default is ACS_TASKS).
    save_path : str or Path, optional
        If provided, the DataFrame will be saved to this path.

    Returns
    -------
    pd.DataFrame
        DataFrame with the following columns:
            - task: str, task name
            - model: str, model name
            - is_inst: int, whether the model is instruction-finetuned
            - bench_hash: str, hash of the benchmarking result
            - num_shots: int, number of shots used for few-shot prompting
            - prompt_format: str, format used for prompts if specified (if None: bullet)
            - prompt_connector: str, connector used for prompts if specified (if None: is)
            - prompt_granularity: str, low/original granularity used for the prompt if specified (if None: orginal)
            - prompt_feature_order: str/list, order in which features are presented in prompt (if None: orginal order)
            - eval_results_path: str, path to the evaluation results file
            - predictions_path: str, path to the predictions file
    """

    # Normalize args
    if isinstance(subfolders, str):
        subfolders = [subfolders]
    if isinstance(tasks, str):
        tasks = [tasks]
    if not isinstance(subfolders, (list, tuple)):
        raise ValueError("subfolders must be list, tuple, or str")
    if not isinstance(tasks, (list, tuple)):
        raise ValueError("tasks must be list, tuple, or str")

    root_dir = Path(root_dir)
    if save_path and isinstance(save_path, str):
        save_path = Path(save_path)
        if save_path.exists():
            raise FileExistsError(
                f"{save_path} already exists. Set a different path or delete the existing file."
            )
    # file name pattern
    pattern_json = r"^results.bench-(?P<hash>\d+)(?:-acc)?[.]json$"

    results_all = []
    processed_batches_count = 0

    for task, folder in product(tasks, subfolders):
        bench_files = find_files(root_dir / folder, pattern_json, dir_pattern=task)
        # Split files into batches
        file_batches = _batch_generator(bench_files, batch_size)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_process_file_batch, task, batch)
                for batch in file_batches
            ]

            for future in as_completed(futures):
                rows = future.result()
                results_all.extend(rows)
                processed_batches_count += 1

                # Save intermediate
                if (
                    save_path
                    and save_every
                    and processed_batches_count % save_every == 0
                ):
                    pd.DataFrame(results_all, columns=_RESULT_COLUMNS).to_csv(
                        save_path, index=False
                    )
                    logging.info(
                        f"Intermediate save after {processed_batches_count} batches → {save_path}"
                    )

    # Final Dataframe
    df = pd.DataFrame(
        results_all,
        columns=_RESULT_COLUMNS,
    )
    logging.info(f"Final DataFrame shape: {df.shape}")
    # Validate Variations
    _validate_variations(df)

    if save_path:
        print(f"Saving dataframe to {save_path}")
        df.to_csv(save_path, index=False)
    return df


def _process_file(task: str, file_path: Path):
    file_path = Path(file_path)
    model_name = file_path.parent.parent.name.replace("model-", "").replace(
        f"_task-{task}", ""
    )

    # filter
    if task == "ACSIncome" and model_name.endswith("PovertyRatio"):
        return None
    if key_to_model(model_name) not in LLM_MODELS:
        logging.warning(f"Skipping unknown model: {model_name}")
        return None

    parsed = parse_results_dict(load_json(file_path))
    bench_hash = file_path.parent.name.split("_bench-")[1]

    row = _build_result_row(task, model_name, bench_hash, file_path, parsed)

    # Duplicate unfitted threshold if overwritten
    if row["threshold_fitted"] and parsed.get("threshold") == 0.5:
        tmp = row.copy()
        tmp["threshold_fitted"] = 0
        return [row, tmp]
    return [row]


def _batch_generator(iterable, batch_size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _process_file_batch(task, file_batch):
    batch_results = []
    for file_path in file_batch:
        file_path = Path(file_path)
        model_name = file_path.parent.parent.name.replace("model-", "").replace(
            f"_task-{task}", ""
        )

        # Filtering
        if task == "ACSIncome" and model_name.endswith("PovertyRatio"):
            continue
        if key_to_model(model_name) not in LLM_MODELS:
            logging.warning(f"Skipping unknown model: {model_name}")
            continue

        bench_hash = file_path.parent.name.split("_bench-")[1]
        parsed = parse_results_dict(load_json(file_path))

        row = _build_result_row(task, model_name, bench_hash, file_path, parsed)

        # Duplicate unfitted threshold if overwritten
        if row["threshold_fitted"] and parsed.get("threshold") == 0.5:
            tmp = row.copy()
            tmp["threshold_fitted"] = 0
            batch_results.extend([row, tmp])
        else:
            batch_results.append(row)
    return batch_results


_model_col = "config_model_name"
_feature_subset_col = "config_feature_subset"
_population_subset_col = "config_population_filter"

_uses_all_features_col = "uses_all_features"
_uses_all_samples_col = "uses_all_samples"


def parse_results_dict(dct) -> dict:
    """Parses results dict; brings all information to the top-level."""
    dct = dct.copy()
    dct.pop("plots", None)
    config = dct.pop("config", {})
    for key, val in config.items():
        if isinstance(config[key], dict):
            style_specs = config[key]  # config.pop(key, {})
            for subkey, subval in style_specs.items():
                dct[f"config_{key}_{subkey}"] = subval
        else:
            dct[f"config_{key}"] = val

    # Parse model name
    dct[_model_col] = parse_model_name(dct[_model_col])
    dct[_uses_all_features_col] = dct[_feature_subset_col] is None
    if dct[_feature_subset_col] is None:
        dct[_feature_subset_col] = "full"

    dct[_uses_all_samples_col] = dct[_population_subset_col] is None

    dct["base_name"] = get_base_name(dct[_model_col])
    dct["is_inst"] = dct["base_name"] != dct[_model_col]

    for key, val in dct.items():
        if isinstance(val, dict):
            print(key, val)
    assert not any(isinstance(val, dict) for val in dct.values()), dct
    return dct


def _build_result_row(
    task: str,
    model_name: str,
    bench_hash: str,
    file_path: Path,
    parsed: dict,
):
    # save relevant metadata and path in df
    num_shots = parsed.get("config_few_shot") or 0
    prompt_config = _parse_prompt_config(parsed, num_shots)
    metric_data = _parse_metrics(file_path, parsed)
    correct_order_bias = int(bool(parsed.get("config_correct_order_bias")))
    return {
        "task": task,
        "model": model_name,
        "is_inst": int(is_instruction_tuned(model_name)),
        **metric_data,
        "bench_hash": bench_hash,
        "num_shots": num_shots,
        **prompt_config,
        "eval_results_path": file_path.as_posix(),
        "predictions_path": parsed["predictions_path"][
            parsed["predictions_path"].find("results/") :
        ],
        "correct_order_bias": correct_order_bias,
    }


def _parse_prompt_config(parsed: dict, num_shots: int) -> dict:
    def get_first(keys, default):
        return next((parsed[k] for k in keys if k in parsed), default)

    # "Prompting style is backward compatible for older results json files"
    prompt_format = get_first(
        [
            "config_prompt_variation_format",
            "config_prompt_style_format",
            "config_prompt_style",
        ],
        "bullet",
    )
    prompt_connector = get_first(
        [
            "config_prompt_variation_connector",
            "config_prompt_style_connector",
            "config_prompt_connector",
        ],
        "is",
    )
    prompt_granularity = parsed.get("config_prompt_variation_granularity", "original")

    prompt_feature_order = map_feature_order_to_short.get(
        parsed.get("config_prompt_variation_order"), "default"
    )

    prompt_example_order = None
    prompt_example_composition = None
    if num_shots > 0:
        prompt_example_order = map_example_order_to_short.get(
            parsed.get("config_prompt_variation_example_order"), "default"
        )
        comp = parsed.get("config_compose_few_shot_examples")
        prompt_example_composition = (
            ",".join(map(str, comp)) if isinstance(comp, list) else comp
        )
    if num_shots == 0:
        example_order = parsed.get("config_prompt_variation_example_order")
        if example_order:
            prompt_example_order = map_example_order_to_short.get(example_order, None)
        comp = parsed.get("config_compose_few_shot_examples")
        if comp:
            prompt_example_composition = (
                ",".join(map(str, comp)) if isinstance(comp, list) else comp
            )

    return {
        "prompt_format": prompt_format,
        "prompt_connector": prompt_connector,
        "prompt_granularity": prompt_granularity,
        "prompt_feature_order": prompt_feature_order,
        "prompt_example_order": prompt_example_order,
        "prompt_example_composition": prompt_example_composition,
    }


def _parse_metrics(file_path: str | Path, parsed: dict):
    """Extract metrics from parsed json result file"""
    if isinstance(file_path, str):
        file_path = Path(file_path)
    threshold = get_metric(file_path, metric="threshold")
    threshold_fitted = int(bool(parsed.get("threshold_fitted_on")))
    threshold_obj = parsed.get("threshold_obj") if threshold_fitted else None
    accuracy = get_metric(file_path, metric="accuracy")
    balanced_accuracy = get_metric(file_path, metric="balanced_accuracy")
    return {
        "threshold_fitted": threshold_fitted,
        "threshold": threshold,
        "threshold_obj": threshold_obj,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
    }


def _validate_variations(df: pd.DataFrame):
    """Ensure variations fall into expected sets."""
    checks = {
        "prompt_format": "format",
        "prompt_connector": "connector",
        "prompt_granularity": "granularity",
        "prompt_feature_order": "feature_order",
    }

    for col, var_key in checks.items():
        unique_vals = set(df[col].unique())
        allowed_vals = set(variations[var_key])
        if not unique_vals.issubset(allowed_vals):
            raise ValueError(
                f"{col} has unexpected values: {unique_vals - allowed_vals}"
            )


def find_files(root_folder, pattern, dir_pattern=""):
    # Compile the regular expression pattern
    regex = re.compile(pattern)

    # Walk through the directory tree
    for dirpath, dirnames, filenames in os.walk(root_folder):
        if dir_pattern in dirpath:
            for filename in filenames:
                if regex.match(filename):
                    # If the filename matches the pattern, add it to the list
                    yield os.path.join(dirpath, filename)


def load_risk_scores(csv_path: str | Path) -> pd.DataFrame:
    """
    Loads risk scores from a csv file, removes the 'label' column, and renames
    the 'risk_score' column to the model name.

    Args:
        csv_path (str | Path): The file path to the csv containing model predictions.

    Returns:
        pd.DataFrame: A DataFrame with risk scores, where the 'risk_score' column
                      is renamed to the model name.
    """
    # load risk scores, change column to <model_name>
    risk_score = (
        pd.read_csv(csv_path, index_col=0)
        .drop("label", axis=1)
        .rename(
            columns={
                "risk_score": re.search(r"model-(.+?)_task-", csv_path)
                .group(1)
                .split("/")[1]
            }
        )
    )
    return risk_score


def get_predictions(
    csv_path: str | Path,
    eval_json_path: str | Path = None,
) -> pd.DataFrame:
    """
    Loads risk scores and binarize usiing binarization threshold from
    evaluation results. Returns a DataFrame.

    Args:
        csv_path (str | Path): The file path to the CSV containing risk scores.
        eval_json_path (str | Path): If stored elsewhere, file path to the json containing evals including the threshold.

    Returns:
        pd.DataFrame: A DataFrame with binarized predictions.
    """
    risk_scores = load_risk_scores(csv_path)
    # binarize using threshold from respective eval results
    if not eval_json_path:
        bench_hash = Path(csv_path).parent.as_posix().split("bench-")[1]
        threshold = load_json(
            Path(csv_path).parent / f"results.bench-{bench_hash}.json"
        ).get("threshold")
    else:
        threshold = load_json(Path(eval_json_path)).get("threshold")
    if threshold is None:
        logging.warning(f"Threshold not found for {csv_path}, defaulting to 0.5")
        threshold = 0.5
    return risk_scores.map(lambda x: int(x >= threshold))


def load_task_data(tasks: str | list, data_dir: Path | str, split: str = "test"):
    if isinstance(tasks, str):
        tasks = [tasks]
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)
    data = {}
    for task in tasks:
        print(task)
        logging.info(f"Loading data for task {task}.")
        if task in ACS_TASKS or task.startswith("ACS"):
            acs_task = ACSTaskMetadata.get_task(task)
            acs_dataset_configs = (
                folktexts.benchmark.Benchmark.ACS_DATASET_CONFIGS.copy()
            )
            dataset = ACSDataset.make_from_task(
                task=acs_task, cache_dir=data_dir, **acs_dataset_configs
            )
        elif task in TABLESHIFT_TASKS or task.startswith("BRFSS"):
            brfss_task = TableshiftBRFSSTaskMetadata.get_task(task)
            dataset_configs = folktexts.benchmark.Benchmark.DATASET_CONFIGS.copy()
            dataset = TableshiftBRFSSDataset.make_from_task(
                task=brfss_task, cache_dir=data_dir, **dataset_configs
            )
        elif task in SIPP_TASKS:
            sipp_task = SIPPTaskMetadata.get_task(task)
            dataset = SIPPDataset.make_from_task(
                task=sipp_task, cache_dir=data_dir, **dataset_configs
            )
            dataset_configs = folktexts.benchmark.Benchmark.DATASET_CONFIGS.copy()
            dataset = SIPPDataset.make_from_task(
                task=sipp_task, cache_dir=data_dir, **dataset_configs
            )
        else:
            raise KeyError(f"{task} not in available tasks.")
        X_test, y_test = dataset.get_data_split(split)
        data[task] = (X_test, y_test)
    return data


def load_data_if_needed(data, tasks, data_dir: Path = Path("./data")):
    if data is None:
        return load_task_data(tasks, data_dir=data_dir)
    elif not all([t in data.keys() for t in tasks]):
        tasks_missing = [t for t in tasks if t not in data.keys()]
        for t in tasks_missing:
            print(f"add {t} to data")
            data_t = load_task_data(t, Path("./data"))
            data.update(data_t)
    return data


def load_model_outputs_same_prompt(
    df: pd.DataFrame,
    tasks: list[str] = ACS_TASKS + TABLESHIFT_TASKS,
    return_risk_scores: bool = True,
):
    mask_same_prompt = (
        (df["prompt_format"] == "bullet")
        & (df["prompt_connector"] == "is")
        & (df["prompt_granularity"] == "original")
        & (df["prompt_feature_order"] == "default")
    )
    outputs = {}
    for task in tasks:
        print(task)
        task_df = df[mask_same_prompt & (df["task"] == task)]
        # get available models
        models = task_df["model"].unique().tolist()
        models.sort(key=lambda m: get_size(m) + int(is_instruction_tuned(m)))
        outputs_per_task = []
        for m in models:
            data_m = task_df[task_df["model"] == m]
            if data_m.shape[0] != 1:
                logging.warning(
                    f"Expected 1 row for model {m}, but found {data_m.shape[0]}. Skipping this model.\n {data_m}"
                )
                continue
            if return_risk_scores:
                outputs_per_task.append(
                    load_risk_scores(data_m.iloc[0]["predictions_path"])
                )
            else:
                outputs_per_task.append(
                    get_predictions(
                        csv_path=data_m.iloc[0]["predictions_path"],
                        eval_json_path=data_m.iloc[0]["eval_results_path"],
                    )
                )
        logging.debug(task, len(outputs_per_task))
        if len(outputs_per_task) > 0:
            outputs[task] = pd.concat(outputs_per_task, axis=1)
    return outputs


def load_results_overview(
    num_shots: int | List[int] = 0,
    threshold_fitted: int = True,
    same_prompt: bool = True,
):
    assert (
        RESULTS_CSV_SAME_PROMPT.is_file()
        if same_prompt
        else RESULTS_CSV_VARY_PROMPT.is_file()
    ), "Results file does not seem to exist."
    num_shots = num_shots if isinstance(num_shots, list) else [num_shots]
    df = pd.read_csv(
        RESULTS_CSV_SAME_PROMPT if same_prompt else RESULTS_CSV_VARY_PROMPT
    )
    mask = (df["num_shots"].isin(num_shots)) & (
        df["threshold_fitted"] == int(threshold_fitted)
    )
    if same_prompt:
        # restrict to default prompting style
        mask_same_prompt = (
            (df["prompt_format"] == "bullet")
            & (df["prompt_connector"] == "is")
            & (df["prompt_granularity"] == "original")
            & (df["prompt_feature_order"] == "default")
        )
        mask &= mask_same_prompt
    return df[mask]


def add_evals_to_df(
    df,
    metrics=[
        # "n_samples",
        # "accuracy",
        "fpr",
        "fnr",
        # "ppr",
        # "num_pred_negatives",
        # "num_pred_positives",
        # "balanced_accuracy",
    ],
):
    for metric in metrics:
        df[metric] = (
            df["eval_results_path"].apply(partial(get_metric, metric=metric)).copy()
        )
    return df


def get_metric(json_path: str | Path, metric: str):
    """
    Retrieves the specified metric from a benchmark results file.

    Args:
        json_path (str | Path): Path to the JSON file containing evaluation results.
        metric (str, optional): The name of the metric to retrieve.

    Returns:
        The value of the specified metric from the evaluation results.
    """
    evals = load_json(json_path)
    return evals[metric]


def get_metrics(json_path: str | Path, metrics: list[str]):
    """
    Retrieves multiple metrics from a benchmark results file.

    Args:
        json_path (str | Path): Path to the JSON file containing evaluation results.
        metrics (list[str]): A list of metric names to retrieve.

    Returns:
        A dictionary with the metric names as keys and their respective values as values.
    """
    evals = load_json(json_path)
    return {metric: evals[metric] for metric in metrics}


# ---------------------
# Filter predictions
# ---------------------


def filter_predictions_and_data(
    predictions,
    data: Tuple[pd.DataFrame, pd.Series] = None,
    task_df: pd.DataFrame = None,
    feature: Optional[str] = None,
    feature_val: Optional[Union[str, float, int, list]] = None,
    label_val: Optional[Union[str, float, int]] = None,
    restrict_models: Optional[Iterable[str]] = None,
    restrict_to_better_const: bool = False,
    restrict_to_top_eps: bool = False,
    restrict_to_topk: bool = False,
    topk: int = 10,
    eps: float = 0.05,
    selection_criterion: str = "balanced_accuracy",
) -> Tuple[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
    filtered_data = data
    # ---- Filter Rows ----
    if data is not None:
        predictions, filtered_data = filter_rows(
            predictions=predictions,
            data=data,
            feature=feature,
            feature_val=feature_val,
            label_val=label_val,
        )

    # ---- Filter Columns ----
    predictions = filter_cols(
        predictions,
        data=data,
        task_df=task_df,
        restrict_models=restrict_models,
        restrict_to_better_const=restrict_to_better_const,
        restrict_to_top_eps=restrict_to_top_eps,
        restrict_to_topk=restrict_to_topk,
        topk=topk,
        eps=eps,
        selection_criterion=selection_criterion,
    )

    return predictions, filtered_data


def filter_rows(
    predictions,
    data,
    feature: Optional[str] = None,
    feature_val: Optional[Union[str, float, int, list]] = None,
    label_val: Optional[Union[str, float, int]] = None,
):
    X, y = data
    # reduce y if predictions have fewer rows
    if predictions.shape[0] < y.shape[0]:
        print("Reduce labels to those with corresponding predictions")
        y = y.loc[predictions.index]

    filter_idx = X.index
    if feature is not None and feature_val is not None:
        if not isinstance(feature_val, list):
            feature_val = [feature_val]
        filter_idx = filter_idx.intersection(X[X[feature].isin(feature_val)].index)

    if label_val is not None:
        if not isinstance(label_val, list):
            label_val = [label_val]
        filter_idx = filter_idx.intersection(y[y.isin(label_val)].index)

    predictions = predictions.loc[filter_idx]
    filtered_data = (data[0].loc[filter_idx], data[1].loc[filter_idx])

    return predictions, filtered_data


def filter_cols(
    predictions,
    data=None,
    task_df=None,
    restrict_models: Optional[Iterable[str]] = None,
    restrict_to_better_const: bool = False,
    restrict_to_top_eps: bool = False,
    restrict_to_topk: bool = False,
    topk: int = 10,
    eps: float = 0.05,
    selection_criterion: str = "balanced_accuracy",
):
    if restrict_models is not None:
        valid_models = [m for m in restrict_models if m in predictions.columns]
        if not valid_models:
            raise ValueError(
                "None of the specified restrict_models exist in the predictions columns."
            )

        predictions = predictions.filter(items=valid_models)
    if task_df is not None:
        if restrict_to_better_const:
            # use unfiltered (original) data to restrict to better const
            if data is None:
                raise ValueError(
                    "`data` (X, y) must be provided to filter models better constant"
                )
            models = get_models_better_const(
                df=task_df, y_true=data[1], acc=selection_criterion
            )
            predictions = predictions.filter(items=models)
        if restrict_to_topk:
            models = get_top_models_by_k(df=task_df, k=topk, acc=selection_criterion)
            predictions = predictions.filter(items=models)
        if restrict_to_top_eps:
            models = get_top_models_by_eps(df=task_df, eps=eps, acc=selection_criterion)
            predictions = predictions.filter(items=models)

    return predictions


def filter_by_label(predictions, data, label_val):
    return filter_predictions_and_data(
        predictions=predictions, data=data, label_val=label_val
    )


def filter_by_feature_val(predictions, data, feature, feature_val):
    assert (
        feature is not None and feature_val is not None
    ), "Provide a feature and the corresponding value"
    return filter_predictions_and_data(
        predictions=predictions, data=data, feature=feature, feature_val=feature_val
    )


def filter_results_all_tasks(
    tasks: Union[list, str],
    predictions,
    df,
    data=None,
    restrict_to_better_const=True,
    restrict_to_top_eps=True,
    restrict_to_topk=False,
    topk=10,
    eps=0.05,
    restrict_to_positive_label=True,
    restrict_to_negative_label=False,
    acc="balanced_accuracy",
):
    if isinstance(tasks, str):
        tasks = [tasks]
    assert not (
        restrict_to_positive_label & restrict_to_negative_label
    ), "Cannot restrict to both positive and negative instances."

    for task in tasks:
        label_val = (
            int(restrict_to_positive_label)
            if restrict_to_positive_label or restrict_to_negative_label
            else None
        )
        print(task)
        print(f"- before: {predictions[task].shape}")
        predictions[task], data[task] = filter_predictions_and_data(
            predictions=predictions[task],
            data=data[task] if data is not None else None,
            task_df=df[df["task"] == task],
            restrict_to_better_const=restrict_to_better_const,
            restrict_to_top_eps=restrict_to_top_eps,
            restrict_to_topk=restrict_to_topk,
            topk=topk,
            eps=eps,
            label_val=label_val,
            selection_criterion=acc,
        )
        print(f"- after: {predictions[task].shape}")

    return predictions, data


def filter_df_by_default_cond(
    df, exclude_key=None, default_cond: dict = variations_defaults
):
    # select all rows, where all but exclude_key are set to default
    indices = set.intersection(
        *[
            set(df[df["prompt_" + key] == val].index)
            for key, val in default_cond.items()
            if key != exclude_key
        ]
    )
    return df.loc[list(indices)]


def get_top_models_by_eps(df, eps=0.05, acc="balanced_accuracy"):
    # sort by acc
    df_sorted_by_acc = df.sort_values(by=acc, ascending=False)
    # top acc
    top_acc = df_sorted_by_acc.iloc[0][acc].item()
    # filter models
    models = df[df[acc] >= top_acc - eps]["model"].to_list()
    return sorted(models, key=get_size_and_it)


def get_top_models_by_k(df, k=10, acc="balanced_accuracy"):
    # sort by acc
    df_sorted_by_acc = df.sort_values(by=acc, ascending=False)
    # get top k models
    models = df_sorted_by_acc.iloc[:k]["model"].to_list()
    return sorted(models, key=get_size_and_it)


def get_models_better_const(df, y_true, acc="balanced_accuracy"):
    print(f"Using {acc} for comparison.")
    if acc == "accuracy":
        const_acc = max((y_true == 1).sum(), (y_true == 0).sum()) / y_true.shape[0]
    else:
        const_acc = 0.5
    models = df["model"][df[acc] > const_acc].to_list()
    return sorted(models, key=get_size_and_it)


# ---------------------
# Other Utils
# ---------------------


def model_to_key(str: str):
    return str.replace("/", "--")


def key_to_model(str: str):
    return str.replace("--", "/")


def prettify_model_name(model_hf_name: str) -> str:
    """Get prettified version of the given model name."""
    dct = {
        # Google Gemma models
        "google/gemma-1.1-2b-it": "Gemma 2B (it)",
        "google/gemma-1.1-7b-it": "Gemma 7B (it)",
        "google/gemma-2b": "Gemma 2B",
        "google/gemma-7b": "Gemma 7B",
        "google/gemma-2-9b": "Gemma 2 9B",
        "google/gemma-2-9b-it": "Gemma 2 9B (it)",
        "google/gemma-2-27b": "Gemma 2 27B",
        "google/gemma-2-27b-it": "Gemma 2 27B (it)",
        # Meta Llama models
        "meta-llama/Meta-Llama-3-70B": "Llama 3 70B",
        "meta-llama/Meta-Llama-3-70B-Instruct": "Llama 3 70B (it)",
        "meta-llama/Meta-Llama-3-8B": "Llama 3 8B",
        "meta-llama/Meta-Llama-3-8B-Instruct": "Llama 3 8B (it)",
        "meta-llama/Meta-Llama-3.1-8B": "Llama 3.1 8B",
        "meta-llama/Meta-Llama-3.1-8B-Instruct": "Llama 3.1 8B (it)",
        "meta-llama/Meta-Llama-3.1-70B": "Llama 3.1 70B",
        "meta-llama/Meta-Llama-3.1-70B-Instruct": "Llama 3.1 70B (it)",
        "meta-llama/Meta-Llama-3.2-1B": "Llama 3.2 1B",
        "meta-llama/Meta-Llama-3.2-1B-Instruct": "Llama 3.2 1B (it)",
        "meta-llama/Meta-Llama-3.2-3B": "Llama 3.2 3B",
        "meta-llama/Meta-Llama-3.2-3B-Instruct": "Llama 3.2 3B (it)",
        "meta-llama/Meta-Llama-3.3-70B-Instruct": "Llama 3.3 70B (it)",
        # Mistral AI models
        "mistralai/Mistral-7B-Instruct-v0.2": "Mistral 7B (it)",
        "mistralai/Mistral-7B-v0.1": "Mistral 7B",
        "mistralai/Mixtral-8x22B-Instruct-v0.1": "Mixtral 8x22B (it)",
        "mistralai/Mixtral-8x22B-v0.1": "Mixtral 8x22B",
        "mistralai/Mixtral-8x7B-Instruct-v0.1": "Mixtral 8x7B (it)",
        "mistralai/Mixtral-8x7B-v0.1": "Mixtral 8x7B",
        "mistralai/Mistral-Small-24B-Base-2501": "Mistral Small 24B",
        "mistralai/Mistral-Small-24B-Instruct-2501": "Mistral Small 24B (it)",
        # Yi models
        "01-ai/Yi-34B": "Yi 34B",
        "01-ai/Yi-34B-Chat": "Yi 34B (chat)",
        "01-ai/Yi-6B-Chat": "Yi 6B (chat)",
        "01-ai/Yi-6B": "Yi 6B",
        "01-ai/Yi-1.5-6B": "Yi 1.5 6B",
        # Qwen2 models
        "Qwen/Qwen2-1.5B": "Qwen 2 1.5B",
        "Qwen/Qwen2-1.5B-Instruct": "Qwen 2 1.5B (it)",
        "Qwen/Qwen2-7B": "Qwen 2 7B",
        "Qwen/Qwen2-7B-Instruct": "Qwen 2 7B (it)",
        "Qwen/Qwen2-72B": "Qwen 2 72B",
        "Qwen/Qwen2-72B-Instruct": "Qwen 2 72B (it)",
        "Qwen/Qwen2.5-7B": "Qwen 2.5 7B",
        "Qwen/Qwen2.5-7B-Instruct": "Qwen 2.5 7B (it)",
        "Qwen/Qwen2.5-72B": "Qwen 2.5 72B",
        "Qwen/Qwen2.5-72B-Instruct": "Qwen 2.5 72B (it)",
        # Tabula
        "mlfoundations/tabula-8b": "Tabula 8B",
        # Olmo
        "allenai/OLMo-1B-0724-hf": "OLMo 1B 0724",
        "allenai/OLMo-1B-hf": "OLMo 1B",
        "allenai/OLMo-7B-0724-hf": "OLMo 7B 0724",
        "allenai/OLMo-7B-hf": "OLMo 7B",
        "allenai/OLMo-7B-Instruct-hf": "OLMo 7B (it)",
        "allenai/OLMo-2-1124-7B": "OLMo 2 7B",
        "allenai/OLMo-2-1124-7B-Instruct": "OLMo 2 7B (it)",
        # GPT
        "gpt-3.5-turbo-0125": "GPT 3.5",
        "gpt-4.1": "GPT 4.1",
    }

    if model_hf_name in dct:
        return dct[model_hf_name]
    else:
        print(f"Couldn't find prettified name for {model_hf_name}.")
        return model_hf_name


def is_instruction_tuned(model_hf_name: str) -> bool:
    """Indicator if a model is instruction tuned (solely inferred from the model name).

    Args:
        model_name (str): name of the model

    Returns:
        bool: model is instruction-finetuned
    """
    indicators = ["Instruct", "it", "Chat"]
    return any(ind in model_hf_name for ind in indicators)


def get_size(model_key):
    if "gpt" in model_key:
        return 10**11
    # just a wrapper to facilitate imports
    return get_model_size_B(model_key)


def get_size_and_it(model_key, eps=0.001):
    return get_size(model_key) + eps * int(is_instruction_tuned(model_key))


def sort_by_size(models: list):
    return sorted(models, key=get_size_and_it)


def sort_by_size_and_family(models: list, factor=1000):
    # factor to ensure model families are well separated
    model_family_to_key = {k: v for v, k in enumerate(model_families, start=1)}

    def sort_key(model_key):
        return get_size_and_it(model_key) + factor * model_family_to_key.get(
            next(
                (mf for mf in model_families if mf.lower() in model_key.lower()), None
            ),
            -1,
        )

    return sorted(models, key=sort_key)


def df_to_dict(df):
    """
    Converts a two-column DataFrame into a dictionary, ignoring the index.
    Assumes the first column is keys and the second column is values.
    """
    assert df.shape[1] == 2, "Assume df has 2 columns"
    return dict(zip(df.iloc[:, 0], df.iloc[:, 1]))


def parse_model_name(name: str) -> str:
    """Parse model name from model key"""
    name = name[name.find("--") + 2 :]
    return name


def get_base_name(name):
    name = re.sub(r"-(Instruct|Chat|it|1\.1)$", "", name, count=1)
    name = re.sub(r"-v0\.2$", "-v0.1", name, count=1)
    return name


def cumulative_sum(array: np.ndarray, end=None) -> np.ndarray:
    """Cumulative sum of a numpy array.

    Args:
        array (np.ndarray): Input array
        end (str, optional): If 'left', compute cumulative sum from right to left.
                             Default is None, computing the sum left to right.

    Returns:
        np.ndarray: cumulative sum
    """
    if end == "left":
        # from right to left (total at xmin)
        return np.flip(np.cumsum(np.flip(array, axis=0), axis=0), axis=0)
    else:
        # from left to right (total at xmax)
        return np.cumsum(array, axis=0)


def apply_cumulative(
    prob_distribution: np.ndarray, at_least: bool = False
) -> np.ndarray:
    """Apply cumulative transformation to a probability distribution.

    Args:
        prob_distribution (np.ndarray:): probability mass of the distribution
        at_least (bool, optional): If true, compute P(X>k), else P(X<=k). Defaults to False.

    Returns:
        np.ndarray:: cumulative sum of the PMF
    """
    if at_least:
        return 1.0 - cumulative_sum(prob_distribution)
    else:
        return cumulative_sum(prob_distribution)


def truncate(num: float, digits: int = 6) -> float:
    return round(num - 10**-digits / 2, digits)


def binarize_using_threshold(col, evals: dict):
    m = col.name
    return col > evals[m]["threshold"]


def no_leading_zero(x, pos):
    if abs(x) < 1:
        return f"{x:.2f}".lstrip("0").replace("-0", "-")
    else:
        return f"{x:.2f}"


def filter_valid_models(predictions, baseline_rates):
    nan_mask = predictions.isnan().any(dim=0)
    if nan_mask.any():
        logging.warning(f"Ignoring {nan_mask.sum().item()} models with NaN values.")
        predictions = predictions[:, ~nan_mask]
        if baseline_rates is not None:
            baseline_rates = baseline_rates[~nan_mask]
    return predictions, baseline_rates


def to_numpy(array):
    if isinstance(array, torch.Tensor):
        return array.numpy()
    return array


def extract_developer(model_name):
    if "/" in model_name:
        dev = model_name.split("/")[0].lower()
        return developer_map.get(dev, dev.capitalize())
    elif model_name.lower().startswith("gpt"):
        return "OpenAI"
    print("Model developer not found in mapping, return 'Unknown'.")
    return "Unknown"


def extract_model_family(model_name: str):
    model_part = model_name.split("/")[-1].lower()
    # GPT
    if model_part.startswith("gpt"):
        return "GPT"
    # Yi
    if model_part.startswith("yi"):
        return "Yi"

    # Gemma and OLMo
    for fam in ["gemma", "olmo"]:
        match = re.search(rf"{fam}(?:-(\d))?(?=-)", model_part)
        if match:
            gen = match.group(1)
            return f"{fam.title()} {gen if gen else ''}"

    # Llama
    for fam in ["llama", "qwen"]:
        match = re.search(rf"{fam}[-\s]?(\d+(?:\.\d+)?)(?=-)", model_part)
        if match:
            gen = match.group(1)
            try:
                gen = int(gen)
            except ValueError:
                gen = float(gen)
            return f"{fam.title()} {gen}"

    # Mistral
    if model_part.startswith("mistral"):
        return "Mistral"
    elif model_part.startswith("mixtral"):
        return "Mistral MoE"

    raise ValueError("No matching model family found. Implement if needed.")


def create_df_family_dev(models=LLM_MODELS):
    result_dict = [
        {
            "model": model_to_key(m),
            "developer": extract_developer(m),
            "family": extract_model_family(m),
        }
        for m in models
    ]
    return pd.DataFrame(result_dict)


def no_leading_0():
    """
    Returns a FuncFormatter that labels only the first and last values in x_vals.

    Usage:
        ax.xaxis.set_major_formatter(endpoints_only(x_vals))
    """

    def formatter(val, pos):
        if val.is_integer():  # integer -> no decimal places
            return f"{int(val)}"
        else:  # float -> 2 decimals, no leading zero
            return f"{val:g}".lstrip("0")

    return mticker.FuncFormatter(formatter)


def endpoints_only(x_vals):
    """
    Returns a FuncFormatter that labels only the first and last values in x_vals.

    Usage:
        ax.xaxis.set_major_formatter(endpoints_only(x_vals))
    """

    def formatter(x_val, pos):
        if x_val == x_vals[0] or x_val == x_vals[-1]:
            return str(x_val)
        return ""

    return mticker.FuncFormatter(formatter)


def min_mid_max(x_vals):
    """
    Returns a FuncFormatter that labels only the first and last values in x_vals.

    Usage:
        ax.xaxis.set_major_formatter(endpoints_only(x_vals))
    """

    x_min, x_max = min(x_vals), max(x_vals)
    x_mid = 0.5 * (x_min + x_max)

    # Use MaxNLocator to get "nice" ticks
    # locator = mticker.MaxNLocator(nbins=3, prune=None)
    # ticks = locator.tick_values(x_min, x_max)
    tick_labels = np.round([x_min, x_mid, x_max], 2)

    # # Select ticks closest to min, mid, max
    # def closest(value):
    #     return ticks[np.argmin(np.abs(ticks - value))]

    # tick_labels = [closest(x_min), closest(x_mid), closest(x_max)]

    print(tick_labels)

    def formatter(val, pos):
        if any(np.isclose(val, t) for t in tick_labels):
            return f"{val:g}"
        return ""

    return mticker.FuncFormatter(formatter)


def add_newline_camelcase(s):
    # Insert a newline before each uppercase letter that follows a lowercase letter
    return re.sub(r"(?<=[a-z])(?=[A-Z])", "\n", s)
