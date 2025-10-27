from mono_multi.setup import BASELINE_RESULTS_PATH
from folktexts._io import load_json
import pandas as pd
import os
from pathlib import Path


def load_baselines(baselines: dict, tasks: list, rerun: bool = False) -> tuple:
    baseline_results_all_tasks = {}
    baseline_risk_scores_all_tasks = {}
    for task_name in tasks:
        print(f"Loading baselines for {task_name}.")
        results = {}
        risk_scores = []
        for clf_name, clf in baselines.items():
            clf_path = (
                BASELINE_RESULTS_PATH
                / f"model-{clf_name}"
                / f"{clf_name}_task-{task_name}"
            )
            json_path = None
            csv_path = None
            bench_folders = [
                f
                for f in os.listdir(clf_path)
                if os.path.isdir(os.path.join(clf_path, f))
            ]
            assert len(bench_folders) == 1
            for bench in bench_folders:
                for subroot, subdirs, files in os.walk(Path(clf_path) / bench):
                    for file in files:
                        if file.endswith(".json"):
                            json_path = Path(clf_path) / bench / file
                        elif file.endswith(".csv"):
                            csv_path = Path(clf_path) / bench / file
                        if json_path and csv_path:
                            break
            if json_path and csv_path:
                print(f"- {clf_name}: Load predictions from '{Path(clf_path)/bench}'.")
                scores = (
                    pd.read_csv(csv_path, index_col=0)
                    .drop("label", axis=1)
                    .rename(columns={"risk_score": clf_name})
                )
                prediction_eval = load_json(json_path)
            else:
                print(f"Skipping {clf_name}")
                prediction_eval = {}
                scores = pd.Series()
            results[clf_name] = prediction_eval
            risk_scores.append(scores)

        baseline_results_all_tasks[task_name] = results
        baseline_risk_scores_all_tasks[task_name] = pd.concat(risk_scores, axis=1)

    return baseline_risk_scores_all_tasks, baseline_results_all_tasks
