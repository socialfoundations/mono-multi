from __future__ import annotations

import logging
from abc import ABC  # , abstractmethod

import numpy as np
import pandas as pd
import torch

from pathlib import Path

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier, XGBModel
from sklearn.dummy import DummyClassifier

from folktexts.task import TaskMetadata
from folktexts._utils import hash_dict  # , hash_function


SCORE_COL_NAME = "risk_score"
LABEL_COL_NAME = "label"

BASELINES = {
    "Constant": DummyClassifier,
    "LogisticRegression": LogisticRegression,
    "GBM": HistGradientBoostingClassifier,
    "XGBoost": XGBClassifier,
    "NN": MLPClassifier,
}


# baselines = {
#     "Constant": DummyClassifier(strategy="prior"),
#     "LR": LogisticRegression(),  # (penalty='l2', *, dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1,
#                                     class_weight=None, random_state=None, solver='lbfgs', max_iter=100,
#                                     multi_class='deprecated', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)
#     "GBM": HistGradientBoostingClassifier(),  #
#     "XGBoost": XGBClassifier(),  #
# }


class BaselineClassifier(ClassifierMixin, BaseEstimator, ABC):
    def __init__(
        self,
        model_name: str,
        task: TaskMetadata | str,
        threshold: float = 0.5,
        seed: int = 42,
        clf_params: dict = {},  # hyperparameters for the classifier
        **inference_kwargs,
    ):
        self._model_name = model_name

        filtered_params = {
            k: v
            for k, v in clf_params.items()
            if k
            in (
                BASELINES[model_name]._get_param_names()
                if model_name != "XGBoost"
                else XGBModel._get_param_names()
            )
        }
        if len(filtered_params) < len(clf_params):
            logging.warning(
                f"Did not recognize: {[k for k in clf_params.keys() if k not in filtered_params.keys()]}"
            )
            if len(filtered_params) > 0:
                logging.warning(
                    f"Using only recognized hyperparameters: {filtered_params}"
                )

        self.clf_params = filtered_params

        # set seed
        if filtered_params.get("random_state"):
            logging.warning("seed overwritten by 'random_state'")
            self._seed = filtered_params.get("random_state")
        else:
            self.clf_params["random_state"] = seed
            self._seed = seed

        # initialize baseline model
        self._clf = BASELINES[model_name](**filtered_params)

        self._task = TaskMetadata.get_task(task) if isinstance(task, str) else task
        self._threshold = threshold

        # Default inference kwargs
        self._inference_kwargs = inference_kwargs

        # Fixed sklearn parameters
        self.classes_ = np.array([0, 1])
        self._is_fitted = False

    def __hash__(self) -> int:
        """Generate a unique hash for this object."""

        # All parameters that affect the model's behavior
        hash_params = dict(
            model_name=self.model_name,
            task_hash=hash(self.task),
            threshold=self.threshold,
        )

        return int(hash_dict(hash_params), 16)

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def task(self) -> TaskMetadata:
        return self._task

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float):
        if not 0 <= value <= 1:
            logging.error(f"Threshold must be between 0 and 1; got {value}.")

        # Clip threshold to valid range
        self._threshold = np.clip(value, 0, 1)
        logging.warning(f"Setting {self.model_name} threshold to {self._threshold}.")

    @property
    def seed(self) -> int:
        return self._seed

    @property
    def inference_kwargs(self) -> dict:
        return self._inference_kwargs

    def set_inference_kwargs(self, **kwargs):
        """Set inference kwargs for the model."""
        self._inference_kwargs.update(kwargs)

    def __sklearn_is_fitted__(self):
        """Check fitted status and return a Boolean value."""
        return hasattr(self, "_is_fitted") and self._is_fitted

    @staticmethod
    def _get_positive_class_scores(risk_scores: np.ndarray) -> np.ndarray:
        """Helper function to get positive class scores from risk scores."""
        if len(risk_scores.shape) > 1:
            return risk_scores[:, -1]
        else:
            return risk_scores

    @staticmethod
    def _make_predictions_multiclass(pos_class_scores: np.ndarray) -> np.ndarray:
        """Converts positive class scores to multiclass scores."""
        return np.column_stack([1 - pos_class_scores, pos_class_scores])

    def fit(
        self,
        X_train: pd.DataFrame | torch.Tensor,
        y_train: pd.Series | np.ndarray | torch.Tensor,
        fillna: bool = False,
    ):
        assert len(X_train) == len(y_train)
        train_nan_count = X_train.isna().any(axis=1).sum()
        if fillna and train_nan_count > 0:
            logging.info(f"Found {train_nan_count} NaN values, fill with -1.")
            # Fill NaNs with value=-1
            X_train = X_train.fillna(axis="columns", value=-1)

        if self._task.name.startswith("BRFSS"):
            print("Filling NOT_ASKED_MISSING values with -2.")
            for col in X_train.select_dtypes(exclude=[np.number]).columns:
                X_train.replace(to_replace="NOTASKED_MISSING", value=-2.0, inplace=True)
                X_train[col] = pd.to_numeric(X_train[col])

        # Fit on train data
        self._clf.fit(X_train, y_train)
        self._is_fitted = True

        return self

    def _load_predictions_from_disk(
        self,
        predictions_save_path: str | Path,
        data: pd.DataFrame,
    ) -> np.ndarray | None:
        """Attempts to load pre-computed predictions from disk."""

        # Load predictions from disk
        predictions_save_path = Path(predictions_save_path).with_suffix(".csv")
        predictions_df = pd.read_csv(predictions_save_path, index_col=0)

        # Check if index matches our current dataframe
        if predictions_df.index.equals(data.index):
            return predictions_df[SCORE_COL_NAME].values
        else:
            logging.error("Saved predictions do not match the current dataframe.")
            return None

    def predict(
        self,
        data: pd.DataFrame,
        predictions_save_path: str | Path = None,
        labels: pd.Series | np.ndarray = None,
        fillna: bool = False,
    ):
        if fillna:
            print("X_train: Filling NaN values with -1.")
            # Fill NaNs with value=-1
            data = data.fillna(axis="columns", value=-1)
        if self._task.name.startswith("BRFSS"):
            print("X_train: Filling NOT_ASKED_MISSING values with -2.")
            for col in data.select_dtypes(exclude=[np.number]).columns:
                data.replace(to_replace="NOTASKED_MISSING", value=-2.0, inplace=True)
                data[col] = pd.to_numeric(data[col])

        risk_scores = self.predict_proba(
            data,
            predictions_save_path=predictions_save_path,
            labels=labels,
        )

        return (self._get_positive_class_scores(risk_scores) >= self.threshold).astype(
            int
        )

    def predict_proba(
        self,
        data: pd.DataFrame,
        predictions_save_path: str | Path = None,
        labels: pd.Series | np.ndarray = None,
        fillna: bool = False,
    ):
        if labels is not None and predictions_save_path is None:
            logging.error(
                "** Ignoring `labels` argument as `predictions_save_path` was not provided. **"
                "The `labels` argument is only used in conjunction with "
                "`predictions_save_path` to save alongside predictions to disk. "
            )

        # Check if `predictions_save_path` exists and load predictions if possible
        if predictions_save_path is not None and Path(predictions_save_path).exists():
            result = self._load_predictions_from_disk(predictions_save_path, data=data)
            if result is not None:
                logging.info(f"Loaded predictions from {predictions_save_path}.")
                return self._make_predictions_multiclass(result)
            else:
                logging.error(
                    f"Failed to load predictions from {predictions_save_path}. "
                    f"Re-computing predictions and overwriting local file..."
                )

        if not isinstance(data, pd.DataFrame):
            raise ValueError(
                f"`data` must be a pd.DataFrame, received {type(data)} instead."
            )

        # Compute risk scores
        if fillna:
            logging.info("X_test: Filling NaN values with -1.")
            # Fill NaNs with value=-1
            data = data.fillna(axis="columns", value=-1)
        if self._task.name.startswith("BRFSS"):
            print("X_test: Filling NOT_ASKED_MISSING values with -2.")
            for col in data.select_dtypes(exclude=[np.number]).columns:
                data.replace(to_replace="NOTASKED_MISSING", value=-2.0, inplace=True)
                data[col] = pd.to_numeric(data[col])

        risk_scores = self._clf.predict_proba(data)

        # Save to disk if `predictions_save_path` is provided
        if predictions_save_path is not None:
            predictions_save_path = Path(predictions_save_path).with_suffix(".csv")

            predictions_df = pd.DataFrame(
                self._get_positive_class_scores(risk_scores),
                index=data.index,
                columns=[SCORE_COL_NAME],
            )
            predictions_df[LABEL_COL_NAME] = labels
            predictions_df.to_csv(predictions_save_path, index=True, mode="w")

        return risk_scores

    def add_meta_data(self, results: dict):
        updated_results = results.copy()
        updated_results["config_task_name"] = self._task
        updated_results["config_model_name"] = self._model_name
        updated_results["config_params"] = self.clf_params  # TODO: Check how to save
        updated_results["name"] = self._model_name
        using_all_features = self._clf.n_features_in_ == len(self._task.features)
        updated_results["num_features"] = (
            -1 if using_all_features else self._clf.n_features_in_
        )
        updated_results["uses_all_features"] = using_all_features
        return updated_results
