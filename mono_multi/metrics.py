from typing import List, Union, Tuple, Callable
import pandas as pd
import itertools
import numpy as np
import math

import scipy as sp
from .setup import LLM_MODELS
from .utils import key_to_model, apply_cumulative
from dataclasses import dataclass

import logging

# ------------------------------
# Baseline
# ------------------------------


def poisson_binom_agreement(
    success_rates: Union[List[float], np.ndarray],
    k: int,
) -> np.float64:
    """
    Poisson Binomal Model: Assume each model to correspond to an independent
    Bernoulli trial with the sucess probability theta (acc, TPR,...).
    What is the probability of k models to succeed (e.g. correct classificatiom, accept)?

    Args:
        success_rates (Union[List[float], np.ndarray]): success probabilities of all models (accuracies, TPR, ...)
        k (int): number of successes

    Returns:
         np.float64: probability of k successes under the Possion Bionomial model
    """
    return sp.stats.poisson_binom.pmf(k=k, p=success_rates)


def compute_poisson_binom_agreement_distribution(
    baseline_rates: Union[List[float], np.ndarray],
    return_success_fractions: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute probabilities for all possible outcomes (any number of successes).

    Args:
        baseline_rates (Union[List[float],  np.ndarray]): success probabilities of all models
        return_success_fractions (bool, optional): Whether to return numer of successes as fractions. Defaults to False.

    Returns:
        Tuple[ np.ndarray,  np.ndarray]: number of successes, corresponding PMF
    """
    if isinstance(baseline_rates, np.ndarray):
        baseline_rates = np.atleast_1d(baseline_rates.squeeze())
        if baseline_rates.ndim == 2:
            N, M = baseline_rates.shape
        else:
            M = baseline_rates.size
    else:
        M = len(baseline_rates)
    successes = np.arange(M + 1)
    prob_expected = np.array(
        [
            poisson_binom_agreement(success_rates=baseline_rates, k=num_agreeing)
            for num_agreeing in successes
        ]
    )
    if return_success_fractions:
        successes = successes / M
    assert prob_expected.ndim == 1, "Return a 1D array with the expected probabilities."
    return successes, prob_expected


def compute_poision_binom_cdf(
    baseline_rates: Union[List[float], np.ndarray],
    return_success_fractions: bool = False,
    at_least: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the cumulative density function under the poisson binomial distribution.

    Args:
        baseline_rates (Union[List[float], np.ndarray]): Success probabilities
        return_success_fractions (bool, optional): Whether to retrun number of successes as fraction. Defaults to False.
        at_least (bool, optional): If true, compute P(X>k), else P(X<=k). Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray]: number of successes, cumulative probability
    """
    successes, prob_expected = compute_poisson_binom_agreement_distribution(
        baseline_rates=baseline_rates, return_success_fractions=return_success_fractions
    )
    cumulative_probs = apply_cumulative(
        prob_distribution=prob_expected, at_least=at_least
    )

    return successes, cumulative_probs


# ------------------------------
# Agreement and Recourse
# ------------------------------


def pairwise_agreement_at_random(acc1: float, acc2: float) -> float:
    return acc1 * acc2 + (1 - acc1) * (1 - acc2)


def get_observed_pairwise_agreement_rate(
    predictions: Union[pd.DataFrame, np.ndarray], predicted_val: int = None
) -> float:
    # if isinstance(predictions, pd.DataFrame):
    #     print("cols", predictions.columns)
    assert predictions.shape[1] == 2, f"Expected 2 columns, got {predictions.shape[1]}"
    if predicted_val is not None:  # conditioned on predicted value
        return _pairwise_agree_in_pred(predictions=predictions, val=predicted_val)
    # Check rows for NaNs
    if isinstance(predictions, pd.DataFrame):
        nan_mask = predictions.notna().any(axis=1)
        predictions_match = (
            predictions[nan_mask].iloc[:, 0] == predictions[nan_mask].iloc[:, 1]
        ).values
    elif isinstance(predictions, np.ndarray):
        nan_mask = np.logical_not(np.isnan(predictions).any(axis=1))
        predictions_match = predictions[nan_mask][:, 0] == predictions[nan_mask][:, 1]
    num_samples = predictions[nan_mask].shape[0]
    return (predictions_match).sum(axis=0) / num_samples


def _pairwise_agree_in_pred(predictions: Union[pd.DataFrame, np.ndarray], val: int):
    assert predictions.shape[1] == 2, f"Expected 2 columns, got {predictions.shape[1]}"
    if isinstance(predictions, pd.DataFrame):
        predictions = predictions.values
    num_samples = predictions.shape[0]
    observed_agreement = (np.all(predictions == val, axis=1)).sum(axis=0) / num_samples
    return observed_agreement


def get_pairwise_neg_agreement_rate(
    predictions: Union[pd.DataFrame, np.ndarray],
) -> float:
    return _pairwise_agree_in_pred(predictions=predictions, val=0)


def get_pairwise_pos_agreement_rate(
    predictions: Union[pd.DataFrame, np.ndarray],
) -> float:
    return _pairwise_agree_in_pred(predictions=predictions, val=1)


def ratio_agreement_wrapper(predictions: pd.DataFrame, evals: pd.DataFrame):
    assert predictions.shape[1] == 2, f"Expected 2 columns, got {predictions.shape[1]}"
    assert (
        predictions.columns == evals.columns
    ), "Both DataFrames should have the same columns names"

    observed = get_observed_pairwise_agreement_rate(predictions=predictions)
    expected = pairwise_agreement_at_random(evals.iloc[:, 0], evals.iloc[:, 1])
    return observed / expected


def diff_agreement_wrapper(predictions: pd.DataFrame, evals: pd.DataFrame):
    assert predictions.shape[1] == 2, f"Expected 2 columns, got {predictions.shape[1]}"
    assert (
        predictions.columns == evals.columns
    ), "Both DataFrames should have the same columns names"

    observed = get_observed_pairwise_agreement_rate(predictions=predictions)
    expected = pairwise_agreement_at_random(evals.iloc[:, 0], evals.iloc[:, 1])
    return observed - expected


def get_fraction_no_recourse(
    predictions: np.ndarray,
    count_acceptances=True,
):
    num_models, fraction_individuals = get_model_agreement_histogram(
        predictions=predictions,
        count_acceptances=count_acceptances,
        fraction_individuals=True,
        padding=True,
    )
    # no recourse: num_models = 0
    return fraction_individuals[num_models == 0]


def get_fraction_obs_acceptance_aggregated(
    predictions: Union[pd.DataFrame, np.array],
    padding=True,  # not used, fractions cannot be meaningfully padded
) -> tuple[np.ndarray, np.ndarray]:
    return get_model_agreement_histogram(
        predictions=predictions,
        fraction_models=True,
        padding=False,
        count_acceptances=True,
    )


def get_fraction_obs_rejections_aggregated(
    predictions: Union[pd.DataFrame, np.ndarray],
    padding=False,  # not used, fractions cannot be meaningfully padded
) -> tuple[np.ndarray, np.ndarray]:
    return get_model_agreement_histogram(
        predictions=predictions,
        fraction_models=True,
        padding=False,
        count_acceptances=False,
    )


def compute_observed_recourse(
    predictions: Union[pd.DataFrame, np.ndarray],
    return_fractions: bool = False,
    count_acceptances: bool = True,
):

    if isinstance(predictions, pd.DataFrame):
        predictions = predictions.values

    N, M = predictions.shape
    # M per individual: filter out NaN values
    M_not_na = np.sum(~np.isnan(predictions), axis=1)
    num_nan_models = np.any(np.isnan(predictions), axis=0).sum()
    if num_nan_models > 0:
        logging.warning(f"{num_nan_models} models contain NaN values.")

    # for every inidvidual, count positive predictions
    counts_per_instance = np.nansum(predictions, axis=1)
    if not count_acceptances:
        # count negative prediction
        counts_per_instance = M_not_na - counts_per_instance

    if return_fractions:
        counts_per_instance = counts_per_instance / M_not_na

    # counts_per_instance = torch.tensor(counts_per_instance)

    return counts_per_instance


def is_subset_close(array: np.ndarray, reference: np.ndarray, atol: float = 1e-8):
    array = np.expand_dims(array.astype(np.float64), axis=-1)  # (I,1)
    reference = reference.astype(np.float64)  # (J,)

    # are all elemenents of tensor close to a reference elem
    array_elems_close = np.any(
        np.isclose(array, reference, atol=atol), axis=1
    )  # I x J -> (I,)

    return all(array_elems_close)


def get_model_agreement_histogram(
    predictions: Union[pd.DataFrame, np.ndarray],
    count_acceptances: bool = True,
    fraction_models: bool = False,
    fraction_individuals: bool = False,
    padding: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes histogram of model agreements (either acceptances or rejections).

    Parameters:
        predictions: Predictions (instances, models).
        y_true: optional binary labels
        restrict_only_pos_instances: Whether to include only positive instances.
        restrict_only_neg_instances: Whether to include only negative instances.
        count_acceptances: If False, counts rejections instead.
        padding: whether to return a padded histogram (length M+1).

    Returns:
        (bin_values, counts): Tensors of unique counts or full histogram.
    """
    M = predictions.shape[1]
    agree_per_instance = compute_observed_recourse(
        predictions=predictions,
        return_fractions=fraction_models,
        count_acceptances=count_acceptances,
    )  # .numpy()

    # aggregate observed agreements into histogram
    models_agreeing, num_individuals = np.unique(agree_per_instance, return_counts=True)
    # sort by increasing agreement
    sort_idx = np.argsort(models_agreeing, axis=0)
    models_agreeing = models_agreeing[sort_idx]
    num_individuals = num_individuals[sort_idx]

    # apply padding (as long as models are not integers)
    if padding:
        valid_bins = np.arange(M + 1)
        if fraction_models:
            valid_bins = np.linspace(0, 1, M + 1)
            # check if subset of valid bins (at least close)
            if not is_subset_close(models_agreeing, valid_bins):
                logging.warning(
                    "No meaningful padding, because data contains NaN values. Returning sorted values."
                )

                return np.sort(agree_per_instance)[0], (
                    np.linspace(0, 1, num=len(agree_per_instance))
                    if fraction_individuals
                    else np.arange(len(agree_per_instance))
                )
        padded_counts = np.zeros_like(valid_bins, dtype=int)
        idx = np.searchsorted(valid_bins, models_agreeing)
        padded_counts[idx] = num_individuals
        models_agreeing, num_individuals = valid_bins, padded_counts

    if fraction_individuals:
        num_individuals = num_individuals / num_individuals.sum()

    return models_agreeing, num_individuals


def compute_monoculture_recourse_step(
    baseline_rates: np.ndarray, at_least: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute fraction models and individuals for monoculture curve

    Args:
        baseline_rates (np.ndarray): Baseline acceptance/rejection per model
        at_least (bool, optional): If True "x individuals are accepted by at leat y models" (P(#accept >= y)),
                                   if False "x individuals are accepted by at most y models" (P(#accept <= y)).
                                   Defaults to False.

    Returns:
        Tuple[list[float], list[float]]: fraction individuals, fraction models
    """
    mean_rate = np.mean(baseline_rates)
    if at_least:
        # step from (0,1) -> (mean_rate,1) -> (1,0)
        fraction_individuals = [mean_rate, 1.0 - mean_rate]
        fraction_models = [1.0, 0.0]
    else:
        # step from (0,0) -> (1-mean_rate,0) -> (1,1)
        fraction_individuals = [1.0 - mean_rate, mean_rate]
        fraction_models = [0.0, 1.0]

    return np.array(fraction_individuals), np.array(fraction_models)


# Bootstrapping
@dataclass
class BootstrapResult:
    """Result object

    Attributes
    ----------
    bootstrap_distribution : ndarray
        The bootstrap distribution, that is, the value of `statistic` for
        each resample. The last dimension corresponds with the resamples
        (e.g. ``res.bootstrap_distribution.shape[-1] == n_resamples``).
    standard_error : float or ndarray
        The bootstrap standard error, that is, the sample standard
        deviation of the bootstrap distribution.

    """

    bootstrap_distribution: np.ndarray
    mean: float | np.ndarray
    standard_error: float | np.ndarray
    # percentile: Tuple[float] | np.ndarray


def sample_combinations(
    choices: np.ndarray, k: int, num_resamples: int, rng: np.random.Generator
):
    """Sample unique combinations of k elements from choices."""
    n = len(choices)
    if k > n:
        raise ValueError(f"Cannot choose {k} elements from {n} without replacement.")
    try:
        num_possible_combinations = math.comb(n, k)
        print(f"(n, k)=({n}, {k})={num_possible_combinations}")
    except ValueError:
        print(
            f"Could not determine number of combinations ((n, k)=({n}, {k})), sample instead."
        )
        num_possible_combinations = num_resamples + 1  # go into else case

    if num_possible_combinations <= num_resamples:
        # return all possible combinations
        return np.array(list(itertools.combinations(choices, k)))
    else:
        # sample num_resamples unique combinations
        combinations = set()
        while len(combinations) < num_resamples:
            c = tuple(sorted(rng.choice(choices, size=k, replace=False)))
            combinations.add(c)
        return np.array(list(combinations))


def resample_predictions(
    predictions: np.ndarray,
    axis=0,
    num_samples: int = None,
    num_resamples: int = 500,
    rng: np.random.Generator = np.random.default_rng(),
    return_idx: bool = False,
) -> np.ndarray:
    """Sample subsets of predictions with replacement

    Args:
        predictions (np.ndarray): dataset
        axis (int, optional): Wich axis to resample, 0 for rows, 1 for columns. Defaults to 0.
        num_samples (int, optional): number of samples to draw, use to restrict to a smaller set. Defaults to None.
        num_resamples (int, optional): number of datasets drawn. Defaults to 10.
        rng (Generator, optional): random number generator. Defaults to np.random.default_rng().

    Returns:
        np.ndarray: set of resampled datasets (num_resamples x N x M, replace N or M by num_samples when provided)
    """
    assert isinstance(predictions, np.ndarray), "Provide predictions as np.array"
    # N, M = predictions.shape
    # print(f"N,M={N,M}, num_samples={num_samples}, num_resamples = {num_resamples}")

    choices = np.arange(predictions.shape[axis])
    num_choices = len(
        choices
    )  # N if row indices (axis = 0), M if column indices (axis=1)
    k = num_samples if num_samples else num_choices

    if num_samples and num_samples <= num_choices:
        # downsample, without replacement
        idx = sample_combinations(
            choices=choices, k=k, num_resamples=num_resamples, rng=rng
        )
    else:
        # Sample {num_resamples}x{k} with replacement
        idx = np.vstack(
            [rng.choice(choices, size=k, replace=True) for _ in range(num_resamples)]
        )
    if axis == 0:
        # row indices
        resamples = predictions[idx, :]  # resampling = 1.st axis
    else:
        resamples = np.moveaxis(
            predictions[:, idx], 1, 0
        )  # move axis for resamplinhg axis to be the first

    if return_idx:
        return resamples, idx
    return resamples


def bootstrap_predictions(
    predictions: Union[np.ndarray, pd.DataFrame],
    stat: Callable,
    axis=0,
    num_samples: int = None,
    num_resamples: int = 500,
    rng=np.random.default_rng(),
) -> BootstrapResult | List[BootstrapResult]:
    """Bootstrap predictions (either rows or columns)

    Args:
        predictions (np.ndarray): dataset, N x M
        stat (callable): function to call on each dataset draw
        axis (int, optional): Axis along which resampling is applied, 0 for rows, 1 for columns. Defaults to 0.
        num_samples (int, optional): Provide to restrict to fixed number of rows/columns,
                                     otherwise no subsampling is done. Defaults to None.
        num_resamples (int, optional): Number of repeated draws. Defaults to 10.
        rng (Generator, optional): Random number generator. Defaults to np.random.default_rng().

    Returns:
        BootstrapResult: results (draws, mean and std error) of bootstrapping
    """
    assert predictions.ndim == 2, "Assuming predictions have two dimensions"
    N, M = predictions.shape
    prediction_values = (
        predictions if isinstance(predictions, np.ndarray) else predictions.values
    )
    resampled, idx = resample_predictions(
        predictions=prediction_values,
        axis=axis,
        num_samples=num_samples,
        num_resamples=num_resamples,
        rng=rng,
        return_idx=True,
    )
    results = []
    for i, sample in enumerate(resampled):
        # ensure we're iterating over the resamples
        assert (
            sample.shape == (N, M)
            if num_samples is None
            else (num_samples, M) if axis == 0 else (N, num_samples)
        )
        if isinstance(predictions, pd.DataFrame):
            if axis == 0:
                sample_df = predictions.iloc[idx[i], :]
            elif axis == 1:
                sample_df = predictions.iloc[:, idx[i]]
            assert (
                sample_df.values == sample
            ).all(), "df and sample should be the same"
            res = stat(sample_df)
        else:
            res = stat(sample)
        results.append(res)

    # if stats returns multiple values
    if isinstance(results[0], Tuple):
        n_outputs = len(results[0])
        bootstrapped_components = []
        for i in range(n_outputs):
            comp = np.stack([r[i] for r in results])  # get i-th component
            ddof = 1 if comp.shape[0] > 1 else 0
            # print(f"comp {i}", comp.shape, comp.mean(axis=0).shape)
            bootstrapped_components.append(
                BootstrapResult(
                    bootstrap_distribution=comp,
                    mean=comp.mean(axis=0),
                    standard_error=comp.std(axis=0, ddof=ddof),
                )
            )
        return bootstrapped_components
    else:
        array = np.stack(results)
        ddof = 1 if array.shape[0] > 1 else 0
        # TODO: return list to unify return type?
        return BootstrapResult(
            bootstrap_distribution=array,
            mean=array.mean(axis=0),
            standard_error=np.std(array, axis=0, ddof=ddof),  # /np.sqrt(num_resamples)
        )


# ------------------------------
# Ambiguity and Discrepancy
# ------------------------------


def get_ambiguity(h0_predictions: pd.Series, predictions: pd.DataFrame):
    N, M = predictions.shape
    # check if same predictions as baseline
    assert all(
        h0_predictions.index == predictions.index
    ), "Index must match to compare correctly"
    # check where different from ref
    diff = predictions.ne(h0_predictions, axis=0)
    # check where not NaN
    valid_mask = predictions.notna()
    # only compare entries that are different and valid
    any_change = (diff & valid_mask).any(axis=1)
    return any_change.sum() / N


def get_discrepancy(h0_predictions: pd.Series, predictions: pd.DataFrame):
    N, M = predictions.shape
    # check if same predictions as baseline
    assert all(
        h0_predictions.index == predictions.index
    ), "Index must match to compare correctly"
    if predictions.isna().any().any():
        print("NaNs contained")
        nan_mask_col = predictions.isna().any()
        num_disagreements_with_h0 = (
            predictions.loc[:, ~nan_mask_col].ne(h0_predictions, axis=0).sum(axis=0)
        )
        max_disagree = num_disagreements_with_h0.max() / N
        # for columns with NaNs check only among those that are not NaN
        no_nan_rows = predictions.notna().all(axis=1)
        num_disagree_filtered_nan = (
            predictions.loc[no_nan_rows, nan_mask_col]
            .ne(h0_predictions[no_nan_rows], axis=0)
            .sum(axis=0)
        ) / h0_predictions[no_nan_rows].shape[0]
        print("reduces to", h0_predictions.shape)
        return max(num_disagree_filtered_nan.max(), max_disagree)
    else:
        num_disagreements_with_h0 = predictions.ne(h0_predictions, axis=0).sum(axis=0)
        return num_disagreements_with_h0.max() / N


def get_ambiguity_all_models(predictions: pd.DataFrame):
    # assume every column contains the predictions of one model
    N, M = predictions.shape
    num_accept = predictions.sum(axis=1)
    # count how often at least 1 model disagrees (1 to (M-1) accepts)
    return num_accept[(num_accept > 0) & (num_accept < M)].shape[0] / N


def get_discrepancy_all_models(predictions: pd.DataFrame):
    # assume every column contains the predictions of one model
    # find max number of different predictions between any two columns (Hamming dist)
    N, M = predictions.shape
    assert all(
        [key_to_model(col) in LLM_MODELS for col in predictions.columns]
    ), "Columns should be only model predictions (column name = model key)."

    max_diff = 0
    for col1, col2 in itertools.combinations(predictions, 2):
        diff = (predictions[col1] != predictions[col2]).sum()
        max_diff = max(max_diff, diff)

    return max_diff / N


# ------------------------------
# UTILS
# ------------------------------


def matrix_pairwise_evals(
    data: pd.DataFrame, fun: Callable, only_lower_diag: bool = True
):
    """evaluate function on each pair of models"""
    assert isinstance(
        data, pd.DataFrame
    ), "Provide data for eval matrix as pd.DataFrame"
    num_models = len(data.columns)
    model_combinations = list(
        itertools.combinations_with_replacement(range(num_models), 2)
    )  # _with_replacement only to add diagonal

    matrix = np.full((num_models, num_models), fill_value=np.nan)
    for idx1, idx2 in model_combinations:
        matrix[idx1, idx2] = fun(data.iloc[:, [idx1, idx2]])
        if not only_lower_diag:
            matrix[idx2, idx1] = matrix[idx1, idx2]

    # return lower diagonal ma
    return matrix.T


def mean_baseline_rate(baseline_rates: Union[List[float], np.ndarray], at_least: bool):
    mean_rate = np.mean(baseline_rates)
    reference_val = abs(float(not at_least) - mean_rate)
    return reference_val
