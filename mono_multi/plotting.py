from __future__ import annotations
import matplotlib as mpl
from matplotlib import axes
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.legend_handler import HandlerTuple
import matplotlib.colors as pltcolors
from matplotlib.patches import Rectangle


from functools import partial

import scipy as sp
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Callable, Iterable, List, Union
from .utils import (
    prettify_model_name,
    key_to_model,
    get_size_and_it,
    apply_cumulative,
    df_to_dict,
    filter_predictions_and_data,
    no_leading_0,
)
from .metrics import (
    matrix_pairwise_evals,
    pairwise_agreement_at_random,
    bootstrap_predictions,
    compute_poisson_binom_agreement_distribution,
    compute_monoculture_recourse_step,
    get_model_agreement_histogram,
    get_observed_pairwise_agreement_rate,
)
from pathlib import Path
import logging
import inspect
from dataclasses import dataclass, asdict


PLOT_DEFAULT_STYLE = {
    "obs_color": "C0",
    "obs_label": "observed",
    "baseline_color": "C1",
    "baseline_label": "random error",
    "monoculture_color": "C3",
    "monoculture_label": "monoculture",
    "random_prediction_color": "C7",
    "random_prediction_label": "random prediction",
    "fill_alpha": 0.1,
    "fill_zorder": -2,
    "fill_facecolor": "C7",
    # "fill_edgecolor": "none",
}

densely_dotted = (0, (1, 1))
# -------------------------------------------------------------
# Utils
# -------------------------------------------------------------


def unique_legend_items(axs) -> dict:
    handles, labels = [], []
    for ax in axs.flat:
        ha, la = ax.get_legend_handles_labels()
        handles.extend(ha)
        labels.extend(la)

    # deduplicate
    unique = dict(zip(labels, handles))
    return unique


def sort_legend_items(legend_items: dict, order: list):
    ordered_labels = [label for label in order if label in legend_items]
    ordered_handles = [legend_items[label] for label in ordered_labels]

    # labels not in predefined order, sorted alphabetically
    remaining_labels = sorted([label for label in legend_items if label not in order])
    remaining_handles = [legend_items[label] for label in remaining_labels]

    # Combine both
    ordered_labels += remaining_labels
    ordered_handles += remaining_handles

    return ordered_labels, ordered_handles


def configure_legend(
    fig,
    axs,
    offset: float = 0.32,
    add_monoculture_mixed_handle: bool = True,
    order: list = ["observed", "random error", "random prediction", "monoculture"],
    color_monoculture: str = PLOT_DEFAULT_STYLE["monoculture_color"],
    columns: bool = True,
    handles_labels: Tuple[list, list] = None,
    **legend_kwargs,
):
    unique = unique_legend_items(axs)

    if add_monoculture_mixed_handle:
        # monoculture
        solid_line = mlines.Line2D([], [], color=color_monoculture, linestyle="-")
        dotted_line = mlines.Line2D([], [], color=color_monoculture, linestyle=":")
        unique["monoculture"] = (solid_line, dotted_line)

    if handles_labels is not None:
        handles, labels = handles_labels
    else:
        if len(order) > 0:
            labels, handles = sort_legend_items(legend_items=unique, order=order)
        else:
            labels, handles = zip(*unique.items())

    lowest_y = min(ax.get_position().y0 for ax in axs.flat)

    default_kwargs = {
        "loc": "lower center",
        "frameon": False,
        "bbox_to_anchor": (0.5, lowest_y - offset),
    }
    if columns and "ncol" not in legend_kwargs.keys():
        legend_kwargs["ncol"] = len(labels)
    legend_kwargs = {**default_kwargs, **legend_kwargs}

    fig.legend(
        labels=labels,
        handles=handles,
        handler_map={tuple: HandlerTuple(ndivide=None)},
        **legend_kwargs,
    )


def sort_agreements(
    obs: np.ndarray = None,
    exp: np.ndarray = None,
    sort_by: str = None,
    others: list[np.ndarray] = None,
):
    if others is None:
        others = []
    if sort_by:
        assert sort_by in [
            "observed",
            "expected",
        ], "sort_by must be one of ['observed', 'expected']."
        sort_target = obs if sort_by == "observed" else exp
        sort_indices = np.argsort(sort_target)
        obs = obs[sort_indices] if obs is not None else None
        exp = exp[sort_indices] if exp is not None else None
        others = [arr[sort_indices] if arr is not None else None for arr in others]
    else:
        obs = np.sort(obs) if obs is not None else None
        exp = np.sort(exp) if exp is not None else None
        others = [np.sort(arr) if arr is not None else None for arr in others]
    return obs, exp, others


def get_kwargs_subset_from_defaults(subset_prefix: str):
    return {k: v for k, v in PLOT_DEFAULT_STYLE.items() if k.startswith(subset_prefix)}


def validate_kwargs(target_func, kwargs, prefix=""):
    """
    Validate that kwargs are accepted by the given target function, filter if not accepted.
    Raises TypeError if an invalid argument is found.
    Returns
    """
    valid_params = get_valid_kwargs(target_func)
    cleaned_kwargs = {k.replace(prefix, ""): v for k, v in kwargs.items()}
    invalid = set(cleaned_kwargs) - valid_params
    if invalid:
        logging.warning(
            f"Received invalid kwargs for {target_func}: {sorted(invalid)}\n"
            f"Valid params include: {sorted(valid_params)}"
        )
    filtered_params = {}
    for key, val in cleaned_kwargs.items():
        if key in valid_params:
            filtered_params[key] = val
    return filtered_params


def get_valid_kwargs(func):
    """
    Returns all valid kwargs for a matplotlib function, including those
    passed via **kwargs to underlying Artist or patch constructors.
    """
    sig = inspect.signature(func)
    params = set(sig.parameters.keys())

    # If **kwargs exists, try to find underlying Artist class
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        # Many mpl plotting functions create Artists like Line2D, Patch, etc.
        # We'll look for a default 'data' argument to guess its type.
        # This works for common cases like plt.plot, plt.scatter, plt.bar...
        try:
            # Mapping from mpl function to typical artist type
            func_to_artist = {
                "plot": mpl.lines.Line2D,
                "step": mpl.lines.Line2D,
                "scatter": mpl.collections.PathCollection,
                "bar": mpl.patches.Rectangle,
                "barh": mpl.patches.Rectangle,
                "fill": mpl.patches.Polygon,
                "hist": mpl.patches.Rectangle,
                "fill_between": mpl.collections.PolyCollection,
            }
            name = func.__name__
            if name in func_to_artist:
                artist_cls = func_to_artist[name]
                # Init params
                init_params = set(
                    inspect.signature(artist_cls.__init__).parameters.keys()
                )
                # Create default instance with minimal args
                if artist_cls == mpl.lines.Line2D:
                    artist_instance = artist_cls([], [])
                elif artist_cls in (
                    mpl.collections.PolyCollection,
                    mpl.collections.PathCollection,
                ):
                    artist_instance = artist_cls([])
                elif artist_cls == mpl.patches.Rectangle:
                    # xy, width, height needed
                    artist_instance = artist_cls((0, 0), 1, 1)
                elif artist_cls == mpl.patches.Polygon:
                    # list of vertices needed
                    artist_instance = artist_cls([[0, 0]])
                else:
                    # fallback to calling without args, might fail
                    logging.warning(
                        f"{artist_cls} could not be matched to specific instantiation."
                    )
                    artist_instance = artist_cls()

                prop_names = set(artist_instance.properties().keys())

                params |= init_params | prop_names
                # manually add color
                params.add("color")
        except Exception as e:
            logging.debug(
                f"Extracting detailed kwargs failed, only use parameters from signature. \n {e}"
            )

    return params


def bin_fractions(M: int, fractions_models: np.ndarray, frequencies: np.ndarray):
    assert all(fractions_models <= 1), "Assume models to be given as fractions."
    # define bin edges
    edges = np.linspace(0.0, 1.0, M + 1, dtype=fractions_models.dtype)
    if len(fractions_models) == len(edges) and np.allclose(
        fractions_models, edges, atol=1e-12
    ):
        return fractions_models, frequencies
    counts = np.zeros(M + 1, dtype=frequencies.dtype)

    # assign each fraction to a bin
    bin_idx = np.clip(a=(fractions_models * M).astype(dtype=np.long), a_min=0, a_max=M)

    # accumulate frequencies (add at respective indices)
    np.add.at(counts, bin_idx, frequencies)
    return edges, counts


def add_mini_legend_with_group_info(
    ax,
    colors,
    group_sizes,
    markersize=3,
    bbox_to_anchor=(1.14, 0.5),
):
    # Create mini-legend handles
    handles = [
        plt.Line2D(
            [0], [0], color=colors[g], marker="o", linestyle="", markersize=markersize
        )
        for g, _ in sorted(group_sizes, key=lambda x: x[1])
    ]
    labels = [
        f"{n}" if isinstance(n, int) else f"{n:.2f}"
        for _, n in sorted(group_sizes, key=lambda x: x[1])
    ]
    ax.legend(
        handles,
        labels,
        title="",
        bbox_to_anchor=bbox_to_anchor,
        loc="center",
        fontsize=7,
        frameon=False,
        handletextpad=-0.15,
    )
    return ax


def create_proxy_handle(orig_handle, update_properties={"color": "C7"}):
    if isinstance(orig_handle, mpl.lines.Line2D):
        properties = {
            "color": orig_handle.get_color(),  # new color for the legend
            "linestyle": orig_handle.get_linestyle(),
            "linewidth": orig_handle.get_linewidth(),
            "marker": orig_handle.get_marker(),
            "markersize": orig_handle.get_markersize(),
            "markerfacecolor": orig_handle.get_markerfacecolor(),
            "markeredgecolor": orig_handle.get_markeredgecolor(),
        }
        properties.update(update_properties)
        data = orig_handle.get_data()
        proxy_handle = mpl.lines.Line2D(data[0], data[1], **properties)
    # elif isinstance(orig_handle, mpl.collections.PolyCollection):
    #     properties = {
    #         "facecolor": orig_handle.get_facecolor(),
    #     }
    #     properties.update(update_properties)
    #     proxy_handle = mpl.collections.PolyCollection([], **properties)
    else:
        print("Couldn't find matching handle type")
        return orig_handle
    return proxy_handle


# -------------------------------------------------------------
# GENERAL PERFORMANCE
# -------------------------------------------------------------


def plot_accuracies(
    ax: axes,
    evals: dict,
    baseline_accs: dict,
    models_above_baseline: list,
    ylims: Tuple = (0, 0.85),
    label_rotation: int = 90,
    ha: str = "center",
    color: str = "grey",
    highlight_color: str = "black",
    hline_xmin=0,
    hline_xmax=1,  #
    marker="o",
    scatter_label=None,
):
    if baseline_accs:
        # plot hline for baselines
        for baseline, c in [("Constant", "dodgerblue"), ("XGBoost", "yellowgreen")]:
            ax.hlines(
                baseline_accs[baseline],
                xmin=hline_xmin,
                xmax=hline_xmax,
                label=baseline,
                colors=c,
                linestyle="-",
                zorder=0,
            )

    # plot model accuracies
    x_labels = []
    scatter_labels = set()
    for idx, (model_key, val) in enumerate(
        sorted(evals.items(), key=lambda item: get_size_and_it(item[0]))
    ):
        keep_model = model_key in models_above_baseline
        ax.scatter(
            idx,
            val,
            zorder=1,
            s=20,
            color="black" if keep_model else "grey",
            marker=marker,
            label=scatter_label if scatter_label not in scatter_labels else None,
        )
        scatter_labels.add(scatter_label)
        x_labels.append(model_key)

    # set xticks
    ax.set_xticks(
        np.arange(len(x_labels)),
        [prettify_model_name(key_to_model(key)) for key in x_labels],
        rotation=label_rotation,
        ha=ha,
        color=color,
    )
    for m in x_labels:
        if m in models_above_baseline:
            idx = x_labels.index(m)
            plt.setp(ax.get_xticklabels()[idx], color=highlight_color)
    ax.set_ylim(ylims)
    ax.set_xlim(-0.5, right=len(x_labels))

    return ax


def plot_label_dist(
    ax: axes, labels: pd.Series, ylims: Tuple = (0, 0.75), bar_width=0.2
):
    counts = labels.value_counts()
    ax.bar(
        counts.index * 2 * bar_width, counts.values / labels.shape[0], width=bar_width
    )
    ax.set_xticks(ticks=counts.index * 2 * bar_width, labels=counts.index)
    ax.set_xlabel("$y$")
    ax.set_ylim(ylims)
    return ax


def plot_neg_predicted(
    ax: axes,
    fractions_neg_pred: dict,
    models_above_baseline: list,
    ylims=(0, 1),
    label_rotation: int = 90,
    ha: str = "center",
    color: str = "grey",
    highlight_color: str = "black",
    omit_non_highlight_labels=False,
    bar_width=0.5,
):
    x_labels = []
    models_sorted = sorted(fractions_neg_pred.keys(), key=get_size_and_it)
    for idx, model_key in enumerate(models_sorted):
        keep_model = model_key in models_above_baseline
        ax.bar(
            idx,
            fractions_neg_pred[model_key],
            zorder=1,
            color="black" if keep_model else "grey",
        )
        x_labels.append(model_key)

    # set xticks
    if not omit_non_highlight_labels:
        ax.set_xticks(
            np.arange(len(x_labels)),
            [prettify_model_name(key_to_model(key)) for key in x_labels],
            rotation=label_rotation,
            ha=ha,
            color=color,
        )
        for m in x_labels:
            if m in models_above_baseline:
                idx = x_labels.index(m)
                plt.setp(ax.get_xticklabels()[idx], color=highlight_color)
    else:
        ax.set_xticks(
            np.arange(len(x_labels)),
            [
                (
                    prettify_model_name(key_to_model(key))
                    if key in models_above_baseline
                    else ""
                )
                for key in x_labels
            ],
            rotation=label_rotation,
            ha=ha,
            color=highlight_color,
        )
    ax.set_ylim(ylims)
    ax.set_xlim(-0.5, right=len(x_labels))

    return ax


def plot_general_performance_task(
    axs_row,
    df,
    baseline_evals,
    ytrue,
    models_to_highlight,
    marker: str = "o",
    scatter_label=None,
    acc="accuracy",
    label_rotation: int = 90,
    ha: str = "center",
):
    """Plot label distribution, accuracy and fraction of negative predictions next to each other for a given task"""
    if len(axs_row) == 3:
        ax_label, ax_accuracy, ax_frac_neg = axs_row
    else:
        ax_label, ax_accuracy = axs_row
    # label distribution
    plot_label_dist(ax_label, ytrue, ylims=(0, 0.95))

    # model accuracy
    acc_dict = df_to_dict(df[["model", acc]])
    plot_accuracies(
        ax=ax_accuracy,
        evals=acc_dict,
        baseline_accs={
            baseline: evals[acc] for baseline, evals in baseline_evals.items()
        },
        models_above_baseline=models_to_highlight,
        ylims=(0, 0.95),
        hline_xmin=-0.2,
        hline_xmax=len(acc_dict.keys()) + 0.1,
        marker=marker,
        scatter_label=scatter_label,
        label_rotation=label_rotation,
        ha=ha,
    )

    if len(axs_row) == 3:
        # fraction of negative predicted samples
        frac_neg_pred_dict = df_to_dict(
            pd.concat(
                [df["model"], df["num_pred_negatives"] / df["n_samples"]], axis=1
            ).rename(columns={0: "frac_negative_preds"})
        )
        plot_neg_predicted(
            ax_frac_neg,
            fractions_neg_pred=frac_neg_pred_dict,
            models_above_baseline=models_to_highlight,
            label_rotation=label_rotation,
            ha=ha,
        )


# -------------------------------------------------------------
# AGREEMENT/RECOURSE
# -------------------------------------------------------------


def plot_recourse_lineplot(
    ax: plt.Axes,
    curves: dict[RecourseData],
    num_models: int,
    xlabel: str = "fraction of positive instances",
    ylabel: str = "fraction of positive predictions",
    title: str = "",
    relative_x: bool = True,
    plot_monoc=True,
    plot_baseline: bool = True,
    plot_pdf: bool = True,
    at_least: bool = False,
    num_ticks=5,
    **kwargs,
):
    """
    x individuals are accepted by **at least** y models

    Parameters:
        ax: The matplotlib axis to plot on.
        curves: x,y data for all lines
        plot_cumulative: Whether to plot cumulative probabilities (default = False, else True
          to plot from left to right, else 'left' to add up all the way to the left)
        ylabel: Label for the y-axis.
        baseline_label: Label for the baseline comparison.
        relative_x: Normalize x-axis to [0, 1].
        count_accepted: Count positive predictions (True) or rejections (False), default: True,
        indicate_mean: Whether to plot a small triangle below the x-axis to indicate the mean value, default=False.

    Returns:
        ax: The matplotlib axis with the plot.
    """

    # Validate inputs
    assert set(curves.keys()).issubset({"observed", "baseline", "monoculture"})
    if "observed" not in curves.keys() and "baseline" not in curves.keys():
        raise ValueError("At least one of 'observed' or 'baseline' must be provided.")

    M = num_models
    bar_width = 0.4 / (M + 1) if relative_x else 0.5

    for component in ["monoculture", "baseline", "observed"]:  # fix plotting order
        curve = curves.get(component, None)
        if curve is None:
            continue
        if component == "monoculture" and plot_monoc:
            monoculture_kwargs = {
                k: v for k, v in kwargs.items() if k.startswith("monoculture_")
            }
            plot_recourse_monoculture(
                ax=ax,
                **asdict(curve),
                **monoculture_kwargs,
            )
        elif component == "baseline" and plot_baseline:
            baseline_kwargs = {
                k: v for k, v in kwargs.items() if k.startswith("baseline_")
            }
            ax = plot_recourse_baseline(
                ax=ax,
                **asdict(curve),
                plot_pdf=plot_pdf,
                bar_width=bar_width,
                at_least=at_least,
                **baseline_kwargs,
            )

        elif component == "observed":
            obs_kwargs = {k: v for k, v in kwargs.items() if k.startswith("obs_")}
            ax = plot_recourse_observed(
                ax=ax,
                **asdict(curve),
                num_models_total=M,
                at_least=at_least,
                plot_pdf=plot_pdf,
                bar_width=bar_width,
                shift=plot_baseline,  # adjust space if also plotting baseline rates
                **obs_kwargs,
            )

    # Configure the plot
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_yticks(np.linspace(0, 1.0, num=num_ticks))
    ax.set_xlabel(xlabel)
    ax.set_xticks(np.linspace(0, 1.0, num=num_ticks))
    ax.set_xlim(-0.01, 1.05)
    formatter = no_leading_0()
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    return ax


def plot_recourse_wrapper(
    ax,
    predictions: pd.DataFrame,
    df: pd.DataFrame,
    count_accepted=True,
    baseline_metric="tpr",
    xlabel: str = "fraction of positive instances",
    ylabel: str = "fraction of positive predictions",
    num_ticks=5,
    bootstrap: bool = False,
    axis: int = 0,
    num_samples: int = None,
    num_resamples: int = 500,
    **plotting_kwargs,
):
    # 1. compute curves
    mask = df["model"].isin(predictions.columns)
    if baseline_metric == "tpr":
        baseline_rates = 1.0 - df[mask]["fnr"].values
    elif baseline_metric == "tnr":
        baseline_rates = list(1.0 - df[mask]["fpr"].values)
    else:
        raise NotImplementedError
    curves = compute_curves_for_recourse_lineplot(
        predictions=predictions.values,
        baseline_rates=baseline_rates,
        count_accepted=count_accepted,
        at_least=False,
        bootstrap=bootstrap,
        axis=axis,
        num_samples=num_samples,
        num_resamples=num_resamples,
    )
    observed = np.column_stack(
        (curves["observed"].fraction_individuals, curves["observed"].fraction_models)
    )
    at_random = np.column_stack(
        (curves["baseline"].fraction_individuals, curves["baseline"].fraction_models)
    )
    # 2. plot
    kwargs = {
        "plot_pdf": True,
        "xlabel": xlabel,
        "ylabel": ylabel,
        "num_ticks": num_ticks,
    }
    M = predictions.shape[1] if predictions is not None else len(baseline_rates)
    if bootstrap and axis == 1:
        M = num_samples
    ax = plot_recourse_lineplot(
        ax=ax,
        curves=curves,
        num_models=M,
        at_least=False,
        **{**kwargs, **plotting_kwargs},
    )
    # sort by model fraction
    return (
        ax,
        observed[np.argsort(observed[:, 1])],
        at_random[np.argsort(at_random[:, 1])],
    )


@dataclass
class RecourseData:
    """Curve object for recourse plots"""

    fraction_models: np.ndarray
    fraction_individuals: np.ndarray
    fill_between: np.ndarray = None


@dataclass
class AgreementData:
    """Curve object for recourse plots"""

    fraction_model_pairs: np.ndarray
    agreement: np.ndarray
    fill_between: np.ndarray = None


# def _agreement_matrix_to_curve(matrix: np.ndarray) -> AgreementData:
#     np.fill_diagonal(matrix, np.nan)
#     agreements = matrix[~np.isnan(matrix)]
#     M_pairs = len(agreements)
#     return AgreementData(
#         agreement=agreements,
#         fraction_model_pairs=np.linspace(0, 1, M_pairs),
#     )


def build_agreement_curve(
    model_data: pd.DataFrame,  # predictions or baseline rates
    pairwise_fun: Callable,
    models: list = None,
) -> np.ndarray:
    assert isinstance(
        model_data, pd.DataFrame
    ), "Provide predictions/baseline rates as pd.DataFrame"
    # restrict to models in predictions
    if models is None:
        filtered_models = sorted(set(model_data.columns), key=get_size_and_it)
    else:
        filtered_models = sorted(
            set([m for m in models if m in model_data.columns]), key=get_size_and_it
        )
    # print(f"filtered_models {filtered_models}")
    # print("these are the columns:", model_data.columns)
    matrix = matrix_pairwise_evals(
        data=model_data[filtered_models],
        fun=pairwise_fun,
    )
    # remove diagnomal because model agreement with itself is 1
    np.fill_diagonal(matrix, np.nan)
    agreements = np.sort(matrix[~np.isnan(matrix)])
    return agreements


def compute_curves_for_agreement_lineplot(
    predictions: pd.DataFrame | None = None,
    baseline_rates: pd.DataFrame | None = None,
    bootstrap: bool = False,
    axis: int = 0,
    num_samples: int = None,
    num_resamples: int = 500,
) -> Dict[AgreementData]:
    assert not (predictions is None and baseline_rates is None), "Provide at least one."
    if predictions is not None and baseline_rates is not None:
        assert set(predictions.columns) == set(
            baseline_rates.columns
        ), "Predictions and baseline models must match."

    curves = {}
    reference_curve = None

    if predictions is not None:
        if predictions.shape[1] == 0:
            print(ValueError("Predictions are empty."))

        if bootstrap:
            agreements = bootstrap_predictions(
                predictions=predictions,
                stat=partial(
                    build_agreement_curve,
                    pairwise_fun=get_observed_pairwise_agreement_rate,
                ),
                axis=axis,
                num_samples=num_samples,
                num_resamples=num_resamples,
            )
            curves["observed"] = AgreementData(
                agreement=agreements.mean,
                fraction_model_pairs=np.linspace(0, 1, len(agreements.mean)),
                fill_between=agreements.standard_error,
            )
        else:
            agreements = build_agreement_curve(
                model_data=predictions,
                pairwise_fun=get_observed_pairwise_agreement_rate,
            )
            curves["observed"] = AgreementData(
                agreement=agreements,
                fraction_model_pairs=np.linspace(0, 1, len(agreements)),
            )

        reference_curve = curves["observed"]

    if baseline_rates is not None:

        def pairwise_agree_fun(evals):
            return pairwise_agreement_at_random(
                acc1=evals.iloc[:, 0].item(), acc2=evals.iloc[:, 1].item()
            )

        if (
            bootstrap and axis == 1
        ):  # no baseline results for specifc groups of individuals available (TODO)
            agreements = bootstrap_predictions(
                predictions=baseline_rates,
                stat=partial(
                    build_agreement_curve,
                    pairwise_fun=pairwise_agree_fun,
                ),
                axis=axis,
                num_samples=(
                    num_samples if axis == 1 else None
                ),  # baseline rates are computed across all individuals, no subsampling possible
                num_resamples=num_resamples if axis == 1 else 1,
            )
            curves["baseline"] = AgreementData(
                agreement=agreements.mean,
                fraction_model_pairs=np.linspace(0, 1, len(agreements.mean)),
                fill_between=agreements.standard_error,
            )
        else:
            agreements = build_agreement_curve(
                model_data=baseline_rates,
                pairwise_fun=pairwise_agree_fun,
            )
            curves["baseline"] = AgreementData(
                agreement=agreements,
                fraction_model_pairs=np.linspace(0, 1, len(agreements)),
            )
        reference_curve = curves["baseline"]

    curves["monoculture"] = AgreementData(
        agreement=np.ones_like(reference_curve.agreement),
        fraction_model_pairs=reference_curve.fraction_model_pairs,
    )
    curves["random_prediction"] = AgreementData(
        agreement=np.full_like(reference_curve.agreement, fill_value=0.5),
        fraction_model_pairs=reference_curve.fraction_model_pairs,
    )
    return curves


def compute_curves_for_recourse_lineplot(
    predictions: np.ndarray | None = None,
    baseline_rates: np.ndarray | None = None,
    count_accepted=True,
    at_least=False,
    bootstrap: bool = False,
    axis: int = 0,
    num_samples: int = None,
    num_resamples: int = 500,
):
    curves = {}
    if predictions is not None:
        if predictions.shape[1] == 0:
            print(ValueError("Predictions are empty."))

        if bootstrap:
            fraction_models_observed, fraction_individuals = bootstrap_predictions(
                predictions=predictions,
                stat=partial(
                    get_model_agreement_histogram,
                    count_acceptances=count_accepted,
                    fraction_models=True,
                    fraction_individuals=True,
                    padding=True,
                ),
                axis=axis,
                num_samples=num_samples,
                num_resamples=num_resamples,
            )
            # compute mean and std
            curves["observed"] = RecourseData(
                fraction_individuals=fraction_individuals.mean,  # mean,
                fraction_models=fraction_models_observed.mean,
                fill_between=fraction_individuals.standard_error,
            )
        else:
            fraction_models_observed, fraction_individuals = (
                get_model_agreement_histogram(
                    predictions=predictions,
                    count_acceptances=count_accepted,
                    fraction_models=True,
                    fraction_individuals=True,
                    padding=False,  # doesn't matter
                )
            )
            curves["observed"] = RecourseData(
                fraction_individuals=fraction_individuals,
                fraction_models=fraction_models_observed,
            )
    if baseline_rates is not None:
        if bootstrap and axis == 1:
            # add dim
            baseline_rates = np.expand_dims(baseline_rates, axis=0)
            fraction_models, fraction_individuals_at_random = bootstrap_predictions(
                predictions=baseline_rates,
                stat=partial(
                    compute_poisson_binom_agreement_distribution,
                    return_success_fractions=True,
                ),
                axis=axis,
                num_samples=(
                    num_samples if axis == 1 else None
                ),  # baseline rates are computed across all individuals, no subsampling possible
                num_resamples=num_resamples if axis == 1 else 1,
            )

            curves["baseline"] = RecourseData(
                fraction_individuals=fraction_individuals_at_random.mean,
                fraction_models=fraction_models.mean,
                fill_between=fraction_individuals_at_random.standard_error,
            )

            fraction_individuals_monoc, fraction_models_monoc = bootstrap_predictions(
                predictions=baseline_rates,
                stat=partial(compute_monoculture_recourse_step, at_least=at_least),
                axis=axis,
                num_samples=(
                    num_samples if axis == 1 else None
                ),  # baseline rates are computed across all individuals, no subsampling possible
                num_resamples=num_resamples if axis == 1 else 1,
            )

            curves["monoculture"] = RecourseData(
                fraction_individuals=fraction_individuals_monoc.mean,
                fraction_models=fraction_models_monoc.mean,
                fill_between=fraction_individuals_monoc.standard_error,
            )
        else:
            fraction_models, fraction_individuals_at_random = (
                compute_poisson_binom_agreement_distribution(
                    baseline_rates=baseline_rates, return_success_fractions=True
                )
            )
            curves["baseline"] = RecourseData(
                fraction_individuals=fraction_individuals_at_random,
                fraction_models=fraction_models,
            )

            x, y = compute_monoculture_recourse_step(
                baseline_rates=baseline_rates, at_least=at_least
            )

            curves["monoculture"] = RecourseData(
                fraction_individuals=x,
                fraction_models=y,
            )
    return curves


def plot_recourse_curves(
    ax,
    label_prefix: str,
    fraction_individuals: np.ndarray,
    fraction_models: np.ndarray,
    num_models_total: int,
    plot_pdf=True,
    fill_between: Tuple[np.ndarray, np.ndarray] = None,
    bar_width: float = 0.1,
    shift: float = 0.0,
    at_least: bool = False,
    **kwargs,
):
    # CDF
    cumulative_fraction_individuals = apply_cumulative(
        prob_distribution=fraction_individuals, at_least=at_least
    )
    # add anchor points
    fraction_models_anchored = np.concat([np.zeros(1), fraction_models, np.ones(1)])
    if at_least:
        cumulative_fraction_individuals = np.concat(
            [np.ones(1), cumulative_fraction_individuals, np.zeros(1)]
        )
    else:

        cumulative_fraction_individuals = np.concat(
            [np.zeros(1), cumulative_fraction_individuals, np.ones(1)]
        )

    # filter kwargs
    prefix_kwargs = {**get_kwargs_subset_from_defaults(label_prefix), **kwargs}
    curve_kwargs = validate_kwargs(plt.step, kwargs=prefix_kwargs, prefix=label_prefix)
    plot_recourse_cdf(
        ax,
        fraction_individuals=cumulative_fraction_individuals,
        fraction_models=fraction_models_anchored,
        fill_between=fill_between,
        **curve_kwargs,
    )

    # PDF
    if plot_pdf:
        pdf_kwargs = validate_kwargs(
            plt.barh, kwargs=prefix_kwargs, prefix=label_prefix
        )

        # if label_prefix not in ["monoculture_", "baseline_"]:
        # bin number of models into evenly spaced bins for pdf (if not already)
        fraction_models, fraction_individuals = bin_fractions(
            num_models_total,
            fractions_models=fraction_models,
            frequencies=fraction_individuals,
        )
        plot_recourse_pdf(
            ax=ax,
            fraction_models=fraction_models,
            fraction_individuals=fraction_individuals,
            width=bar_width,
            shift=shift,
            **pdf_kwargs,
        )

    return ax


def plot_recourse_observed(
    ax: plt.Axes,
    fraction_models: np.ndarray,
    num_models_total: int,
    fraction_individuals: np.ndarray,
    fill_between: Tuple[np.ndarray, np.ndarray] = None,
    plot_pdf=True,
    at_least=False,
    bar_width=0.1,
    shift=False,
    **kwargs,
):
    if at_least:
        raise NotImplementedError("'at least' not currently implemented")
    return plot_recourse_curves(
        ax=ax,
        fraction_individuals=fraction_individuals,
        fraction_models=fraction_models,
        num_models_total=num_models_total,
        label_prefix="obs_",
        fill_between=fill_between,
        plot_pdf=plot_pdf,
        bar_width=bar_width,
        shift=bar_width / 2 if shift else 0.0,
        at_least=at_least,
        **kwargs,
    )


def plot_recourse_baseline(
    ax: plt.Axes,
    fraction_models: np.ndarray,
    fraction_individuals: np.ndarray,
    plot_pdf=True,
    fill_between: Tuple[np.ndarray, np.ndarray] = None,
    bar_width=0.1,
    at_least=True,
    **kwargs,
):
    M = len(fraction_models) - 1  # known, because computed based on baseline_rates
    return plot_recourse_curves(
        ax=ax,
        fraction_individuals=fraction_individuals,
        fraction_models=fraction_models,
        num_models_total=M,
        label_prefix="baseline_",
        fill_between=fill_between,
        plot_pdf=plot_pdf,
        bar_width=bar_width,
        shift=-bar_width / 2,
        at_least=at_least,
        **kwargs,
    )


def plot_recourse_monoculture(
    ax,
    fraction_individuals,
    fraction_models,
    fill_between: Tuple[np.ndarray, np.ndarray] = None,
    **kwargs,
):
    plot_recourse_curves(
        ax=ax,
        fraction_individuals=fraction_individuals,
        fraction_models=fraction_models,
        num_models_total=len(fraction_models),
        fill_between=fill_between,
        label_prefix="monoculture_",
        plot_pdf=False,
        linestyle="dotted",
        zorder=-1,
        **kwargs,
    )


def plot_recourse_pdf(
    ax,
    fraction_models,
    fraction_individuals,
    width,
    shift=0.0,
    alpha=0.7,
    **kwargs,
):
    filtered_kwargs = validate_kwargs(plt.barh, kwargs=kwargs)
    if len(fraction_individuals) <= 2:
        width /= 2
        shift /= 2
    ax.barh(
        y=fraction_models - shift,
        width=fraction_individuals,
        height=width,
        alpha=alpha,
        zorder=0,
        linewidth=0,  # no edges
        **filtered_kwargs,
    )


def plot_recourse_cdf(
    ax,
    fraction_individuals: np.ndarray,
    fraction_models,
    fill_between: Tuple[np.ndarray, np.ndarray] = None,
    zorder=0,
    **kwargs,
):
    filtered_kwargs = validate_kwargs(plt.step, kwargs=kwargs)
    if fill_between is not None:
        # only first anchor point due to post-step (work-around TODO)
        stderr_frac_indivividuals = np.concat(
            [fraction_individuals[:1], fill_between, fraction_individuals[-1:]]
        )

        ls = filtered_kwargs.pop("linestyle", None)
        poly = ax.fill_betweenx(
            y=fraction_models[:-1],
            x1=(fraction_individuals - stderr_frac_indivividuals)[:-1],
            x2=(fraction_individuals + stderr_frac_indivividuals)[:-1],
            step="post",
            alpha=0.25,
            **filtered_kwargs,
            # linewidth=0.0,
            edgecolor="none",  # avoid edge
            zorder=-2,
        )

        # clip outside Rectangle to remove step artefact
        clip_rect = Rectangle(
            (0.0, 0.0),  # lower-left corner
            1.0,  # width
            1.0,  # height
            transform=ax.transData,  # data coordinates
        )
        poly.set_clip_path(clip_rect)
        if ls is not None:
            filtered_kwargs["linestyle"] = ls
    ax.step(
        x=fraction_individuals,
        y=fraction_models,
        zorder=zorder,
        where="pre",
        **filtered_kwargs,
    )


def plot_agreement_lineplot(
    ax: plt.Axes,
    curves: dict[AgreementData],
    sort_by=None,
    xlabel: str = "fraction of model pairs",
    ylabel: str = "agreement",
    title: str = "",
    plot_monoc=True,
    plot_baseline: bool = True,
    plot_scatter=False,
    ylim=(0.45, 1.0),
    num_ticks=5,
    **kwargs,
):
    """
    x individuals are accepted by **at least** y models

    Parameters:
        ax: The matplotlib axis to plot on.
        predictions: A tensor of predictions from the model.
        y_true: True labels (optional).
        baseline_rates: Baseline rejection rates for comparison (optional).
        plot_cumulative: Whether to plot cumulative probabilities (default = False, else True
          to plot from left to right, else 'left' to add up all the way to the left)
        ylabel: Label for the y-axis.
        baseline_label: Label for the baseline comparison.
        relative_x: Normalize x-axis to [0, 1].
        count_accepted: Count positive predictions (True) or rejections (False), default: True,
        indicate_mean: Whether to plot a small triangle below the x-axis to indicate the mean value, default=False.

    Returns:
        ax: The matplotlib axis with the plot.
    """
    # Validate inputs
    assert set(curves.keys()).issubset(
        {"observed", "baseline", "monoculture", "random_prediction"}
    )
    if "observed" not in curves.keys() and "baseline" not in curves.keys():
        raise ValueError("At least one of 'observed' or 'baseline' must be provided.")
    # TODO: sorting when only one is available
    if "observed" in curves.keys() and "baseline" in curves.keys():
        others = [curves["observed"].fill_between, curves["baseline"].fill_between]
        curves["observed"].agreement, curves["baseline"].agreement, others = (
            sort_agreements(
                obs=curves["observed"].agreement,
                exp=curves["baseline"].agreement,
                sort_by=sort_by,
                others=others,
            )
        )
        curves["observed"].fill_between, curves["baseline"].fill_between = others

    if not plot_monoc:
        curves.pop("monoculture", None)
    if not plot_baseline:
        curves.pop("baseline", None)

    # fix plotting order
    for component in ["random_prediction", "monoculture", "baseline", "observed"]:
        curve = curves.get(component, None)
        if curve is None:
            continue
        plot_agreement(
            ax=ax, **asdict(curve), component=component, scatter=plot_scatter, **kwargs
        )

    # Configure the plot
    ax.set_yticks(np.linspace(0, 1.0, num=5))
    ax.set_ylim(bottom=ylim[0], top=ylim[1] + 0.01)
    # ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.set_title(title.replace("ACS", "ACS ").replace("_", " "))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.linspace(0, 1.0, num=num_ticks))
    ax.set_xlim(-0.01, 1.05)
    formatter = no_leading_0()
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    return ax


def plot_agreement_wrapper(
    ax,
    predictions: pd.DataFrame,
    df=pd.DataFrame,
    xlabel="fraction of model pairs",
    ylabel="agreement",
    baseline_metric="accuracy",
    bootstrap: bool = False,
    axis: int = 0,
    num_samples: int = None,
    num_resamples: int = 500,
    sort_by: str | None = None,
    **plotting_kwargs,
):
    # compute agreement
    mask = df["model"].isin(predictions.columns)
    baseline_df = df[mask][["model", baseline_metric]].set_index("model").T
    agreement_data = compute_curves_for_agreement_lineplot(
        predictions=predictions,
        baseline_rates=baseline_df,
        bootstrap=bootstrap,
        axis=axis,
        num_samples=num_samples,
        num_resamples=num_resamples,
    )

    # plot
    ax = plot_agreement_lineplot(
        ax=ax,
        curves=agreement_data,
        sort_by=sort_by,
        ylim=(0.45, 1.0),
        xlabel=xlabel,
        ylabel=ylabel,
        **plotting_kwargs,
    )

    return ax, agreement_data


def plot_agreement(
    ax,
    agreement: np.ndarray,
    fraction_model_pairs: np.ndarray,
    component: str,
    fill_area_to_monoc: bool = True,
    scatter: bool = False,
    fill_between: Tuple[np.ndarray, np.ndarray] = None,
    **kwargs,
):
    """
    Plot agreement curves for different components.

    Args:
        ax: matplotlib axis to plot on.
        agreement: Agreement values.
        fraction_model_pairs: Fraction of model pairs.
        component: One of {"observed", "monoculture", "random_prediction", ...}.
        fill: If True and component=="observed", fill between curve and 1.
        scatter: If True, plot as scatter points instead of curve.
        kwargs: Additional plotting style arguments, prefixed per component.
    """
    # repeat when only one value to plot horizontal line
    if len(agreement) == 1:
        agreement = np.repeat(agreement, 2)
        fraction_model_pairs = np.linspace(0.0, 1.0, len(agreement))

    # build kwargs
    prefix = "obs_" if component == "observed" else component + "_"
    filtered_kwargs = {
        **get_kwargs_subset_from_defaults(prefix),
        **{k: v for k, v in kwargs.items() if k.startswith(prefix)},
    }
    curve_kwargs = validate_kwargs(plt.plot, kwargs=filtered_kwargs, prefix=prefix)

    # add scatter points
    if scatter:
        curve_kwargs.update({"marker": "o", "markersize": 3})

    # special styles by component
    if component == "random_prediction":
        curve_kwargs.update({"linestyle": "dashed", "linewidth": 1, "zorder": -1})

    # plot
    if fill_between is not None:
        ax.fill_between(
            x=fraction_model_pairs,
            y1=agreement - fill_between,
            y2=agreement + fill_between,
            alpha=0.25,
            **curve_kwargs,
        )
    ax.plot(fraction_model_pairs, agreement, **curve_kwargs)

    # for observed data, fill area to monoculture
    if component == "observed" and fill_area_to_monoc:
        fill_kwargs = validate_kwargs(
            plt.fill_between,
            {
                **get_kwargs_subset_from_defaults("fill_"),
                **{
                    k: v
                    for k, v in kwargs.items()
                    if k != "fill_between" and k.startswith("fill_")
                },
            },
            prefix="fill_",
        )
        ax.fill_between(
            fraction_model_pairs,
            agreement,
            np.ones_like(agreement),  # monoculture
            **fill_kwargs,
        )


def plot_agreement_matrix(
    models,
    agreements: np.ndarray,
    title: str = "",
    file_name: str | Path = None,
    figsize=(8, 8),
    vmin=0,
    vmax=1,
    cmap="magma_r",
    norm=None,
    exclude_diagonal=False,
    annotate: bool = False,
):
    if exclude_diagonal:
        mask = np.zeros_like(agreements)
        np.fill_diagonal(mask, val=np.nan)
        agreements += mask
    if norm is None:
        norm = pltcolors.Normalize(vmin=vmin, vmax=vmax)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    num_models = agreements.shape[0]
    assert len(models) == num_models
    im = ax.imshow(
        agreements.numpy(),
        aspect="equal",
        extent=(0, num_models, num_models, 0),  # left, right, bottom, top
        origin="upper",
        cmap=cmap,
        norm=norm,
    )
    # im = ax.pcolormesh(agreements.numpy(), cmap='magma_r', vmin=0, vmax=1)
    ax.set_title(title)
    ax.grid(False)
    ax.set_xticks(
        np.arange(num_models) + 0.5,
        [prettify_model_name(key_to_model(m)) for m in models],
        rotation=90,
        ha="right",  # horizontal alignment (right, center, left)
        # rotation_mode="anchor",
    )
    ax.set_yticks(
        np.arange(num_models) + 0.5,
        [prettify_model_name(key_to_model(m)) for m in models],
    )

    if annotate:
        for y_index, y in enumerate(np.arange(num_models)):
            for x_index, x in enumerate(np.arange(num_models)):
                if y_index != x_index or not exclude_diagonal:
                    if not (
                        agreements[y_index, x_index].item()
                        != agreements[y_index, x_index].item()
                    ):  # not nan
                        label = f"{agreements[y_index, x_index].item():.2f}"
                        text_x = x + 0.5
                        text_y = y + 0.5
                        ax.text(
                            text_x,
                            text_y,
                            label,
                            color="black",
                            ha="center",
                            va="center",
                        )

    plt.colorbar(im, fraction=0.046, pad=0.04)
    if file_name:
        plt.savefig(file_name)
    plt.show()
    # return fig, ax


# ------------------------------------------------------
# Agreement and Recourse for subgroups
# ------------------------------------------------------


def plot_groups(
    what_to_plot: str,
    ax: plt.Axes,
    predictions: pd.DataFrame,
    df: pd.DataFrame,
    data: Tuple[pd.DataFrame, pd.Series],
    groups: Iterable,
    group_map,  # contain feature name and value map per task
    colors: List[str],
    plot_all: bool = True,
    min_group_size: int = None,
    bootstrap: bool = False,
    axis: int = 0,
    num_resamples: int = 500,
):
    # ---- Groups ----
    feature_name = group_map["feature"]  # str
    group_to_member_map = group_map["values"]  # map group to its members

    group_sizes = []
    group_dict = {}
    for g, group in enumerate(groups):
        print(f"- {g} {group:40}", end=" ")
        if group not in group_to_member_map.keys():
            print("-")
            continue
        res = get_group_predictions(
            feature_name=feature_name,
            predictions=predictions,
            data=data,
            group=group,
            members=group_to_member_map[group],
            min_group_size=min_group_size,
        )
        if res is None:
            continue
        (predictions_group, _), _, group_size = res
        group_sizes.append((g, group_size))
        group_dict[group] = (predictions_group, g)

    smallest_group = (
        min(group_sizes, key=lambda x: x[1])[1] if len(group_sizes) > 0 else None
    )

    if plot_all:
        # ---- All models ----
        mask = df["model"].isin(predictions.columns)
        color_all = "C7"
        common_args = dict(
            ax=ax,
            predictions=predictions,
            df=df[mask],
            xlabel="",
            ylabel="",
            plot_monoc=False,
            obs_color=color_all,
            obs_label="all",
            baseline_color=color_all,
            baseline_linestyle=densely_dotted,
            bootstrap=bootstrap,
            axis=axis,
            num_samples=smallest_group,
            num_resamples=4 * num_resamples,
        )
        if what_to_plot == "agreement":
            ax, agree_data = plot_agreement_wrapper(
                **common_args, fill_area_to_monoc=False, fill_alpha=0.1
            )
            print(
                f"all: mean agreement: {agree_data['observed'].agreement.mean():.4f}, "
                f"mean agreement at random: {agree_data['baseline'].agreement.mean():.4f}"
            )
        else:
            ax, recourse_obs, recourse_at_random = plot_recourse_wrapper(
                **common_args, plot_pdf=False
            )
            # print(
            #     f"all: mean recourse level {(recourse_obs[:, 0] * recourse_obs[:, 1]).mean():.4f}"
            # )
            # print(
            #     f"mean recourse level at random {(recourse_at_random[:, 0] * recourse_at_random[:, 1]).mean():.4f}"
            # )

    for group, (predictions_group, g) in group_dict.items():
        # Plot
        mask = df["model"].isin(predictions_group.columns)
        common_args = dict(
            ax=ax,
            predictions=predictions_group,
            df=df[mask],
            xlabel="",
            ylabel="",
            plot_monoc=False,
            obs_color=colors[g],
            obs_label=group,
            baseline_color=colors[g],
            baseline_linestyle=densely_dotted,
            bootstrap=bootstrap,
            axis=axis,
            num_samples=smallest_group,
            num_resamples=num_resamples,
        )
        if what_to_plot == "agreement":
            ax, agree_data = plot_agreement_wrapper(
                **common_args,
                fill_area_to_monoc=False,
                fill_alpha=0.1,
                obs_linewidth=1.2,
            )
            print(
                f"{group}: mean agreement: {agree_data['observed'].agreement.mean():.4f}, "
                f"mean agreement at random: {agree_data['baseline'].agreement.mean():.4f}"
            )
        elif what_to_plot == "recourse":
            ax, recourse_obs, recourse_at_random = plot_recourse_wrapper(
                **common_args, plot_pdf=False, plot_baseline=False
            )
            # print(
            #     f"{group}: mean recourse level {(recourse_obs[:, 0] * recourse_obs[:, 1]).mean():.4f}"
            # )
            # print(
            #     f"mean recourse level at random {(recourse_at_random[:, 0] * recourse_at_random[:, 1]).mean():.4f}"
            # )
        else:
            print("not implemented")
    ax = add_mini_legend_with_group_info(
        ax=ax,
        colors=colors,
        group_sizes=(
            group_sizes
            if axis == 1
            else [(g, gs / predictions.shape[0]) for (g, gs) in group_sizes]
        ),
        bbox_to_anchor=(1.03 if axis == 1 else 1.14, 0.5),
    )
    return ax


def get_group_predictions(
    predictions: pd.DataFrame,
    data: Tuple[pd.DataFrame, pd.Series],
    feature_name: str,
    group: str,
    members: List,
    min_group_size: Union[int, float] = None,
):
    if feature_name == "model":
        # ensure to only account for memeber that are present in predictions
        members = [m for m in members if m in predictions.columns]
        num_members = len(members)
        if num_members == 1 or (min_group_size and num_members < min_group_size):
            print(f"{num_members} model(s) - skipping")
            return None
        print(f"{num_members} models")
        return (
            filter_predictions_and_data(
                predictions=predictions, data=data, restrict_models=members
            ),
            members,
            num_members,
        )

    else:
        predictions_group, data_group = filter_predictions_and_data(
            predictions=predictions,
            data=data,
            feature=feature_name,
            feature_val=members,
        )
        size = predictions_group.shape[0]
        total = predictions.shape[0]
        if min_group_size and size < int(min_group_size * total):
            print(f"too small ({size}<{int(min_group_size*total)}) - skipping")
            return None
        print(f"{size:8} individuals ({size/total:.2f})")
        return (predictions_group, data_group), members, size


def plot_for_groups_all_tasks(
    what_to_plot: str,
    tasks: List[str],
    predictions: Dict[str, pd.DataFrame],
    data: Dict[Tuple[pd.Series, pd.DataFrame]],
    df: pd.DataFrame,
    group_map: dict,  # contain feature name and value map per task
    colors: List[str],
    plot_all: bool = True,
    bootstrap: bool = False,
    axis: int = 0,
    num_resamples: int = 500,
    nrows: int = 2,
    fig_width: float = plt.rcParams["figure.figsize"][0],
    fig_height: float = plt.rcParams["figure.figsize"][1],
    ylabel: str = "",
    xlabel: str = "",
    min_group_size: Union[int, float] = None,
    file_name: Path = None,
    legend_yoffset=0.1,
    legend_columns: bool = True,
    legend_loc="lower center",
    legend_ncol=None,
    place_legend_in_ax: bool = False,
    legend_xshift=0.9,
):

    # Figure Setup
    ncols = (len(tasks) + 1) // nrows
    fig, axs = plt.subplots(
        nrows,
        ncols,
        figsize=(fig_width, fig_height),
        sharey="row",
        constrained_layout=True,
    )

    # Collect all unique group labels across tasks for consistent plotting order
    groups = sorted(
        list(
            {g for task_info in group_map.values() for g in task_info["values"].keys()}
        )
    )

    axs_flat = axs.flat
    for i, ax in enumerate(axs_flat):
        if i >= len(tasks):
            if not place_legend_in_ax:
                ax.set_axis_off()
            continue

        task = tasks[i]
        print(f"\n{task:15} {predictions[task].shape[0]:8} individuals ")
        ax = plot_groups(
            what_to_plot=what_to_plot,
            ax=ax,
            predictions=predictions[task],
            df=df[df["task"] == task],
            data=data[task],
            groups=groups,
            group_map=group_map[task],
            plot_all=plot_all,
            colors=colors,
            min_group_size=min_group_size,
            bootstrap=bootstrap,
            axis=axis,
            num_resamples=num_resamples,
        )
        ax.set_title(task.replace("ACS", "").replace("BRFSS", "").replace("_", " "))

    # Configure figure
    for r in range(nrows):
        axs[r, 0].set_ylabel(ylabel)
        axs[r, 0].set_xlabel(xlabel)

    labels, handles = zip(*unique_legend_items(axs).items())
    if "random error" in labels:
        idx = labels.index("random error")
        handles = list(handles)
        handles[idx] = create_proxy_handle(
            orig_handle=handles[idx],
            update_properties={"color": "C7", "linestyle": densely_dotted},
        )
        print(handles[idx].get_linestyle())

    if len(tasks) < nrows * ncols and place_legend_in_ax:
        legend_ax = axs_flat[-1]
        bbox = legend_ax.get_position()
        x_center = legend_xshift * (bbox.x0 + bbox.x1) / 2
        y_center = (bbox.y0 + bbox.y1) / 2

        configure_legend(
            fig=fig,
            axs=axs,
            add_monoculture_mixed_handle=False,
            loc=legend_loc,
            columns=False,
            bbox_to_anchor=(x_center, y_center),
            handles_labels=(handles, labels),
        )
        axs.flat[-1].axis("off")
    else:
        configure_legend(
            fig=fig,
            axs=axs,
            add_monoculture_mixed_handle=False,
            columns=legend_columns,
            order=[],
            offset=legend_yoffset,
            **(dict(ncol=legend_ncol or len(labels) // 2) if legend_columns else {}),
            loc=legend_loc,
            handles_labels=(handles, labels),
        )

    if file_name:
        print(f"Save as {file_name}")
        for ending in [".png", ".pdf"]:
            assert file_name.parent.exists()
            plt.savefig(f"{file_name.as_posix()}{ending}")

    plt.show()


def minimal_example_binomial(color_by_recourse=False, save_path=None):
    num = 5
    cut_off = (num + 1) // 2
    if color_by_recourse:
        colors = (
            ["forestgreen"]
            + (cut_off - 1) * ["yellowgreen"]
            + (cut_off - 1) * ["orange"]
            + ["firebrick"]
        )
    else:
        colors = "C0"
    probs = np.array(
        [sp.special.binom(num, i) * 2 ** (-num) for i in range(0, num + 1)]
    )
    fig, ax = plt.subplots()
    bars = ax.bar(np.arange(num + 1), probs, color=colors)
    ax.bar_label(bars, fmt="%.3f", label_type="edge", size=8, padding=2)
    ax.set_xticks(np.arange(num + 1))

    if save_path:
        plt.savefig(save_path)
    plt.show()
    return probs[0], probs[:cut_off].sum(), probs[cut_off:-1].sum(), probs[-1]
