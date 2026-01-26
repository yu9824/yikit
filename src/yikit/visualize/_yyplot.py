import sys
from functools import reduce
from types import MappingProxyType
from typing import Optional, overload

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from yikit.metrics import root_mean_squared_error

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

if sys.version_info >= (3, 9):
    from collections.abc import Sequence
else:
    from typing import Sequence

METRIC_INFO_MAP = MappingProxyType(
    {
        "r2": {"fmt": "#.2f", "label": "$R^2$", "func": r2_score},
        "rmse": {
            "fmt": "#.3g",
            "label": "RMSE",
            "func": root_mean_squared_error,
        },
        "mae": {"fmt": "#.3g", "label": "MAE", "func": mean_absolute_error},
        "mse": {"fmt": "#.3g", "label": "MSE", "func": mean_squared_error},
    }
)


@overload
def yyplot(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    labels: Optional[Sequence[Optional[str]]] = None,
    metrics: Sequence[Literal["r2", "rmse", "mae", "mse"]] = (
        "r2",
        "rmse",
    ),
    ax: Optional[matplotlib.axes.Axes] = None,
    alpha: float = 0.05,
) -> matplotlib.axes.Axes: ...


@overload
def yyplot(
    y_train: ArrayLike,
    y_pred_on_train: ArrayLike,
    y_test: ArrayLike,
    y_pred_on_test: ArrayLike,
    *,
    labels: Optional[Sequence[str]] = ("train", "test"),
    metrics: Sequence[Literal["r2", "rmse", "mae", "mse"]] = (
        "r2",
        "rmse",
    ),
    ax: Optional[matplotlib.axes.Axes] = None,
    alpha: float = 0.05,
) -> matplotlib.axes.Axes: ...


@overload
def yyplot(
    y_train: ArrayLike,
    y_pred_on_train: ArrayLike,
    y_val: ArrayLike,
    y_pred_on_val: ArrayLike,
    y_test: ArrayLike,
    y_pred_on_test: ArrayLike,
    *,
    labels: Optional[Sequence[Optional[str]]] = ("train", "val", "test"),
    metrics: Sequence[Literal["r2", "rmse", "mae", "mse"]] = (
        "r2",
        "rmse",
    ),
    ax: Optional[matplotlib.axes.Axes] = None,
    alpha: float = 0.05,
) -> matplotlib.axes.Axes: ...


@overload
def yyplot(
    *y_data: ArrayLike,
    labels: Optional[Sequence[Optional[str]]] = None,
    metrics: Sequence[Literal["r2", "rmse", "mae", "mse"]] = (
        "r2",
        "rmse",
    ),
    ax: Optional[matplotlib.axes.Axes] = None,
    alpha: float = 0.05,
) -> matplotlib.axes.Axes: ...


def yyplot(  # type: ignore[misc]
    *y_data: ArrayLike,
    labels: Optional[Sequence[Optional[str]]] = None,
    metrics: Sequence[Literal["r2", "rmse", "mae", "mse"]] = (
        "r2",
        "rmse",
    ),
    ax: Optional[matplotlib.axes.Axes] = None,
    alpha: float = 0.05,
) -> matplotlib.axes.Axes:
    if len(y_data) == 0:
        raise ValueError(
            "At least one pair of data (y_true, y_pred) must be provided."
        )
    if len(y_data) % 2 != 0:
        raise ValueError(
            "y_data must be a list of even length (pairs of (y_true, y_pred))."
        )

    n_data_species = len(y_data) // 2

    if labels is None:
        if n_data_species == 2:
            labels = ("train", "test")
        elif n_data_species == 3:
            labels = ("train", "val", "test")
        else:
            labels = tuple([None] * n_data_species)
    else:
        if len(labels) != n_data_species:
            raise ValueError(
                f"Length of labels ({len(labels)}) does not match number of data sets ({n_data_species})."
            )

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(3.6, 3.6), dpi=144)
    else:
        fig = ax.figure  # type: ignore[assignment]

    _data_min = float("inf")
    _data_max = -float("inf")

    list_score_texts = []
    for i_data_species in range(n_data_species):
        x = y_data[2 * i_data_species]
        y = y_data[2 * i_data_species + 1]

        label = labels[i_data_species] if labels is not None else None

        if label is None:
            ax.scatter(x, y)
        else:
            ax.scatter(x, y, label=label)

        for metric in metrics:
            if metric not in METRIC_INFO_MAP:
                raise ValueError(f"Metric '{metric}' not recognized.")
            metric_info = METRIC_INFO_MAP[metric]
            metric_value = metric_info["func"](x, y)

            if label is None:
                suffix = ""
            else:
                suffix = f"$_\\mathrm{{{label}}}$"

            list_score_texts.append(
                f"{metric_info['label']}{suffix}$ = {metric_value:{metric_info['fmt']}}$"
            )

        _data_min = reduce(min, (_data_min, np.min(x), np.min(y)))
        _data_max = reduce(max, (_data_max, np.max(x), np.max(y)))

    # set plot limits
    datalim = (_data_min, _data_max)
    offset = (_data_max - _data_min) * alpha
    plotlim = (datalim[0] - offset, datalim[1] + offset)

    # plot y = x reference line
    ax.plot(plotlim, plotlim, color="gray", zorder=0.5)

    ax.set_xlabel("$y_\\mathrm{true}$")
    ax.set_ylabel("$y_\\mathrm{pred}$")

    ax.set_xlim(plotlim)
    ax.set_ylim(plotlim)
    ax.set_aspect("equal")

    if labels is not None and any(label is not None for label in labels):
        ax.legend(loc="lower right")

    ax.text(
        datalim[0],
        datalim[1],
        "\n".join(list_score_texts),
        ha="left",
        va="top",
        bbox=dict(
            facecolor="white",
            alpha=0.5,
            edgecolor="none",
        ),
    )

    fig.tight_layout()
    return ax
