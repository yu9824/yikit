"""Visualization utilities for regression model evaluation.

This module provides a `yyplot` function, which draws a scatter plot of
true vs. predicted values (sometimes called a *y-y plot*).  It can handle
one or multiple pairs of ``(y_true, y_pred)`` sequences (e.g., train/test
or train/validation/test) and annotates the figure with common regression
metrics such as :math:`R^2`, RMSE, MAE, and MSE.
"""

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
    """Plot true vs. predicted values for one or more data sets.

    This function draws a scatter plot of multiple ``(y_true, y_pred)`` pairs
    and overlays the identity line :math:`y = x` as a reference.  It computes
    summary regression metrics for each pair and displays them as text in the
    top-left of the plot.  Typical usage is to compare model performance on
    different splits such as train, validation, and test.

    Parameters
    ----------
    *y_data : ArrayLike
        A sequence of arrays interpreted as consecutive pairs
        ``(y_true_0, y_pred_0, y_true_1, y_pred_1, ...)``.  The length of
        ``y_data`` must be even, and each pair must be broadcastable to the
        same shape.
    labels : Sequence of str or None, optional
        Labels for each ``(y_true, y_pred)`` pair, used in the legend and
        metric annotations.  If ``None`` (default), labels are automatically
        set to ``("train", "test")`` for two data sets, to
        ``("train", "val", "test")`` for three data sets, or to all-``None``
        for other numbers of data sets.
    metrics : Sequence of {"r2", "rmse", "mae", "mse"}, optional
        Metrics to compute for each data set.  Each metric is shown in the
        annotation box together with the corresponding label (if provided).
        Defaults to ``("r2", "rmse")``.
    ax : matplotlib.axes.Axes, optional
        Existing Axes on which to draw the plot.  If ``None``, a new
        figure and axes are created.
    alpha : float, optional
        Relative margin added to the data range when determining plot limits.
        The default is ``0.05`` (5% margin on each side).

    Returns
    -------
    matplotlib.axes.Axes
        The axes object with the scatter plot and annotations.

    Raises
    ------
    ValueError
        If no data is provided, or if the number of positional arguments
        is odd (i.e., there is an unmatched ``y_true`` or ``y_pred``), or
        if the length of ``labels`` does not match the number of data sets,
        or if an unknown metric name is given in ``metrics``.

    Examples
    --------
    Plot a single data set:

    >>> import numpy as np
    >>> from yikit.visualize import yyplot
    >>> y_true = np.array([1.0, 2.0, 3.0])
    >>> y_pred = np.array([0.8, 2.1, 2.9])
    >>> ax = yyplot(y_true, y_pred)

    Plot train and test data sets:

    >>> y_train, y_pred_train = np.arange(5), np.arange(5) + 0.1
    >>> y_test, y_pred_test = np.arange(5), np.arange(5) - 0.2
    >>> ax = yyplot(y_train, y_pred_train, y_test, y_pred_test,
    ...             labels=("train", "test"))
    """
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
