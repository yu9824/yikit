"""Visualization utilities for regression model evaluation.

This module provides a `yyplot` function, which draws a scatter plot of
true vs. predicted values (sometimes called a *y-y plot*).  It can handle
one or multiple pairs of ``(y_true, y_pred)`` sequences (e.g., train/test
or train/validation/test) and annotates the figure with common regression
metrics such as :math:`R^2`, RMSE, MAE, and MSE.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import reduce
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, cast, overload

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

if TYPE_CHECKING:
    # ``NDArray`` is missing from numpy < 1.21, so import it for typing only.
    from numpy.typing import NDArray


def _rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return the root mean squared error of the predictions.

    The value equals scikit-learn's RMSE with its default
    ``multioutput="uniform_average"``: for multi-output targets it is the
    mean of the RMSE of each output, not the root of the overall MSE.
    scikit-learn's own RMSE function needs scikit-learn >= 1.4 and the
    ``squared=False`` option was removed later, so the square root is taken
    here (works with scikit-learn >= 0.24.1).
    """
    mse_of_each_output = mean_squared_error(
        y_true, y_pred, multioutput="raw_values"
    )
    return float(np.mean(np.sqrt(mse_of_each_output)))


METRIC_INFO_MAP = MappingProxyType(
    {
        "r2": {"fmt": "#.2f", "label": "$R^2$", "func": r2_score},
        "rmse": {"fmt": "#.3g", "label": "RMSE", "func": _rmse},
        "mae": {"fmt": "#.3g", "label": "MAE", "func": mean_absolute_error},
        "mse": {"fmt": "#.3g", "label": "MSE", "func": mean_squared_error},
    }
)


@overload
def yyplot(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    labels: Sequence[str | None] | None = None,
    metrics: Sequence[Literal["r2", "rmse", "mae", "mse"]] = (
        "r2",
        "rmse",
    ),
    ax: matplotlib.axes.Axes | None = None,
    alpha: float = 0.05,
) -> matplotlib.axes.Axes: ...


@overload
def yyplot(
    y_train: ArrayLike,
    y_pred_on_train: ArrayLike,
    y_test: ArrayLike,
    y_pred_on_test: ArrayLike,
    *,
    labels: Sequence[str] | None = ("train", "test"),
    metrics: Sequence[Literal["r2", "rmse", "mae", "mse"]] = (
        "r2",
        "rmse",
    ),
    ax: matplotlib.axes.Axes | None = None,
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
    labels: Sequence[str | None] | None = ("train", "val", "test"),
    metrics: Sequence[Literal["r2", "rmse", "mae", "mse"]] = (
        "r2",
        "rmse",
    ),
    ax: matplotlib.axes.Axes | None = None,
    alpha: float = 0.05,
) -> matplotlib.axes.Axes: ...


@overload
def yyplot(
    *y_data: ArrayLike,
    labels: Sequence[str | None] | None = None,
    metrics: Sequence[Literal["r2", "rmse", "mae", "mse"]] = (
        "r2",
        "rmse",
    ),
    ax: matplotlib.axes.Axes | None = None,
    alpha: float = 0.05,
) -> matplotlib.axes.Axes: ...


def yyplot(  # type: ignore[misc]
    *y_data: ArrayLike,
    labels: Sequence[str | None] | None = None,
    metrics: Sequence[Literal["r2", "rmse", "mae", "mse"]] = (
        "r2",
        "rmse",
    ),
    ax: matplotlib.axes.Axes | None = None,
    alpha: float = 0.05,
) -> matplotlib.axes.Axes:
    """Plot predicted against true values for one or more data sets.

    Each ``(y_true, y_pred)`` pair is drawn as a scatter plot over the
    identity line :math:`y = x`, and the chosen regression metrics of each
    pair are written in the top-left corner.  Typical use is to compare a
    model on its train, validation and test sets.

    Parameters
    ----------
    *y_data : array-like
        Consecutive pairs ``(y_true_0, y_pred_0, y_true_1, y_pred_1, ...)``
        of true and predicted values.  At least one pair is required, and
        the two arrays of a pair must have the same length.
    labels : sequence of str or None, optional
        Name of each pair, shown in the legend and next to its metrics
        exactly as written: characters such as ``$``, ``_``, ``^`` and
        ``%`` are not read as mathtext.  A pair labeled None or ``""`` has
        no legend entry, and its metrics carry no label.  If None
        (default), two pairs are labeled ``("train", "test")``, three pairs
        ``("train", "val", "test")``, and any other number of pairs is not
        labeled.
    metrics : sequence of {"r2", "rmse", "mae", "mse"}, default=("r2", "rmse")
        Metrics written for each pair, in this order: the coefficient of
        determination, the root mean squared error, the mean absolute error
        and the mean squared error.  For multi-output targets, RMSE is the
        mean of the RMSE of each output, as in scikit-learn.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.  Only this axes changes: the layout of its figure
        (such as the subplot parameters) is left as it is.  If None
        (default), a new 3.6 x 3.6 inch figure is created and laid out with
        ``tight_layout``.
    alpha : float, default=0.05
        Margin added on each side of the range of all the values, as a
        fraction of that range, to set the limits of both axes.

    Returns
    -------
    matplotlib.axes.Axes
        The axes that was drawn on.

    Raises
    ------
    ValueError
        If no pair or an odd number of arrays is given, if the length of
        ``labels`` differs from the number of pairs, or if ``metrics``
        contains an unknown name.  These are checked before anything is
        drawn, so no figure is created and ``ax`` is left untouched.

    Notes
    -----
    The legend shows the scatter plots labeled here together with the
    labeled artists already on ``ax``.  Labels are drawn as written unless
    ``rcParams["text.usetex"]`` is enabled, in which case LaTeX reads them.

    Examples
    --------
    Plot a single data set:

    >>> import numpy as np
    >>> from yikit.visualize import yyplot
    >>> y_true = np.array([1.0, 2.0, 3.0])
    >>> y_pred = np.array([0.8, 2.1, 2.9])
    >>> ax = yyplot(y_true, y_pred)

    Plot train and test data sets with the mean absolute error as well:

    >>> y_train, y_pred_train = np.arange(5), np.arange(5) + 0.1
    >>> y_test, y_pred_test = np.arange(5), np.arange(5) - 0.2
    >>> ax = yyplot(y_train, y_pred_train, y_test, y_pred_test,
    ...             labels=("train", "test"), metrics=("r2", "rmse", "mae"))

    Draw on one axes of an existing figure, whose layout is kept:

    >>> import matplotlib.pyplot as plt
    >>> fig, axes = plt.subplots(1, 2)
    >>> ax = yyplot(y_true, y_pred, labels=["fold_1 (5% noise)"], ax=axes[0])
    >>> ax is axes[0]
    True
    >>> plt.close("all")
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
    elif len(labels) != n_data_species:
        raise ValueError(
            f"Length of labels ({len(labels)}) does not match number of "
            f"data sets ({n_data_species})."
        )

    unknown_metrics = [
        metric for metric in metrics if metric not in METRIC_INFO_MAP
    ]
    if unknown_metrics:
        raise ValueError(
            f"Unknown metric(s) {unknown_metrics}; "
            f"choose from {list(METRIC_INFO_MAP)}."
        )

    # Everything is computed before drawing, so that an error leaves no
    # half-drawn figure behind.
    pairs = [
        (y_data[2 * i_data_species], y_data[2 * i_data_species + 1])
        for i_data_species in range(n_data_species)
    ]
    # An empty label means no label, as it does for matplotlib's legend.
    shown_labels = [
        None if label is None or label == "" else _escape_dollar(str(label))
        for label in labels
    ]
    list_score_texts = [
        _format_metric(metric, shown_label, x, y)
        for (x, y), shown_label in zip(pairs, shown_labels)
        for metric in metrics
    ]

    _data_min = float("inf")
    _data_max = -float("inf")
    for x, y in pairs:
        # ``ArrayLike`` also admits str and bytes, which numpy's stubs reject
        # in ``np.min``/``np.max``. The plotted values are numeric.
        x_values = cast("NDArray[Any] | Sequence[float]", x)
        y_values = cast("NDArray[Any] | Sequence[float]", y)
        _data_min = reduce(
            min, (_data_min, np.min(x_values), np.min(y_values))
        )
        _data_max = reduce(
            max, (_data_max, np.max(x_values), np.max(y_values))
        )

    # set plot limits
    datalim = (_data_min, _data_max)
    offset = (_data_max - _data_min) * alpha
    plotlim = (datalim[0] - offset, datalim[1] + offset)

    new_figure: matplotlib.figure.Figure | None = None
    if ax is None:
        new_figure, ax = plt.subplots(1, 1, figsize=(3.6, 3.6), dpi=144)

    # The legend lists the labeled artists already on ``ax`` and the scatter
    # plots labeled here.
    legend_handles, legend_labels = ax.get_legend_handles_labels()
    for (x, y), shown_label in zip(pairs, shown_labels):
        if shown_label is None:
            ax.scatter(x, y)
        else:
            legend_handles.append(ax.scatter(x, y, label=shown_label))
            legend_labels.append(shown_label)

    # plot y = x reference line
    ax.plot(plotlim, plotlim, color="gray", zorder=0.5)

    ax.set_xlabel("$y_\\mathrm{true}$")
    ax.set_ylabel("$y_\\mathrm{pred}$")

    ax.set_xlim(plotlim)
    ax.set_ylim(plotlim)
    ax.set_aspect("equal")

    if any(shown_label is not None for shown_label in shown_labels):
        # Matplotlib < 3.10 leaves out of the legend any label starting with
        # "_", even one given explicitly, so the legend is made with
        # placeholders and its texts are set afterwards.
        legend = ax.legend(
            legend_handles, [" "] * len(legend_handles), loc="lower right"
        )
        for legend_text, legend_label in zip(
            legend.get_texts(), legend_labels
        ):
            legend_text.set_text(legend_label)

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

    # Only a figure created here is laid out; the figure of a given ``ax``
    # keeps its layout.
    if new_figure is not None:
        new_figure.tight_layout()
    return ax


def _escape_dollar(text: str) -> str:
    r"""Return ``text`` with each ``$`` escaped as ``\$``.

    Matplotlib draws a line with an even number of unescaped dollar signs
    as mathtext.  Outside the math parts, ``\$`` is the only special
    sequence and is drawn as ``$``, so the escaped text, kept outside math,
    is drawn exactly as written (``_``, ``^``, ``%`` and backslashes too).
    """
    return text.replace("$", r"\$")


def _format_metric(
    metric: str, label: str | None, y_true: ArrayLike, y_pred: ArrayLike
) -> str:
    """Return the annotation line of one metric of one data set.

    ``label`` must already be escaped with :func:`_escape_dollar`.  It is
    put in parentheses outside math, so it is drawn as written (the closing
    parenthesis also keeps a trailing backslash of the label from escaping
    the ``$`` that opens the value).  Only the value and the name ``$R^2$``
    are mathtext.
    """
    metric_info = METRIC_INFO_MAP[metric]
    metric_value = metric_info["func"](y_true, y_pred)
    name = (
        metric_info["label"]
        if label is None
        else f"{metric_info['label']} ({label})"
    )
    return f"{name}$ = {metric_value:{metric_info['fmt']}}$"
