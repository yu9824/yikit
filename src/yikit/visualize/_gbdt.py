"""Visualization utilities for NGBoost and gradient boosting models.

This module provides visualization functions for NGBoost models including
distribution plots and learning curves for gradient boosting models.
"""

from __future__ import annotations

import warnings
from decimal import Decimal
from math import ceil
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import check_array

from yikit.helpers import is_installed
from yikit.visualize._utils import (
    COLORS,
    set_font,
    with_custom_matplotlib_settings,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

if is_installed("tqdm"):
    from tqdm.auto import tqdm
else:
    from yikit.helpers import dummy_tqdm as tqdm

# The classes of the optional packages. A tuple stays empty when its package
# is missing, so that ``isinstance`` matches nothing (not even None).
_LIGHTGBM_MODELS: tuple[type[Any], ...] = ()
_NGBOOST_MODELS: tuple[type[Any], ...] = ()
_NGBOOST_DISTRIBUTIONS: tuple[type[Any], ...] = ()

if is_installed("lightgbm"):
    from lightgbm import LGBMModel  # type: ignore[reportMissingImports]

    _LIGHTGBM_MODELS = (LGBMModel,)

if is_installed("ngboost"):
    from ngboost import NGBoost  # type: ignore[reportMissingImports]
    from ngboost.distns import Distn  # type: ignore[reportMissingImports]

    _NGBOOST_MODELS = (NGBoost,)
    _NGBOOST_DISTRIBUTIONS = (Distn,)

#: Number of points of the grid on which the densities are drawn.
_N_GRID_POINTS = 200
#: Margin of the grid on each side of the predicted means, as a fraction of
#: their range.
_MARGIN_RATIO = 0.05
#: Margin of the grid on each side of the predicted means when they span no
#: range, in standard deviations of the widest distribution.
_MARGIN_STDS = 3.0
#: Margin of the grid on each side of the predicted means when they span no
#: range and no distribution has a finite, positive standard deviation.
_FALLBACK_MARGIN = 1.0


@with_custom_matplotlib_settings()
def get_dist_figure(
    y_dist,
    y_true=None,
    keep_y_range: bool = True,
    return_axis: bool = False,
    verbose: bool = True,
    titles: Sequence[object] | None = None,
    fontfamily: str | None = None,
):
    """Draw the predicted distribution of each sample in its own subplot.

    Each subplot shows the probability density of the distribution of one
    sample, a vertical line at its mean ("pred") and, when ``y_true`` is
    given, a vertical line at the true value ("ground truth"). The
    subplots fill a grid of ``ceil(sqrt(n_samples))`` rows row by row, and
    the unused subplots at the end are hidden. Every density is drawn on
    the same 200 points, spanning the predicted means with a margin of 5%
    of their range on each side. When the means span no range (such as for
    one sample), the margin is three times the largest finite, positive
    standard deviation of the distributions instead, or 1.0 if no
    distribution has one (such as Student's t with ``df <= 2``, whose
    standard deviation is infinite). The global font settings of matplotlib
    (``rcParams["font.family"]`` and ``rcParams["font.size"]``) are changed
    while the figure is drawn and restored afterwards.

    Parameters
    ----------
    y_dist : ngboost.distns.Distn
        The predicted distributions of the samples, such as
        ``NGBRegressor.pred_dist(X)``. It needs the methods ``pdf``,
        ``mean`` and ``std``, as continuous regression distributions such
        as ``Normal`` have (``Poisson`` has no ``pdf``, and the
        distributions of classifiers have no ``mean``).
    y_true : array-like of shape (n_samples,) or None, default=None
        The true values of the samples. None draws no true values.
    keep_y_range : bool, default=True
        Whether every subplot has the same top of the y axis: 1.05 times
        the highest density of all the samples. If False, each subplot
        scales its own y axis.
    return_axis : bool, default=False
        Whether to return the grid of subplots along with the figure.
    verbose : bool, default=True
        Whether to show a progress bar while drawing.
    titles : sequence of length n_samples or None, default=None
        The titles of the subplots, one for each sample. None, or an empty
        sequence, titles them "idx: 0", "idx: 1", and so on.
    fontfamily : str or None, default=None
        The font family of the figure. None means "Helvetica" on macOS and
        "DejaVu Sans" on other platforms (see
        :func:`yikit.visualize.set_font`).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure. Returned alone when ``return_axis`` is False.
    axes : numpy.ndarray of matplotlib.axes.Axes of shape (n_rows, n_cols)
        The grid of subplots, two-dimensional even when it has one row or
        one column. Returned as ``(fig, axes)`` only when ``return_axis``
        is True.

    Raises
    ------
    TypeError
        If ``y_dist`` is not a distribution of ngboost.
    ValueError
        If ``titles`` is not empty and its length differs from the number
        of samples.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from ngboost.distns import Normal
    >>> from yikit.visualize import get_dist_figure

    ngboost's ``Normal`` takes the rows (loc, log(scale)), here the means
    10, 25 and 40 with the standard deviations 4, 6 and 5.

    >>> params = np.array([[10.0, 25.0, 40.0], np.log([4.0, 6.0, 5.0])])
    >>> y_dist = Normal(params)
    >>> fig, axes = get_dist_figure(
    ...     y_dist,
    ...     y_true=[12.0, 20.0, 43.0],
    ...     verbose=False,
    ...     fontfamily="DejaVu Sans",
    ...     return_axis=True,
    ... )
    >>> axes.shape
    (2, 2)
    >>> [ax.get_title() for ax in axes.flat]
    ['idx: 0', 'idx: 1', 'idx: 2', '']
    >>> plt.close(fig)
    """
    if not isinstance(y_dist, _NGBOOST_DISTRIBUTIONS):
        raise TypeError(
            "y_dist must be a distribution of ngboost (ngboost.distns.Distn)"
            ", such as NGBRegressor.pred_dist(X), but got "
            f"{type(y_dist).__name__}."
        )
    y_pred = check_array(y_dist.mean(), ensure_2d=False)
    n_samples = len(y_pred)
    if y_true is not None:
        y_true = check_array(y_true, ensure_2d=False)
    if titles is None or len(titles) == 0:
        subplot_titles = [f"idx: {idx}" for idx in range(n_samples)]
    elif len(titles) == n_samples:
        subplot_titles = [f"{title}" for title in titles]
    else:
        raise ValueError(
            f"titles must have one title for each of the {n_samples} "
            f"samples, but got {len(titles)} titles."
        )

    low, high = np.min(y_pred), np.max(y_pred)
    if high > low:
        margin = (high - low) * _MARGIN_RATIO
    else:
        # Some standard deviations are infinite or undefined, such as those
        # of Student's t with df <= 2; they would make the grid NaN.
        stds = np.asarray(y_dist.std(), dtype=float)
        usable_stds = stds[np.isfinite(stds) & (stds > 0)]
        margin = (
            _MARGIN_STDS * np.max(usable_stds)
            if usable_stds.size > 0
            else _FALLBACK_MARGIN
        )
    x = np.linspace(low - margin, high + margin, _N_GRID_POINTS)
    # ``pdf`` broadcasts the column of points against the samples.
    densities = y_dist.pdf(x.reshape((-1, 1))).transpose()

    n_rows = ceil(Decimal(n_samples).sqrt())
    n_cols = n_samples // n_rows + int(n_samples % n_rows > 0)

    set_font(fontfamily=fontfamily)

    # ``squeeze=False`` keeps the grid two-dimensional for one row or column.
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        squeeze=False,
        facecolor="white",
        dpi=72,
        figsize=(6.4 * n_cols, 4.8 * n_rows),
    )

    if verbose:
        pbar = tqdm(
            total=n_cols * n_rows + 1, desc="Drawing distribution figure"
        )
    # Draw the distribution of each sample.
    for idx in range(n_samples):
        ax = axes[idx // n_cols, idx % n_cols]
        ax.plot(x, densities[idx], c="#022C5E")

        # The vertical lines reach the highest density of this sample.
        peak = max(densities[idx])
        if y_true is not None:
            ax.vlines(y_true[idx], 0, peak, "#ab4e15", label="ground truth")
        ax.vlines(y_pred[idx], 0, peak, "#dc143c", label="pred")
        ax.set_ylabel("Probability density")
        ax.legend(loc="best", facecolor="#f0f0f0", edgecolor="None")
        ax.set_title(subplot_titles[idx])

        ax.set_xlim(x[0], x[-1])
        if keep_y_range:
            ax.set_ylim(None, np.max(densities) * 1.05)

        if verbose:
            pbar.update(1)
    # Hide the unused subplots.
    for idx in range(n_samples, n_rows * n_cols):
        ax = axes[idx // n_cols, idx % n_cols]
        ax.axis("off")
        if verbose:
            pbar.update(1)

    fig.tight_layout()

    if verbose:
        pbar.update(1)
        pbar.set_description(desc="Completed")
        pbar.close()
    if return_axis:
        return fig, axes
    else:
        return fig


def is_correct_dist(y_pred, y_dist):
    b = np.allclose(y_dist.mean(), y_pred)
    if not b:
        warnings.warn("`y_dist.mean()` and `y_pred` is not close.")
    return b


def _get_evals_result(
    estimator: Any,
) -> Mapping[str, Mapping[str, Sequence[float]]]:
    """Return the evaluation record of a fitted NGBoost or LightGBM model.

    The record maps the name of each evaluation data to the scores of each
    metric over the boosting iterations.
    """
    if isinstance(estimator, _NGBOOST_MODELS):
        attribute = "evals_result"
        hint = (
            "fit it first; NGBoost records the scores of the training data "
            "and of the validation data (X_val and Y_val)."
        )
    elif isinstance(estimator, _LIGHTGBM_MODELS):
        attribute = "evals_result_"
        hint = (
            "fit it with evaluation data (eval_set, or eval_X and eval_y in "
            "LightGBM 4.7 and later) and a metric other than 'None'."
        )
    else:
        raise TypeError(
            "estimator must be an NGBoost or LightGBM model, such as "
            "NGBRegressor or LGBMRegressor, but got "
            f"{type(estimator).__name__}."
        )

    name = type(estimator).__name__
    try:
        evals_result = getattr(estimator, attribute)
    except AttributeError as error:
        # NGBoost sets the attribute in ``fit``; LightGBM raises its
        # NotFittedError, an AttributeError, before ``fit``.
        raise ValueError(
            f"{name} has no evaluation record ({attribute}) as it is not "
            f"fitted: {hint}"
        ) from error
    # Without evaluation data, LightGBM 4 records {} and LightGBM 3 None.
    if not evals_result or not all(evals_result.values()):
        raise ValueError(
            f"{name} has no evaluation record ({attribute} has no scores): "
            f"{hint}"
        )
    return evals_result


@with_custom_matplotlib_settings()
def get_learning_curve_gb(
    estimator, fontfamily: str | None = None, return_axis: bool = False
):
    """Plot the learning curve of a fitted NGBoost or LightGBM model.

    For each evaluation data recorded during the fit, the scores of the
    first metric are drawn as a line over the boosting iterations, labeled
    with the name of the data. The data named "train", "val" and "test"
    (NGBoost names its training and validation data "train" and "val")
    are drawn in fixed colors; other names, such as LightGBM's default
    "training" and "valid_0", take the colors of matplotlib's color cycle
    in order. The global font settings of matplotlib
    (``rcParams["font.family"]`` and ``rcParams["font.size"]``) are changed
    while the figure is drawn and restored afterwards.

    Parameters
    ----------
    estimator : NGBoost or LightGBM model
        A fitted NGBoost model, such as NGBRegressor or NGBClassifier,
        whose ``evals_result`` holds the scores of the training data and of
        the validation data passed to ``fit`` (``X_val`` and ``Y_val``). Or
        a fitted LightGBM model of the scikit-learn API, such as
        LGBMRegressor or LGBMClassifier, whose ``evals_result_`` holds the
        scores of the evaluation data passed to ``fit`` (``eval_set``, or
        ``eval_X`` and ``eval_y`` in LightGBM 4.7 and later).
    fontfamily : str or None, default=None
        The font family of the figure. None means "Helvetica" on macOS and
        "DejaVu Sans" on other platforms (see
        :func:`yikit.visualize.set_font`).
    return_axis : bool, default=False
        Whether to return the axes of the figure along with the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure. Returned alone when ``return_axis`` is False.
    ax : matplotlib.axes.Axes
        The axes of the figure. Returned as ``(fig, ax)`` only when
        ``return_axis`` is True.

    Raises
    ------
    TypeError
        If ``estimator`` is not an NGBoost or LightGBM model.
    ValueError
        If ``estimator`` has no evaluation record: it is not fitted, or it
        is a LightGBM model fitted without evaluation data or with
        ``metric="None"``.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from lightgbm import LGBMRegressor
    >>> from yikit.visualize import get_learning_curve_gb
    >>> rng = np.random.RandomState(334)
    >>> X = rng.uniform(-1.0, 1.0, size=(100, 3))
    >>> y = X @ np.array([3.0, -2.0, 1.0])
    >>> X_train, X_val, y_train, y_val = X[:80], X[80:], y[:80], y[80:]

    Unless ``eval_names`` is given, LightGBM names the training data
    "training" and the other evaluation data "valid_<position>".

    >>> model = LGBMRegressor(n_estimators=20, verbose=-1).fit(
    ...     X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)]
    ... )
    >>> fig, ax = get_learning_curve_gb(
    ...     model, fontfamily="DejaVu Sans", return_axis=True
    ... )
    >>> [line.get_label() for line in ax.get_lines()]
    ['training', 'valid_1']
    >>> ax.get_xlabel(), ax.get_ylabel()
    ('n_iter', 'l2')
    >>> plt.close(fig)
    """
    evals_result = _get_evals_result(estimator)

    set_font(fontfamily=fontfamily)

    fig, ax = plt.subplots(1, 1, facecolor="white", dpi=72)

    for data_name, scores_by_metric in evals_result.items():
        # NGBoost and LightGBM record the same metrics for every data.
        metric_name, scores = next(iter(scores_by_metric.items()))
        # Known names keep their colors; ``color=None`` takes the next
        # color of matplotlib's color cycle.
        ax.plot(
            range(len(scores)),
            scores,
            label=data_name,
            color=COLORS.get(data_name),
        )

    ax.set_xlabel("n_iter")
    ax.set_ylabel(metric_name)

    ax.legend(facecolor="#f0f0f0", edgecolor="None", fontsize=10)
    fig.tight_layout()
    if return_axis:
        return fig, ax
    else:
        return fig
