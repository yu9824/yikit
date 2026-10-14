"""Permutation importance visualization.

This module provides utilities for visualizing permutation importance
results from machine learning models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from yikit.helpers import is_argument
from yikit.visualize._utils import set_font, with_custom_matplotlib_settings

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

_MEAN = "importances_mean"
_STD = "importances_std"


class SummarizePI:
    """Summarize permutation importances as a bar chart.

    The importances are divided by the sum of the per-feature means, so the
    means of the features sum to 1. For each feature, :meth:`get_data`
    returns the mean and the standard deviation of the divided importances
    over the repeats, and :meth:`get_figure` draws them as horizontal bars
    with error bars, the most important feature on top.

    Parameters
    ----------
    importances : pandas.DataFrame of shape (n_features, n_repeats)
        The permutation importances: one row per feature (the index holds
        the feature names) and one column per repeat, such as
        ``pd.DataFrame(result.importances, index=X.columns)`` for the
        ``result`` of :func:`sklearn.inspection.permutation_importance`.

    Attributes
    ----------
    importances : pandas.DataFrame of shape (n_features, n_repeats)
        The importances given to the constructor.
    fig : matplotlib.figure.Figure
        The figure drawn by the last call of :meth:`get_figure`. Defined
        only after it is called.
    ax : matplotlib.axes.Axes
        The axes of :attr:`fig`. Defined only after :meth:`get_figure` is
        called.

    Examples
    --------
    >>> import pandas as pd
    >>> from yikit.visualize import SummarizePI
    >>> importances = pd.DataFrame(
    ...     [[0.4, 0.8], [0.2, 0.2], [1.0, 1.4]], index=["a", "b", "c"]
    ... )
    >>> summarize_pi = SummarizePI(importances)
    >>> summarize_pi.get_data()
       importances_mean  importances_std
    c               0.6              0.1
    a               0.3              0.1
    b               0.1              0.0
    >>> fig, ax = summarize_pi.get_figure(fontfamily="DejaVu Sans")
    >>> [label.get_text() for label in ax.get_yticklabels()]
    ['c', 'a', 'b']

    >>> import matplotlib.pyplot as plt
    >>> plt.close(fig)
    """

    def __init__(self, importances: pd.DataFrame):
        self.importances = importances

    def get_data(self) -> pd.DataFrame:
        """Return the values drawn by :meth:`get_figure`.

        Returns
        -------
        pandas.DataFrame of shape (n_features, 2)
            One row per feature, indexed by the index of
            :attr:`importances` and sorted by ``"importances_mean"`` in
            descending order (features with equal means keep their order).
            The column ``"importances_mean"`` holds the mean of the
            importances over the repeats divided by the sum of these means,
            and ``"importances_std"`` the standard deviation of the divided
            importances over the repeats (with ``ddof=0``, as the
            ``importances_std`` of
            :func:`sklearn.inspection.permutation_importance`).
        """
        mean = self.importances.mean(axis=1)
        normalized = self.importances / mean.sum()
        data = pd.DataFrame(
            {
                _MEAN: normalized.mean(axis=1),
                _STD: normalized.std(axis=1, ddof=0),
            }
        )
        # A stable sort keeps the order of the features with equal means.
        return data.sort_values(_MEAN, ascending=False, kind="mergesort")

    @with_custom_matplotlib_settings()
    def get_figure(self, fontfamily: str | None = None) -> tuple[Figure, Axes]:
        """Draw the values of :meth:`get_data` as horizontal bars.

        Each bar is the normalized mean importance of a feature and its
        error bar spans one standard deviation on each side. The global
        font settings of matplotlib (``rcParams["font.family"]`` and
        ``rcParams["font.size"]``) are changed while the figure is drawn
        and restored afterwards.

        Parameters
        ----------
        fontfamily : str or None, default=None
            The font family of the figure. None means "Helvetica" on macOS
            and "DejaVu Sans" on other platforms (see
            :func:`yikit.visualize.set_font`).

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure, also stored in :attr:`fig`.
        ax : matplotlib.axes.Axes
            The axes of the figure, also stored in :attr:`ax`.
        """
        data = self.get_data()

        set_font(fontfamily=fontfamily)

        self.fig = plt.figure(facecolor="white")
        self.ax = self.fig.add_subplot(111)

        # One bar per column of the one-row wide-form frame, at y = 0, 1,
        # ... from the top in the order of ``data``. ``order`` keeps that
        # order for numeric feature names too, which seaborn >= 0.13 would
        # otherwise sort. The error bars are drawn below, so seaborn's own
        # are turned off (``ci`` before seaborn 0.12).
        sns.barplot(
            data=data[[_MEAN]].transpose(),
            order=list(data.index),
            ax=self.ax,
            orient="h",
            **(
                {"errorbar": None}
                if is_argument(sns.barplot, "errorbar")
                else {"ci": None}
            ),
        )
        # seaborn sets the y range to the bars and leaves autoscaling on;
        # keep that range instead of the margins the error bars would add.
        ylim = self.ax.get_ylim()
        self.ax.errorbar(
            data[_MEAN].to_numpy(),
            np.arange(len(data)),
            xerr=data[_STD].to_numpy(),
            fmt="none",
            ecolor=".26",
        )
        self.ax.set_ylim(ylim, auto=None)

        self.fig.tight_layout()

        return self.fig, self.ax
