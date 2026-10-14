"""Visualization utilities for Optuna optimization results.

This module provides visualization functions for Optuna study results,
including learning curves and optimization history plots.
"""

from __future__ import annotations

from itertools import accumulate
from typing import TYPE_CHECKING, cast

import matplotlib.pyplot as plt
import optuna
from optuna.study import StudyDirection
from optuna.trial import TrialState

from yikit.visualize._utils import set_font, with_custom_matplotlib_settings

if TYPE_CHECKING:
    # ``LegendLocType`` is missing from older matplotlib, so import it for
    # typing only.
    from matplotlib.typing import LegendLocType


@with_custom_matplotlib_settings()
def get_learning_curve_optuna(
    study: optuna.study.Study,
    loc: LegendLocType = "best",
    fontfamily: str | None = None,
    return_axis: bool = False,
):
    """Plot the optimization history of a single-objective Optuna study.

    The objective value of each complete trial is drawn as a point at its
    trial number, and the best value so far (the lowest for a minimized
    study, the highest for a maximized one) as a line. Trials in the other
    states (failed, pruned, running or waiting) are left out, even when
    they hold a value, such as the last intermediate value that optuna
    keeps for a pruned trial. The global font settings of matplotlib
    (``rcParams["font.family"]`` and ``rcParams["font.size"]``) are changed
    while the figure is drawn and restored afterwards.

    Parameters
    ----------
    study : optuna.study.Study
        A single-objective study with a direction and at least one complete
        trial.
    loc : str, int or tuple of float, default="best"
        The location of the legend, passed to
        :meth:`matplotlib.axes.Axes.legend`: a location string such as
        ``"upper right"``, a location code, or the ``(x, y)`` position of
        the lower left corner of the legend in axes coordinates.
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
    ValueError
        If the study has more than one objective, has no direction
        (``StudyDirection.NOT_SET``), or has no complete trial.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import optuna
    >>> from yikit.visualize import get_learning_curve_optuna
    >>> def objective(trial):
    ...     x = trial.suggest_float("x", -10, 10)
    ...     return (x - 2) ** 2
    >>> study = optuna.create_study(
    ...     sampler=optuna.samplers.RandomSampler(seed=334)
    ... )
    >>> study.optimize(objective, n_trials=20)
    >>> fig, ax = get_learning_curve_optuna(
    ...     study, fontfamily="DejaVu Sans", return_axis=True
    ... )
    >>> ax.get_xlabel(), ax.get_ylabel()
    ('Trials', 'Objective Values')
    >>> plt.close(fig)
    """
    directions = study.directions
    if len(directions) > 1:
        raise ValueError(
            "get_learning_curve_optuna supports only single-objective "
            f"studies, but the study has {len(directions)} objectives."
        )
    direction = directions[0]
    if direction not in (StudyDirection.MINIMIZE, StudyDirection.MAXIMIZE):
        raise ValueError(
            "The study has no optimization direction (StudyDirection."
            f"{direction.name}), so its best value is undefined. Create it "
            "with direction='minimize' or direction='maximize'."
        )

    complete_trials = study.get_trials(
        deepcopy=False, states=(TrialState.COMPLETE,)
    )
    if not complete_trials:
        raise ValueError(
            "The study has no complete trial among its "
            f"{len(study.trials)} trials: only trials in the state "
            "TrialState.COMPLETE have an objective value to plot."
        )
    numbers = [trial.number for trial in complete_trials]
    # A complete trial of a single-objective study always has a value.
    values = [cast("float", trial.value) for trial in complete_trials]
    better = min if direction == StudyDirection.MINIMIZE else max
    best_values = list(accumulate(values, better))

    set_font(fontfamily=fontfamily)

    fig = plt.figure(facecolor="white")
    ax = fig.add_subplot(111)

    ax.scatter(
        numbers,
        values,
        s=10,
        c="#00293c",
        label="Objective Value",
    )
    ax.plot(
        numbers,
        best_values,
        c="#f62a00",
        zorder=0,
        label="Best Value",
    )

    ax.set_xlabel("Trials")
    ax.set_ylabel("Objective Values")

    ax.legend(facecolor="#f0f0f0", edgecolor="None", fontsize=10, loc=loc)

    fig.tight_layout()
    if return_axis:
        return fig, ax
    else:
        return fig
