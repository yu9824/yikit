"""Visualization utilities for Optuna optimization results.

This module provides visualization functions for Optuna study results,
including learning curves and optimization history plots.
"""

from typing import Optional

import matplotlib.pyplot as plt
import optuna
import pandas as pd

from yikit.visualize._utils import set_font, with_custom_matplotlib_settings


@with_custom_matplotlib_settings()
def get_learning_curve_optuna(
    study: optuna.study.Study,
    loc="best",
    fontfamily: Optional[str] = None,
    return_axis: bool = False,
):
    """Plot learning curve for Optuna optimization study.

    This function visualizes the optimization history of an Optuna study,
    showing both the objective values of all trials and the best value
    found so far at each trial.

    Parameters
    ----------
    study : optuna.study.Study
        An Optuna study object containing the optimization history.
    loc : str, optional
        Legend location. Can be any valid matplotlib legend location string.
        Default is 'best'.
    fontfamily : str or None, optional
        Font family to use for the plot. If None, uses default font.
        Default is None.
    return_axis : bool, optional
        Whether to return the matplotlib axis object along with the figure.
        Default is False.

    Returns
    -------
    matplotlib.pyplot.Figure or tuple
        If return_axis=False, returns only the figure.
        If return_axis=True, returns (figure, axis) tuple.

    Examples
    --------
    >>> from yikit.visualize import get_learning_curve_optuna
    >>> import optuna
    >>>
    >>> def objective(trial):
    ...     x = trial.suggest_float('x', -10, 10)
    ...     return (x - 2) ** 2
    >>>
    >>> study = optuna.create_study()
    >>> study.optimize(objective, n_trials=100)
    >>> fig = get_learning_curve_optuna(study)
    """
    # 2: maximize, 1: minimize
    if study.direction == optuna.study.StudyDirection(1):
        min_or_max = min
        temp = float("inf")
    elif study.direction == optuna.study.StudyDirection(2):
        min_or_max = max
        temp = -float("inf")
    df_values = pd.DataFrame(
        [trial.value for trial in study.trials], columns=["value"]
    )

    best_values = []
    for v in df_values.loc[:, "value"]:
        temp = min_or_max(temp, v)
        best_values.append(temp)
    df_values["best_value"] = best_values

    set_font(fontfamily=fontfamily)

    fig = plt.figure(facecolor="white")
    ax = fig.add_subplot(111)

    ax.scatter(
        df_values.index,
        df_values["value"],
        s=10,
        c="#00293c",
        label="Objective Value",
    )
    ax.plot(
        df_values.index,
        df_values["best_value"],
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
