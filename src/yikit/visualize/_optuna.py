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
    """get_leraning_curve

    Parameters
    ----------
    study : optuna.study.Study

    loc : str, optional
        legend's location, by default 'best'
    fontfamily : str, optional
        fontfamily, by default 'Helvetica'
    return_axis : bool, optional
        return axis or not, by default False

    Returns
    -------
    if return_axis is True:
        tuple (matplotlib.pyplot.figure, matplotlib.pyplot.axis)
    else:
        matplotlib.pyplot.figure
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
