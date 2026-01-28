"""Visualization tools for machine learning models.

This module provides utilities for visualizing model results, including
permutation importance, learning curves, distribution plots, and matplotlib
configuration helpers.
"""

from yikit.helpers import is_installed

from ._ngboost import get_dist_figure, get_learning_curve_gb
from ._permutation_importance import SummarizePI
from ._utils import set_font, with_custom_matplotlib_settings
from ._yyplot import yyplot

__all__ = [
    "SummarizePI",
    "set_font",
    "with_custom_matplotlib_settings",
    "yyplot",
]

if is_installed("optuna"):
    from ._optuna import get_learning_curve_optuna

    __all__ += ["get_learning_curve_optuna"]

if is_installed("ngboost"):
    from ._ngboost import get_dist_figure

    __all__ += ["get_dist_figure"]

if is_installed("lightgbm") or is_installed("ngboost"):
    from ._ngboost import get_learning_curve_gb

    __all__ += ["get_learning_curve_gb"]
