"""Machine learning models and hyperparameter optimization utilities.

This module provides scikit-learn compatible regressors (an ensemble of
regressors and, when lightgbm is installed, a gradient boosting regressor)
and, when optuna is installed, utilities for hyperparameter optimization.
"""

from __future__ import annotations

from yikit.helpers import is_installed

from ._ensemble import EnsembleRegressor

__all__ = [
    "EnsembleRegressor",
]

if is_installed("optuna"):
    from ._optuna import Objective, ParamDistributions, RecommendedParams

    __all__ += ["Objective", "ParamDistributions", "RecommendedParams"]

if is_installed("lightgbm"):
    from ._gbdt import GBDTRegressor  # noqa: F401

    __all__ += ["GBDTRegressor"]
