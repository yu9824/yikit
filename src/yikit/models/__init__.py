"""Machine learning model wrappers and ensemble methods.

This module provides scikit-learn compatible regressors including ensemble methods,
linear models, support vector machines, gradient boosting, neural networks, and
hyperparameter optimization utilities.
"""

from yikit.helpers import is_installed

from ._ensemble import EnsembleRegressor
from ._linear import LinearModelRegressor
from ._svm import SupportVectorRegressor

__all__ = [
    "EnsembleRegressor",
    "LinearModelRegressor",
    "SupportVectorRegressor",
]

if is_installed("optuna"):
    from ._optuna import Objective

    __all__ += ["Objective"]

if is_installed("lightgbm"):
    from ._gbdt import GBDTRegressor  # noqa: F401

    __all__ += ["GBDTRegressor"]

if is_installed("keras"):
    from ._mlp import NNRegressor  # noqa: F401

    __all__ += ["NNRegressor"]
