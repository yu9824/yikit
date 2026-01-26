from yikit.helpers import is_installed

from ._ensemble import EnsembleRegressor
from ._linear import LinearModelRegressor
from ._optuna import Objective
from ._svm import SupportVectorRegressor

__all__ = [
    "EnsembleRegressor",
    "LinearModelRegressor",
    "Objective",
    "SupportVectorRegressor",
]

if is_installed("lightgbm"):
    from ._gbdt import GBDTRegressor  # noqa: F401

    __all__ += ["GBDTRegressor"]

if is_installed("keras"):
    from ._mlp import NNRegressor  # noqa: F401

    __all__ += ["NNRegressor"]
