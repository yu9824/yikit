"""Evaluation metrics for machine learning models.

This module provides various metrics for evaluating regression models,
including root mean squared error and log-likelihood for NGBoost.
"""

from yikit.helpers import is_installed

from ._common import root_mean_squared_error

__all__ = ["root_mean_squared_error"]

if is_installed("ngboost"):
    from ._ngboost import log_likelihood  # noqa: F401

    __all__.append("log_likelihood")
