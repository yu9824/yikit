"""Evaluation metrics that scikit-learn does not provide.

``log_likelihood`` scores NGBoost models and is available only when ngboost
is installed. The root mean squared error was removed in 0.4.0; use
scikit-learn's implementation instead.
"""

from __future__ import annotations

from yikit.helpers import is_installed

__all__: list[str] = []

if is_installed("ngboost"):
    from ._ngboost import log_likelihood  # noqa: F401

    __all__.append("log_likelihood")
