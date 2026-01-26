"""Feature selection methods for machine learning.

This module provides filter methods and wrapper methods for feature selection,
including correlation-based filtering and Boruta algorithm.
"""

from yikit.helpers import is_installed

from ._filter import FilterSelector

__all__ = ["FilterSelector"]

if is_installed("boruta"):
    from ._wrapper import BorutaPy  # noqa: F401

    __all__.append("BorutaPy")
