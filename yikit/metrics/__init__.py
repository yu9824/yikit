from yikit.helpers import is_installed

from ._common import root_mean_squared_error

__all__ = ["root_mean_squared_error"]

if is_installed("ngboost"):
    from ._ngboost import log_likelihood  # noqa: F401

    __all__.append("log_likelihood")
