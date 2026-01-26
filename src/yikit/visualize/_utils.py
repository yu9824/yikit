"""Utility functions for matplotlib visualization.

This module provides helper functions for configuring matplotlib settings,
including font configuration and custom matplotlib settings context manager.
"""

import platform
from functools import wraps
from typing import Callable, Optional, TypeVar

import matplotlib.pyplot as plt
from sklearn.utils import Bunch

F = TypeVar("F", bound=Callable)

COLORS = Bunch(
    train="#283655",
    test="#cf3721",
    val="#cf3721",
)


def set_font(fontfamily: Optional[str] = None, fontsize: int = 13):
    """Set matplotlib font family and size.

    This function configures the default font family and size for matplotlib plots.
    If no font family is specified, it uses a platform-appropriate default.

    Parameters
    ----------
    fontfamily : str or None, default=None
        Font family to use. If None, uses platform default:
        - macOS: 'Helvetica'
        - Windows/Linux: 'DejaVu Sans'
    fontsize : int, default=13
        Font size to use for all text elements.

    Examples
    --------
    >>> from yikit.visualize import set_font
    >>> set_font(fontfamily="Arial", fontsize=14)
    >>> # Now all matplotlib plots will use Arial font at size 14
    """
    if fontfamily is None:
        fontfamily = _default_fontfamily()

    plt.rcParams["font.size"] = fontsize
    plt.rcParams["font.family"] = fontfamily


def _default_fontfamily():
    if platform.system() == "Darwin":
        return "Helvetica"
    else:  # Window or Linux
        return "DejaVu Sans"


def with_custom_matplotlib_settings(
    fontfamily: Optional[str] = None, fontsize: int = 13, restore: bool = True
):
    """Decorator to apply custom matplotlib settings temporarily.

    This decorator applies custom font family and size settings to matplotlib
    for the duration of the decorated function execution. After execution,
    the original settings are restored if `restore=True`.

    Parameters
    ----------
    fontfamily : str or None, optional
        Font family to use. If None, uses platform default:
        - macOS: 'Helvetica'
        - Windows/Linux: 'DejaVu Sans'
        Default is None.
    fontsize : int, optional
        Font size to use for all text elements. Default is 13.
    restore : bool, optional
        Whether to restore original matplotlib settings after function execution.
        Default is True.

    Returns
    -------
    Callable
        Decorator function that wraps the target function.

    Examples
    --------
    >>> from yikit.visualize import with_custom_matplotlib_settings
    >>> import matplotlib.pyplot as plt
    >>>
    >>> @with_custom_matplotlib_settings(fontfamily="Helvetica", fontsize=14)
    ... def plot_data():
    ...     plt.plot([1, 2, 3], [1, 4, 9])
    ...     return plt.gcf()
    >>>
    >>> fig = plot_data()  # Uses Helvetica font at size 14
    >>> # After function execution, original settings are restored
    """
    if fontfamily is None:
        fontfamily = _default_fontfamily()

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 現在の設定を保存
            if restore:
                original_font_size = plt.rcParams["font.size"]
                original_font_family = plt.rcParams["font.family"]

            # 設定を適用
            plt.rcParams["font.size"] = fontsize
            plt.rcParams["font.family"] = fontfamily

            try:
                # 関数を実行
                result = func(*args, **kwargs)
                return result
            finally:
                # 設定を復元
                if restore:
                    plt.rcParams["font.size"] = original_font_size
                    plt.rcParams["font.family"] = original_font_family

        return wrapper  # type: ignore

    return decorator
