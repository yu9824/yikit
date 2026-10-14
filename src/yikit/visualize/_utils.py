"""Utility functions for matplotlib visualization.

This module provides helper functions for configuring matplotlib settings,
including font configuration and custom matplotlib settings context manager.
"""

from __future__ import annotations

import platform
from collections.abc import Callable
from functools import wraps
from typing import TypeVar

import matplotlib.pyplot as plt
from sklearn.utils import Bunch

F = TypeVar("F", bound=Callable)

COLORS = Bunch(
    train="#283655",
    test="#cf3721",
    val="#cf3721",
)


def set_font(fontfamily: str | None = None, fontsize: float = 13) -> None:
    """Set the global font family and font size of matplotlib.

    This changes ``matplotlib.pyplot.rcParams["font.family"]`` and
    ``rcParams["font.size"]`` for the rest of the session: every text
    created afterwards uses them, and they are not restored. To change them
    only for a block of code, wrap it in :func:`matplotlib.rc_context`; to
    change them only while a function runs, decorate it with
    :func:`with_custom_matplotlib_settings`.

    Parameters
    ----------
    fontfamily : str or None, default=None
        The font family. None means "Helvetica" on macOS and "DejaVu Sans"
        on other platforms. The name is not checked here: when a figure is
        drawn with a font that is not installed, matplotlib warns and falls
        back to its default font.
    fontsize : float, default=13
        The default font size in points. Sizes such as "medium" or "large"
        of the labels are relative to it.

    See Also
    --------
    with_custom_matplotlib_settings : Set the font only while a function
        runs.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from yikit.visualize import set_font
    >>> set_font(fontfamily="Arial", fontsize=14)  # doctest: +SKIP

    The settings stay changed after the call; ``matplotlib.rc_context``
    restores them at the end of the block.

    >>> with plt.rc_context():
    ...     set_font(fontfamily="DejaVu Sans", fontsize=14)
    ...     print(plt.rcParams["font.family"], plt.rcParams["font.size"])
    ['DejaVu Sans'] 14.0
    """
    if fontfamily is None:
        fontfamily = _default_fontfamily()

    plt.rcParams["font.size"] = fontsize
    plt.rcParams["font.family"] = fontfamily


def _default_fontfamily() -> str:
    if platform.system() == "Darwin":
        return "Helvetica"
    else:  # Windows or Linux
        return "DejaVu Sans"


def with_custom_matplotlib_settings(
    fontfamily: str | None = None, fontsize: float = 13, restore: bool = True
) -> Callable[[F], F]:
    """Return a decorator that sets the font while the function runs.

    The decorated function runs with the global font family and font size
    of matplotlib (``rcParams["font.family"]`` and ``rcParams["font.size"]``)
    set to ``fontfamily`` and ``fontsize``. With ``restore=True`` the
    previous values of these two settings are restored when the function
    returns or raises; other settings changed inside the function are not
    restored.

    Parameters
    ----------
    fontfamily : str or None, default=None
        The font family. None means "Helvetica" on macOS and "DejaVu Sans"
        on other platforms (see :func:`set_font`).
    fontsize : float, default=13
        The default font size in points.
    restore : bool, default=True
        Whether to restore the previous font family and font size after the
        call. If False, the settings stay changed as with :func:`set_font`.

    Returns
    -------
    Callable
        A decorator that wraps a function, keeping its name and docstring.

    See Also
    --------
    set_font : Set the font for the rest of the session.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from yikit.visualize import with_custom_matplotlib_settings
    >>> @with_custom_matplotlib_settings(fontfamily="DejaVu Sans", fontsize=14)
    ... def font_size():
    ...     return plt.rcParams["font.size"]
    >>> with plt.rc_context({"font.size": 10}):
    ...     print(font_size(), plt.rcParams["font.size"])
    14.0 10.0
    """
    if fontfamily is None:
        fontfamily = _default_fontfamily()

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Save the current settings.
            if restore:
                original_font_size = plt.rcParams["font.size"]
                original_font_family = plt.rcParams["font.family"]

            # Apply the settings.
            plt.rcParams["font.size"] = fontsize
            plt.rcParams["font.family"] = fontfamily

            try:
                # Run the function.
                result = func(*args, **kwargs)
                return result
            finally:
                # Restore the settings, also when the function raised.
                if restore:
                    plt.rcParams["font.size"] = original_font_size
                    plt.rcParams["font.family"] = original_font_family

        return wrapper  # type: ignore

    return decorator
