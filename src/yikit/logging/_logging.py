"""Implementation of :mod:`yikit.logging`.

See the docstring of :mod:`yikit.logging` for how yikit configures the
``"yikit"`` logger and how to silence or replace its default handler.
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
from logging import (
    INFO,
    NOTSET,
    Formatter,
    Handler,
    Logger,
    StreamHandler,
    getLogger,
)
from types import TracebackType
from typing import TypeVar

HandlerType = TypeVar("HandlerType", bound=Handler)


def _color_supported() -> bool:
    """
    Detect whether color support is available for logging output.

    Returns
    -------
    bool
        True if color support is available (colorlog is installed, NO_COLOR
        is not set, and stderr is a TTY), False otherwise.
    """
    if not importlib.util.find_spec("colorlog"):
        return False

    # NO_COLOR environment variable:
    if os.environ.get("NO_COLOR", None):
        return False

    if not hasattr(sys.stderr, "isatty") or not sys.stderr.isatty():
        return False
    else:
        return True


_default_handler: StreamHandler | None = None
"""Default handler of the "yikit" logger.

None until :func:`get_root_logger` configures the logger for the first time.
"""


def _get_root_logger_name() -> str:
    """
    Retrieve the name of the root logger, corresponding to the package name.

    Returns
    -------
    str
        The name of the root logger, derived from the top-level package name.
    """
    return __name__.split(".")[0]


def create_default_formatter() -> Formatter:
    """
    Create a default log formatter with optional color support.

    If the environment supports colorized output (``colorlog`` is installed,
    the ``NO_COLOR`` environment variable is not set, and ``sys.stderr`` is a
    terminal), a ``colorlog.ColoredFormatter`` is returned. Otherwise, a
    standard :class:`logging.Formatter` with the same fields is returned.

    Returns
    -------
    Formatter
        A new log formatter, either colored or plain depending on the
        environment.
    """
    if _color_supported():
        from colorlog import ColoredFormatter

        return ColoredFormatter(
            "%(asctime)s - %(name)s:%(lineno)d%(log_color)s[%(levelname)s]%(reset)s - %(message)s"
        )
    else:
        return Formatter(
            "%(asctime)s - %(name)s:%(lineno)d[%(levelname)s] - %(message)s"
        )


default_formatter: Formatter = create_default_formatter()
"""
Default log formatter instance used for configuring log output.

This formatter is either colorized or plain depending on environment support.
"""


def get_handler(
    handler: HandlerType,
    formatter: Formatter | None = None,
    level: int = NOTSET,
) -> HandlerType:
    """
    Configure a log handler with a formatter and a log level.

    Use it to build a handler that replaces or complements the default
    handler of the ``"yikit"`` logger.

    Parameters
    ----------
    handler : logging.Handler
        The log handler to configure. It is modified in place.
    formatter : logging.Formatter or None, default=None
        The formatter to set on the handler. If None, a new default formatter
        (the one the default handler uses) is set.
    level : int, default=logging.NOTSET
        The level to set on the handler (e.g., ``logging.DEBUG``,
        ``logging.INFO``).

    Returns
    -------
    logging.Handler
        The same ``handler``, configured.

    Examples
    --------
    >>> import logging
    >>> from yikit.logging import WARNING, get_handler
    >>> handler = get_handler(logging.StreamHandler(), level=WARNING)
    >>> handler.level == WARNING
    True
    """
    handler.setLevel(level)
    handler.setFormatter(
        formatter if formatter else create_default_formatter()
    )
    return handler


def _create_default_handler() -> StreamHandler:
    """
    Create a default stream handler with standard configuration.

    Returns
    -------
    StreamHandler
        A stream handler configured with the default formatter and log level.
    """
    return get_handler(StreamHandler())


def _configure_library_root_logger() -> None:
    """
    Configure the root logger for the library if it has not been set.

    This function initializes and attaches a default handler to the library's
    root logger, sets the logging level to INFO, and disables propagation to
    avoid duplicate log messages. It is safe to call multiple times;
    configuration is applied only once.

    Returns
    -------
    None
    """
    global _default_handler

    if _default_handler is not None:
        # This library has already configured the library root logger.
        return

    _default_handler = _create_default_handler()

    # Apply our default configuration to the library root logger.
    library_root_logger = get_root_logger()
    library_root_logger.addHandler(_default_handler)
    library_root_logger.setLevel(INFO)
    library_root_logger.propagate = False


def _get_default_handler() -> StreamHandler:
    """
    Return the default handler, configuring the logger if needed.

    Returns
    -------
    StreamHandler
        The default handler of the ``"yikit"`` logger.

    Raises
    ------
    RuntimeError
        If the default handler does not exist after the configuration.
    """
    _configure_library_root_logger()

    if _default_handler is None:
        raise RuntimeError(
            "The default handler of the "
            f"{_get_root_logger_name()!r} logger was not created by the "
            "logger configuration."
        )
    return _default_handler


def _is_default_handler_attached() -> bool:
    """
    Return whether the default handler is attached to the root logger.

    Returns
    -------
    bool
        True if the default handler is attached to the ``"yikit"`` logger.
    """
    return _get_default_handler() in get_root_logger().handlers


def get_root_logger() -> Logger:
    """
    Return the ``"yikit"`` logger, configuring it on the first call.

    The first call attaches the default handler (a
    :class:`logging.StreamHandler` that writes to ``sys.stderr`` with the
    default formatter) to the ``"yikit"`` logger, sets its level to
    ``logging.INFO``, and sets ``propagate`` to False so that the messages
    are not printed again by the handlers of the Python root logger. Later
    calls return the logger without changing it, so the level, handlers and
    ``propagate`` set afterwards are kept. The level and ``propagate`` set on
    ``logging.getLogger("yikit")`` before the first call are overwritten, so
    change them on the logger that this function returns.

    Returns
    -------
    logging.Logger
        The ``"yikit"`` logger.

    See Also
    --------
    disable_default_handler : Detach the default handler.
    get_child_logger : Return a child of the ``"yikit"`` logger.

    Examples
    --------
    >>> from yikit.logging import get_root_logger
    >>> get_root_logger().name
    'yikit'
    """
    _configure_library_root_logger()

    return getLogger(_get_root_logger_name())


def get_child_logger(name: str, propagate: bool = True) -> Logger:
    """
    Return a child of the ``"yikit"`` logger for a module.

    Pass ``__name__``. A module inside yikit (``"yikit.<module>"``) gets the
    logger of the same name, and a script run as ``__main__`` gets the
    ``"yikit.__main__"`` logger. The ``"yikit"`` logger is configured first
    (see :func:`get_root_logger`), so the messages of the child are printed
    by the default handler while ``propagate`` is True.

    Parameters
    ----------
    name : str
        The module name, ``__name__``. It must be a module inside yikit
        (``"yikit.<module>"``) or ``"__main__"``.
    propagate : bool, default=True
        Whether the messages of the child are passed to the handlers of the
        ``"yikit"`` logger. It is set on the returned logger.

    Returns
    -------
    logging.Logger
        The child logger.

    Raises
    ------
    ValueError
        If ``name`` is neither a module inside yikit nor ``"__main__"``.

    Examples
    --------
    >>> from yikit.logging import get_child_logger
    >>> get_child_logger("yikit.feature_selection").name
    'yikit.feature_selection'
    >>> get_child_logger("__main__").name
    'yikit.__main__'
    """
    root_logger = get_root_logger()

    _result_logger = re.match(rf"{_get_root_logger_name()}\.(.+)", name)
    if _result_logger:
        child_logger = root_logger.getChild(_result_logger.group(1))
    elif name == "__main__":
        child_logger = root_logger.getChild(name)
    else:
        raise ValueError("You should use '__name__'.")

    child_logger.propagate = propagate
    return child_logger


def enable_default_handler() -> None:
    """
    Attach the default handler to the ``"yikit"`` logger.

    It undoes :func:`disable_default_handler`. The handler is never attached
    twice, so calling it again does nothing.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If the default handler does not exist after configuring the logger.

    See Also
    --------
    disable_default_handler : Detach the default handler.
    catch_default_handler : Detach the default handler inside a block.

    Examples
    --------
    >>> from yikit.logging import enable_default_handler
    >>> enable_default_handler()
    """
    get_root_logger().addHandler(_get_default_handler())


def disable_default_handler() -> None:
    """
    Detach the default handler from the ``"yikit"`` logger.

    The default handler no longer prints yikit's messages. Handlers that the
    user attached are kept. Calling it again does nothing.

    If the ``"yikit"`` logger has no other handler, Python's last-resort
    handler (``logging.lastResort``) still writes messages of ``WARNING`` and
    above to ``sys.stderr``. To hide them too, also add a
    :class:`logging.NullHandler` to the logger returned by
    :func:`get_root_logger`, or raise its level.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If the default handler does not exist after configuring the logger.

    See Also
    --------
    enable_default_handler : Attach the default handler again.
    catch_default_handler : Detach the default handler inside a block.

    Examples
    --------
    >>> from yikit.logging import (
    ...     disable_default_handler,
    ...     enable_default_handler,
    ... )
    >>> disable_default_handler()  # the default handler stops printing
    >>> enable_default_handler()  # it prints yikit's messages again
    """
    get_root_logger().removeHandler(_get_default_handler())


class catch_default_handler:
    """
    Context manager that detaches the default handler inside a block.

    On entering, the default handler is detached from the ``"yikit"``
    logger. On exiting, also when an exception is raised inside the block,
    the default handler goes back to the state before entering: it is
    attached again if it was attached, and stays detached if it was
    detached. Other handlers are not touched. The same instance may be
    entered again while it is active; each exit restores the state of the
    matching entry.

    Inside the block, if the ``"yikit"`` logger has no other handler,
    Python's last-resort handler (``logging.lastResort``) still writes
    messages of ``WARNING`` and above to ``sys.stderr``; see
    :func:`disable_default_handler`.

    Raises
    ------
    RuntimeError
        On exiting without entering, or if the default handler does not exist
        after configuring the logger.

    See Also
    --------
    disable_default_handler : Detach the default handler.

    Examples
    --------
    >>> from yikit.logging import catch_default_handler, get_root_logger
    >>> logger = get_root_logger()
    >>> with catch_default_handler():
    ...     logger.info("This message is not printed by the default handler.")
    >>> logger.info("This message is printed again.")
    """

    def __init__(self) -> None:
        # Whether the default handler was attached at each active entry.
        self._was_attached: list[bool] = []

    def __enter__(self) -> None:
        self._was_attached.append(_is_default_handler_attached())
        disable_default_handler()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if not self._was_attached:
            raise RuntimeError(
                "catch_default_handler was exited without entering it."
            )
        if self._was_attached.pop():
            enable_default_handler()
        else:
            disable_default_handler()
