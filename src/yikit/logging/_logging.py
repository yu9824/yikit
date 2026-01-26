"""Module for logging configuration and utilities."""

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
from typing import Optional, TypeVar

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


_default_handler: Optional[StreamHandler] = None
"""default root logger handler

if not configured, None
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

    If the environment supports colorized output, a `ColoredFormatter` from the
    `colorlog` package is returned. Otherwise, a standard `Formatter` is used.

    Returns
    -------
    Formatter
        A log formatter instance, either colored or plain depending on environment support.
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
    formatter: Optional[Formatter] = None,
    level: int = NOTSET,
) -> HandlerType:
    """
    Configure a log handler with a formatter and log level.

    This function simplifies handler configuration by allowing optional
    specification of a formatter and log level. If no formatter is provided,
    a default one will be used.

    Parameters
    ----------
    handler : HandlerType
        The log handler to configure.
    formatter : Optional[Formatter], optional
        The formatter to apply to the handler. If None, a default formatter
        is used. Default is None.
    level : int, optional
        The logging level to set for the handler (e.g., logging.DEBUG,
        logging.INFO). Default is logging.NOTSET.

    Returns
    -------
    HandlerType
        The configured log handler.
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

    if _default_handler:
        # This library has already configured the library root logger.
        return

    _default_handler = _create_default_handler()

    # Apply our default configuration to the library root logger.
    library_root_logger = get_root_logger()
    library_root_logger.addHandler(_default_handler)
    library_root_logger.setLevel(INFO)
    library_root_logger.propagate = False


def get_root_logger() -> Logger:
    """
    Retrieve the root logger for this library package.

    Ensures that the logger is properly configured with a default handler
    and logging level before returning it.

    Returns
    -------
    Logger
        The root logger instance for the current package.
    """
    _configure_library_root_logger()

    return getLogger(_get_root_logger_name())


def get_child_logger(name: str, propagate: bool = True) -> Logger:
    """
    Retrieve a child logger associated with the given module name.

    This function returns a child logger derived from the library's root logger.
    The `name` argument should typically be set to `__name__` to ensure that
    the logger is correctly nested under the library's namespace. If the name
    does not match the expected pattern, a ValueError is raised.

    Parameters
    ----------
    name : str
        The module name, typically set to `__name__`.
    propagate : bool, optional
        Whether log messages should propagate to the parent logger.
        Default is True.

    Returns
    -------
    Logger
        A configured child logger instance.

    Raises
    ------
    ValueError
        If the provided name does not belong to the library's namespace and
        is not '__main__'.
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
    Enable the default handler for the library's root logger.

    Re-attaches the default handler to the root logger if it has been
    previously removed.

    Returns
    -------
    None
    """
    _configure_library_root_logger()

    assert _default_handler is not None
    get_root_logger().addHandler(_default_handler)


def disable_default_handler() -> None:
    """
    Disable the default handler for the library's root logger.

    Detaches the default handler from the root logger to suppress logging output.

    Returns
    -------
    None
    """
    _configure_library_root_logger()

    assert _default_handler is not None
    get_root_logger().removeHandler(_default_handler)


class catch_default_handler:
    """
    Context manager to temporarily disable the default handler.

    When entering the context, the library's default log handler is removed
    from the root logger to suppress output. When exiting, the handler is
    re-attached.

    Examples
    --------
    >>> logger = get_child_logger(__name__)
    >>> with catch_default_handler():
    ...     logger.info("This message will not be logged.")
    >>> logger.info("This message will be logged.")
    """

    def __enter__(self) -> None:
        disable_default_handler()

    def __exit__(
        self,
        exc_type: "Optional[type[Exception]]",
        exc_value: Optional[Exception],
        traceback: Optional[TracebackType],
    ) -> None:
        enable_default_handler()
