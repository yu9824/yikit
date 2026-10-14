"""Tests of ``yikit.logging``."""

from __future__ import annotations

import io
import logging

import pytest

import yikit.logging
from yikit.logging import (
    CRITICAL,
    DEBUG,
    ERROR,
    INFO,
    NOTSET,
    WARNING,
    _logging,
    catch_default_handler,
    disable_default_handler,
    enable_default_handler,
    get_child_logger,
    get_handler,
    get_root_logger,
)

#: Name of the logger that yikit configures.
ROOT_LOGGER_NAME = "yikit"


@pytest.fixture(autouse=True)
def yikit_logger():
    """Yield the "yikit" logger and restore its global state afterwards.

    The handlers, level and propagation of the "yikit" logger and the
    module's default handler are restored, so the logging of the other
    tests is not affected by these tests. The test must not leave new
    instance attributes on the default handler either.
    """
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    handlers = list(logger.handlers)
    level = logger.level
    propagate = logger.propagate
    default_handler = _logging._default_handler
    attributes = set(vars(default_handler)) if default_handler else set()
    try:
        yield logger
    finally:
        _logging._default_handler = default_handler
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
        for handler in handlers:
            logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = propagate
    if default_handler is not None:
        assert set(vars(default_handler)) == attributes


def _default_handler() -> logging.Handler:
    """Return the default handler, configuring the logger if needed."""
    get_root_logger()
    handler = _logging._default_handler
    assert handler is not None
    return handler


def _is_attached(logger: logging.Logger) -> bool:
    """Return whether the default handler is attached to ``logger``."""
    return _default_handler() in logger.handlers


def _record() -> logging.LogRecord:
    """Return a log record with fixed contents."""
    record = logging.LogRecord(
        name="yikit.test",
        level=logging.WARNING,
        pathname=__file__,
        lineno=1,
        msg="message %d",
        args=(334,),
        exc_info=None,
    )
    record.created = 0.0
    record.msecs = 0.0
    return record


# --- Public names ------------------------------------------------------------


def test_public_names_are_unchanged():
    assert tuple(yikit.logging.__all__) == (
        "catch_default_handler",
        "disable_default_handler",
        "enable_default_handler",
        "get_child_logger",
        "get_handler",
        "get_root_logger",
        "CRITICAL",
        "DEBUG",
        "ERROR",
        "INFO",
        "NOTSET",
        "WARNING",
    )


def test_level_constants_are_the_standard_library_values():
    assert (CRITICAL, DEBUG, ERROR, INFO, NOTSET, WARNING) == (
        logging.CRITICAL,
        logging.DEBUG,
        logging.ERROR,
        logging.INFO,
        logging.NOTSET,
        logging.WARNING,
    )


# --- get_root_logger ---------------------------------------------------------


def test_get_root_logger_configures_the_yikit_logger(yikit_logger):
    # Start from a logger that yikit has not configured yet.
    _logging._default_handler = None
    for handler in list(yikit_logger.handlers):
        yikit_logger.removeHandler(handler)
    yikit_logger.setLevel(NOTSET)
    yikit_logger.propagate = True

    logger = get_root_logger()

    assert logger is yikit_logger
    default_handler = _logging._default_handler
    assert isinstance(default_handler, logging.StreamHandler)
    assert logger.handlers == [default_handler]
    assert logger.level == INFO
    assert logger.propagate is False


def test_get_root_logger_configures_the_logger_only_once(yikit_logger):
    default_handler = _default_handler()
    yikit_logger.setLevel(WARNING)

    logger = get_root_logger()

    # A second call keeps the user's settings and adds no handler.
    assert _logging._default_handler is default_handler
    assert logger.handlers.count(default_handler) == 1
    assert logger.level == WARNING


# --- get_child_logger --------------------------------------------------------


def test_get_child_logger_returns_a_child_of_the_yikit_logger(yikit_logger):
    logger = get_child_logger("yikit.feature_selection._filter")

    assert logger is logging.getLogger("yikit.feature_selection._filter")
    assert logger.parent is yikit_logger
    assert logger.propagate is True


def test_get_child_logger_accepts_main():
    logger = get_child_logger("__main__")

    assert logger.name == "yikit.__main__"


def test_get_child_logger_sets_propagate():
    name = "yikit._test_logging_propagate"
    try:
        assert get_child_logger(name, propagate=False).propagate is False
        assert get_child_logger(name).propagate is True
    finally:
        logging.getLogger(name).propagate = True


@pytest.mark.parametrize("name", [__name__, "numpy.linalg", "yikitx.module"])
def test_get_child_logger_rejects_a_name_outside_yikit(name):
    with pytest.raises(ValueError, match="__name__"):
        get_child_logger(name)


# --- enable_default_handler / disable_default_handler ------------------------


def test_disable_and_enable_never_duplicate_the_handler(yikit_logger):
    default_handler = _default_handler()

    for _ in range(3):
        disable_default_handler()
        assert default_handler not in yikit_logger.handlers

    for _ in range(3):
        enable_default_handler()
        assert yikit_logger.handlers.count(default_handler) == 1


@pytest.mark.parametrize(
    "function", [enable_default_handler, disable_default_handler]
)
def test_missing_default_handler_raises_runtime_error(monkeypatch, function):
    # The explicit check must not depend on ``assert`` (``python -O``).
    monkeypatch.setattr(_logging, "_configure_library_root_logger", lambda: 0)
    _logging._default_handler = None

    with pytest.raises(RuntimeError, match="default handler"):
        function()


# --- catch_default_handler ---------------------------------------------------


def test_catch_default_handler_restores_an_attached_handler(yikit_logger):
    enable_default_handler()

    with catch_default_handler():
        assert not _is_attached(yikit_logger)

    assert yikit_logger.handlers.count(_default_handler()) == 1


def test_catch_default_handler_keeps_a_detached_handler(yikit_logger):
    disable_default_handler()

    with catch_default_handler():
        assert not _is_attached(yikit_logger)

    assert not _is_attached(yikit_logger)


@pytest.mark.parametrize("attached", [True, False])
def test_catch_default_handler_restores_on_an_exception(
    yikit_logger, attached
):
    if attached:
        enable_default_handler()
    else:
        disable_default_handler()

    with pytest.raises(KeyError):
        with catch_default_handler():
            raise KeyError("raised inside the context")

    assert _is_attached(yikit_logger) is attached


def test_catch_default_handler_suppresses_the_messages(
    yikit_logger, monkeypatch
):
    # ``stream`` and ``formatter`` are instance attributes, so monkeypatch
    # restores them as they were. A plain formatter keeps the output the same
    # whether the default formatter is colored or not.
    default_handler = _default_handler()
    stream = io.StringIO()
    monkeypatch.setattr(default_handler, "stream", stream)
    monkeypatch.setattr(
        default_handler, "formatter", logging.Formatter("%(message)s")
    )
    yikit_logger.setLevel(INFO)
    enable_default_handler()
    logger = get_child_logger("yikit._test_logging_catch")

    with catch_default_handler():
        logger.info("inside the block")
    logger.info("after the block")

    assert stream.getvalue() == "after the block\n"


def test_nested_catch_default_handler_restores_each_state(yikit_logger):
    enable_default_handler()
    outer = catch_default_handler()

    with outer:
        with catch_default_handler():
            assert not _is_attached(yikit_logger)
        assert not _is_attached(yikit_logger)
        # The same instance may also be entered again while it is active.
        with outer:
            assert not _is_attached(yikit_logger)
        assert not _is_attached(yikit_logger)

    assert _is_attached(yikit_logger)


def test_catch_default_handler_exit_without_enter_raises():
    with pytest.raises(RuntimeError, match="without entering"):
        catch_default_handler().__exit__(None, None, None)


# --- get_handler -------------------------------------------------------------


def test_get_handler_sets_the_level_and_formatter():
    formatter = logging.Formatter("%(levelname)s:%(message)s")
    handler = logging.StreamHandler()

    returned = get_handler(handler, formatter=formatter, level=WARNING)

    assert returned is handler
    assert handler.level == WARNING
    assert handler.formatter is formatter


def test_get_handler_uses_the_default_formatter():
    handler = get_handler(logging.StreamHandler())

    assert handler.level == NOTSET
    assert handler.formatter is not None
    expected = _logging.create_default_formatter()
    assert type(handler.formatter) is type(expected)
    assert handler.formatter.format(_record()) == expected.format(_record())


def test_default_formatter_is_plain_without_color(monkeypatch):
    monkeypatch.setattr(_logging, "_color_supported", lambda: False)

    handler = get_handler(logging.StreamHandler())

    assert type(handler.formatter) is logging.Formatter
    message = handler.format(_record())
    assert message.endswith(" - yikit.test:1[WARNING] - message 334")
