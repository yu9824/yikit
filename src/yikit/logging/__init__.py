"""Logging utilities of yikit.

yikit writes its log messages to the ``"yikit"`` logger and its children
(for example ``"yikit.feature_selection._filter"``). The first call of
:func:`get_root_logger` configures the ``"yikit"`` logger once.
:func:`get_child_logger` calls it, and the yikit modules that log (such as
:mod:`yikit.feature_selection`) call :func:`get_child_logger` when they are
imported, so importing them configures the logger as follows.

- The default handler, a :class:`logging.StreamHandler` that writes to
  ``sys.stderr``, is attached. Its formatter is colored when ``colorlog`` is
  installed, the ``NO_COLOR`` environment variable is not set, and
  ``sys.stderr`` is a terminal.
- The level is set to ``INFO``, so messages of ``INFO`` and above are
  printed.
- ``propagate`` is set to False, so the messages are not passed to the
  handlers of the Python root logger and are not printed twice.

Later calls do not change the logger again, so the level, handlers and
``propagate`` set afterwards are kept. The level and ``propagate`` set on
``logging.getLogger("yikit")`` before that first configuration are
overwritten by it, so change them on the logger returned by
:func:`get_root_logger`, which configures the logger first; this works
whether yikit's modules are imported before or after. To silence or
replace the default handler:

- ``get_root_logger().setLevel(WARNING)`` keeps only warnings and errors.
- :func:`disable_default_handler` detaches the default handler and
  :func:`enable_default_handler` attaches it again.
  :class:`catch_default_handler` detaches it only inside a ``with`` block.
  If the ``"yikit"`` logger has no other handler, Python's last-resort
  handler (``logging.lastResort``) still writes messages of ``WARNING`` and
  above to ``sys.stderr``. To hide them too, also add a
  :class:`logging.NullHandler` with
  ``get_root_logger().addHandler(logging.NullHandler())``, or raise the
  level.
- To use your own handler, detach the default handler and add yours (for
  example, one made by :func:`get_handler`) with
  ``get_root_logger().addHandler(handler)``.
- To use the handlers of the Python root logger instead, detach the default
  handler and set ``get_root_logger().propagate = True``.

The logging levels ``CRITICAL``, ``DEBUG``, ``ERROR``, ``INFO``, ``NOTSET``,
and ``WARNING`` are re-exported from the standard :mod:`logging` module.

Examples
--------
Print yikit's messages with your own handler instead of the default one:

>>> import logging
>>> import sys
>>> from yikit.logging import (
...     INFO,
...     WARNING,
...     disable_default_handler,
...     enable_default_handler,
...     get_handler,
...     get_root_logger,
... )
>>> logger = get_root_logger()
>>> handler = get_handler(
...     logging.StreamHandler(sys.stdout),
...     formatter=logging.Formatter("%(levelname)s:%(name)s:%(message)s"),
... )
>>> disable_default_handler()
>>> logger.addHandler(handler)
>>> logger.info("fitted")
INFO:yikit:fitted

Keep only warnings and errors:

>>> logger.setLevel(WARNING)
>>> logger.info("fitted")
>>> logger.warning("not converged")
WARNING:yikit:not converged

Go back to the default configuration:

>>> logger.setLevel(INFO)
>>> logger.removeHandler(handler)
>>> enable_default_handler()
"""

from __future__ import annotations

from logging import CRITICAL, DEBUG, ERROR, INFO, NOTSET, WARNING

from ._logging import (
    catch_default_handler,
    disable_default_handler,
    enable_default_handler,
    get_child_logger,
    get_handler,
    get_root_logger,
)

__all__ = (
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
