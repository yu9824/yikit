"""A Python toolkit for machine learning focused on regression tasks.

This package provides models, feature selection, visualization, metrics, and other utilities
for machine learning workflows.
"""

from __future__ import annotations

from pathlib import Path

# setuptools-scm writes ``_version.py`` from the git tag when the package
# is built or installed; a source tree that was never installed lacks it.
_version_file = Path(__file__).parent / "_version.py"

if _version_file.is_file():
    from ._version import __version__
else:
    __version__ = "unknown"

__license__ = "Apache-2.0"
__author__ = "yu9824"
__copyright__ = "Copyright © 2025 yu9824"
