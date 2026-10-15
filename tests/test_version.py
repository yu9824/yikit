"""Tests of how :data:`yikit.__version__` is read.

setuptools-scm writes ``yikit/_version.py`` from the git tag when yikit is
built or installed, and ``yikit/__init__.py`` reads the version from that
file, or falls back to ``"unknown"`` when the file is missing (a source tree
that was never installed).

The version is compared with ``yikit._version`` and never with
:mod:`importlib.metadata`: a stale ``yikit.egg-info`` at the top of a
development checkout makes :mod:`importlib.metadata` report a wrong version.
"""

from __future__ import annotations

import importlib
import importlib.util
import shutil
import sys
from pathlib import Path

import pytest
from packaging.version import Version

import yikit

#: Name of the version file that setuptools-scm writes next to
#: ``yikit/__init__.py``.
VERSION_FILE_NAME = "_version.py"


def test_version_comes_from_generated_file() -> None:
    """The installed yikit reports the version of its generated file."""
    version_file = Path(yikit.__file__).with_name(VERSION_FILE_NAME)
    if not version_file.is_file():
        pytest.fail(
            f"The version file {version_file} is missing. setuptools-scm "
            "writes it when yikit is built or installed, so reinstall yikit "
            "(for a development checkout: `pip install -e .`) and run the "
            "tests again."
        )

    version_module = importlib.import_module("yikit._version")

    assert yikit.__version__ == version_module.version
    # Raises ``InvalidVersion`` unless the version follows PEP 440.
    Version(yikit.__version__)


def test_version_falls_back_without_version_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a version file, the import works and the version is unknown.

    The installed ``yikit/__init__.py`` is copied into a package of another
    name under ``tmp_path``, which has no version file, and loaded from
    there, so the installed yikit is left untouched.
    """
    package_name = "yikit_without_version_file"
    package_dir = tmp_path / package_name
    package_dir.mkdir()
    init_file = package_dir / "__init__.py"
    shutil.copyfile(yikit.__file__, init_file)
    assert not (package_dir / VERSION_FILE_NAME).exists()

    spec = importlib.util.spec_from_file_location(
        package_name,
        init_file,
        submodule_search_locations=[str(package_dir)],
    )
    assert spec is not None and spec.loader is not None, (
        f"Cannot build an import spec for {init_file}."
    )
    module = importlib.util.module_from_spec(spec)
    # Register the package like a real import would, so that relative
    # imports resolve; monkeypatch removes it after the test.
    monkeypatch.setitem(sys.modules, package_name, module)
    spec.loader.exec_module(module)

    assert module.__version__ == "unknown"
