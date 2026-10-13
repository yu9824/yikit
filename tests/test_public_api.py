"""Coverage tests of the public API of the non-model subpackages.

For each of :mod:`yikit.helpers`, :mod:`yikit.logging`, :mod:`yikit.metrics`,
:mod:`yikit.feature_selection` and :mod:`yikit.visualize`, these tests walk
the names that are in ``__all__`` now, so names that need a missing optional
dependency are skipped naturally, and check that

- every name, except the standard logging levels that :mod:`yikit.logging`
  re-exports, has an English docstring with a summary line, and with a
  numpy-style ``Parameters`` section when it takes parameters;
- every name, the logging levels included, is used as an identifier in at
  least one test file of the module-quality spec (this file does not count);
- every module of the subpackages has an English module docstring, and the
  comments and docstrings of these modules and of ``tests/conftest.py`` and
  the spec's test files are in English.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import io
import re
import tokenize
from pathlib import Path

import pytest

#: Subpackages whose public API is checked.
SUBPACKAGES = (
    "yikit.helpers",
    "yikit.logging",
    "yikit.metrics",
    "yikit.feature_selection",
    "yikit.visualize",
)

#: Logging levels that ``yikit.logging`` re-exports from the standard
#: library. They are plain integers without their own docstrings, so only
#: the usage rule applies to them.
STDLIB_LOGGING_LEVELS = frozenset(
    {"CRITICAL", "DEBUG", "ERROR", "INFO", "NOTSET", "WARNING"}
)

TESTS_DIR = Path(__file__).resolve().parent

#: Test files of the module-quality spec. Every public name must be used in
#: one of them; this file is deliberately left out, because it names every
#: public name only to check them.
SPEC_TEST_FILES = tuple(
    TESTS_DIR / name
    for name in (
        "test_helpers.py",
        "test_logging.py",
        "test_metrics.py",
        "test_filtermethod.py",
        "test_boruta.py",
        "test_visualize.py",
    )
)

#: CJK symbols and punctuation, hiragana, katakana, CJK unified ideographs
#: (with extension A), and half-width and full-width forms.
_JAPANESE = re.compile(
    r"[\u3000-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uff00-\uffef]"
)

#: A numpy-style ``Parameters`` header in a docstring cleaned by
#: :func:`inspect.cleandoc`.
_PARAMETERS_SECTION = re.compile(r"^Parameters\n-{10}$", re.MULTILINE)

#: Tokens skipped when looking for the token before a name, so that a ``.``
#: at the end of a line still marks the name on the next line as an
#: attribute.
_LAYOUT_TOKENS = frozenset(
    {
        tokenize.COMMENT,
        tokenize.NL,
        tokenize.NEWLINE,
        tokenize.INDENT,
        tokenize.DEDENT,
    }
)


def _public_names(package_name: str) -> list[str]:
    """Return the current ``__all__`` of ``package_name``."""
    return list(importlib.import_module(package_name).__all__)


def _package_sources(package_name: str) -> list[Path]:
    """Return the Python source files of ``package_name``."""
    package = importlib.import_module(package_name)
    return sorted(
        path
        for directory in package.__path__
        for path in Path(directory).rglob("*.py")
    )


def _takes_parameters(obj: object) -> bool:
    """Return whether calling ``obj`` accepts any parameter.

    For a class this is the signature of its constructor without ``self``.
    ``*args`` and ``**kwargs`` count as parameters, because numpy-style
    docstrings document them in the ``Parameters`` section as well.
    """
    if not callable(obj):
        return False
    return len(inspect.signature(obj).parameters) > 0


def _docstring_problem(obj: object) -> str | None:
    """Return why the docstring of ``obj`` breaks the rule, or None.

    ``obj.__doc__`` is read instead of :func:`inspect.getdoc`, which would
    fall back to the docstring of a base class.
    """
    doc = obj.__doc__
    if not isinstance(doc, str) or not doc.strip():
        return "no docstring"
    doc = inspect.cleandoc(doc)
    lines = doc.splitlines()
    if len(lines) > 1 and set(lines[1]) == {"-"}:
        return "no summary line before the first section"
    if _JAPANESE.search(doc):
        return "Japanese characters in the docstring"
    if _takes_parameters(obj) and not _PARAMETERS_SECTION.search(doc):
        return "takes parameters but has no 'Parameters' section"
    return None


def _identifiers(path: Path) -> set[str]:
    """Return the names that the code of ``path`` uses directly.

    Comments and string literals, docstrings included, are not code, so a
    name that is only mentioned there is not counted as used. Neither is an
    attribute name, the name right after a ``.``: ``logging.INFO`` uses the
    level of the standard library, not ``yikit.logging.INFO``.
    """
    with path.open(encoding="utf-8") as file:
        tokens = [
            token
            for token in tokenize.generate_tokens(file.readline)
            if token.type not in _LAYOUT_TOKENS
        ]
    return {
        token.string
        for previous, token in zip([None, *tokens], tokens)
        if token.type == tokenize.NAME
        and not (
            previous is not None
            and previous.type == tokenize.OP
            and previous.string == "."
        )
    }


def _japanese_in_comments_and_docstrings(path: Path) -> list[str]:
    """Return ``"<file>:<line>"`` of each Japanese comment or docstring."""
    source = path.read_text(encoding="utf-8")
    places = [
        f"{path.name}:{token.start[0]} (comment)"
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type == tokenize.COMMENT and _JAPANESE.search(token.string)
    ]
    for node in ast.walk(ast.parse(source)):
        if not isinstance(
            node,
            (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            continue
        doc = ast.get_docstring(node, clean=False)
        if doc is not None and _JAPANESE.search(doc):
            line = getattr(node, "lineno", 1)
            places.append(f"{path.name}:{line} (docstring)")
    return places


@pytest.mark.parametrize("package_name", SUBPACKAGES)
def test_public_names_have_numpy_docstrings(package_name):
    package = importlib.import_module(package_name)
    exempt = STDLIB_LOGGING_LEVELS if package_name == "yikit.logging" else ()
    found = {
        name: _docstring_problem(getattr(package, name))
        for name in _public_names(package_name)
        if name not in exempt
    }
    problems = {name: found[name] for name in found if found[name]}
    assert not problems, (
        f"Public names of {package_name} without a valid docstring: {problems}"
    )


@pytest.mark.parametrize("package_name", SUBPACKAGES)
def test_public_names_are_used_in_spec_tests(package_name):
    used = set().union(*(_identifiers(path) for path in SPEC_TEST_FILES))
    unused = [name for name in _public_names(package_name) if name not in used]
    assert not unused, (
        f"Public names of {package_name} not used in any of "
        f"{[path.name for path in SPEC_TEST_FILES]}: {unused}"
    )


@pytest.mark.parametrize("package_name", SUBPACKAGES)
def test_modules_have_english_docstrings(package_name):
    problems = {}
    for path in _package_sources(package_name):
        doc = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8")))
        if doc is None or not doc.strip():
            problems[path.name] = "no module docstring"
        elif doc.startswith("Copyright"):
            problems[path.name] = "module docstring is the license header"
        elif _JAPANESE.search(doc):
            problems[path.name] = "Japanese characters in the docstring"
    assert not problems, (
        f"Modules of {package_name} without an English module docstring: "
        f"{problems}"
    )


@pytest.mark.parametrize(
    "paths",
    [pytest.param(_package_sources(name), id=name) for name in SUBPACKAGES]
    + [
        pytest.param([TESTS_DIR / "conftest.py", *SPEC_TEST_FILES], id="tests")
    ],
)
def test_comments_and_docstrings_are_english(paths):
    places = [
        place
        for path in paths
        for place in _japanese_in_comments_and_docstrings(path)
    ]
    assert not places, f"Japanese comments or docstrings: {places}"
