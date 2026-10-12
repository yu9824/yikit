"""Tests of ``yikit.helpers`` and ``yikit.helpers._validation``."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

import yikit.helpers
from yikit.helpers import dummy_tqdm, is_argument, is_installed
from yikit.helpers._validation import feature_names_of

#: Small data shared by the tests of ``feature_names_of``.
X = np.arange(12, dtype=float).reshape(4, 3)
COLUMNS = ["a", "b", "c"]


def _frame(columns: list) -> pd.DataFrame:
    """Return ``X`` as a DataFrame whose column names are kept as given."""
    # ``dtype=object`` keeps the names as they are (pandas may otherwise
    # convert ``numpy.str_`` to ``str`` when it infers a string dtype).
    return pd.DataFrame(X, columns=pd.Index(columns, dtype=object))


# --- Public names ------------------------------------------------------------


def test_public_names_are_unchanged():
    assert tuple(yikit.helpers.__all__) == (
        "dummy_tqdm",
        "is_argument",
        "is_installed",
        "tqdm_joblib",
    )
    assert not hasattr(yikit.helpers, "feature_names_of")


# --- is_installed ------------------------------------------------------------


def test_is_installed_takes_one_package_name():
    # yikit.models (and the tests that replace it) rely on this signature.
    assert list(inspect.signature(is_installed).parameters) == ["package_name"]


@pytest.mark.parametrize("package_name", ["numpy", "sklearn", "yikit"])
def test_is_installed_returns_true_for_an_installed_package(package_name):
    assert is_installed(package_name) is True


def test_is_installed_returns_false_for_a_missing_package():
    assert is_installed("yikit_package_that_does_not_exist") is False


# --- is_argument -------------------------------------------------------------


def _function(alpha, beta=1, *args, gamma=None, **kwargs):
    """Take parameters of every kind (used by the tests of is_argument)."""


@pytest.mark.parametrize("arg_name", ["alpha", "beta", "gamma"])
def test_is_argument_returns_true_for_a_parameter(arg_name):
    assert is_argument(_function, arg_name) is True


@pytest.mark.parametrize("arg_name", ["delta", "Alpha", ""])
def test_is_argument_returns_false_for_a_missing_parameter(arg_name):
    # ``**kwargs`` does not make every name a parameter.
    assert is_argument(_function, arg_name) is False


def test_is_argument_inspects_classes_and_bound_methods():
    assert is_argument(LinearRegression, "fit_intercept") is True
    assert is_argument(LinearRegression().fit, "sample_weight") is True
    assert is_argument(LinearRegression().fit, "eval_set") is False
    # A bound method does not show ``self``.
    assert is_argument(LinearRegression().fit, "self") is False


# --- dummy_tqdm --------------------------------------------------------------


@pytest.mark.parametrize(
    ("make_iterable", "expected"),
    [
        (lambda: [3, 1, 2], [3, 1, 2]),
        (lambda: (3, 1, 2), [3, 1, 2]),
        (lambda: range(3), [0, 1, 2]),
        (lambda: (value for value in [3, 1, 2]), [3, 1, 2]),
        (lambda: iter([3, 1, 2]), [3, 1, 2]),
        (lambda: {"b": 1, "a": 2}, ["b", "a"]),
    ],
    ids=["list", "tuple", "range", "generator", "iterator", "dict"],
)
def test_dummy_tqdm_iterates_the_iterable_in_order(make_iterable, expected):
    assert list(dummy_tqdm(make_iterable())) == expected


def test_dummy_tqdm_ignores_the_arguments_of_tqdm():
    progress_bar = dummy_tqdm([1, 2], desc="fit", total=2, leave=False)

    assert list(progress_bar) == [1, 2]


@pytest.mark.parametrize(
    "progress_bar",
    [dummy_tqdm(), dummy_tqdm(total=3)],
    ids=["no-arguments", "total"],
)
def test_dummy_tqdm_without_an_iterable_is_empty(progress_bar):
    assert list(progress_bar) == []


@pytest.mark.parametrize(
    "method",
    ["update", "close", "refresh", "reset", "set_description", "set_postfix"],
)
def test_dummy_tqdm_accepts_calls_and_does_nothing(method):
    progress_bar = dummy_tqdm([1, 2, 3], total=3)

    assert getattr(progress_bar, method)() is None
    assert getattr(progress_bar, method)(1, n=2, desc="fit") is None
    # The calls do not change the iteration.
    assert list(progress_bar) == [1, 2, 3]


def test_dummy_tqdm_accepts_update_while_iterating():
    progress_bar = dummy_tqdm(["x", "y"])
    seen = []
    for value in progress_bar:
        progress_bar.update(1)
        seen.append(value)
    progress_bar.close()

    assert seen == ["x", "y"]


# --- feature_names_of --------------------------------------------------------


@pytest.mark.parametrize("version", ["1.0.0", "1.3.2", "1.9.1", "1.10.dev0"])
def test_feature_names_of_returns_str_column_names(monkeypatch, version):
    monkeypatch.setattr(sklearn, "__version__", version)

    names = feature_names_of(_frame(COLUMNS))

    assert isinstance(names, np.ndarray)
    assert names.dtype == object
    assert names.tolist() == COLUMNS
    assert all(type(name) is str for name in names)


@pytest.mark.parametrize("version", ["0.24.1", "0.24.2", "0.9.0"])
def test_feature_names_of_returns_none_before_sklearn_1_0(
    monkeypatch, version
):
    # scikit-learn < 1.0 has no ``feature_names_in_``.
    monkeypatch.setattr(sklearn, "__version__", version)

    assert feature_names_of(_frame(COLUMNS)) is None


@pytest.mark.parametrize(
    "columns",
    [
        [0, 1, 2],
        ["a", 1, "c"],
        # Only ``str`` itself counts (as in scikit-learn).
        [np.str_("a"), np.str_("b"), np.str_("c")],
        ["a", np.str_("b"), "c"],
    ],
    ids=["int", "mixed", "numpy-str", "str-and-numpy-str"],
)
def test_feature_names_of_returns_none_unless_all_names_are_str(
    monkeypatch, columns
):
    monkeypatch.setattr(sklearn, "__version__", "1.9.1")

    assert feature_names_of(_frame(columns)) is None


@pytest.mark.parametrize(
    "data",
    [X, X.tolist(), pd.DataFrame(X).to_numpy()],
    ids=["ndarray", "list", "ndarray-from-dataframe"],
)
def test_feature_names_of_returns_none_without_columns(monkeypatch, data):
    monkeypatch.setattr(sklearn, "__version__", "1.9.1")

    assert feature_names_of(data) is None


def test_feature_names_of_returns_none_for_no_columns(monkeypatch):
    monkeypatch.setattr(sklearn, "__version__", "1.9.1")

    assert feature_names_of(pd.DataFrame(index=range(3))) is None


def test_feature_names_of_raises_for_an_unreadable_sklearn_version(
    monkeypatch,
):
    monkeypatch.setattr(sklearn, "__version__", "unknown")

    with pytest.raises(ValueError, match="scikit-learn"):
        feature_names_of(_frame(COLUMNS))


@pytest.mark.parametrize(
    "data",
    [
        _frame(COLUMNS),
        _frame([0, 1, 2]),
        _frame([np.str_("a"), np.str_("b"), np.str_("c")]),
        X,
    ],
    ids=["str", "int", "numpy-str", "ndarray"],
)
def test_feature_names_of_agrees_with_the_installed_sklearn(data):
    # The installed scikit-learn (not a mocked version) sets the same
    # ``feature_names_in_``, or none at all.
    expected = getattr(StandardScaler().fit(data), "feature_names_in_", None)

    names = feature_names_of(data)

    if expected is None:
        assert names is None
    else:
        assert names is not None
        assert names.tolist() == expected.tolist()
