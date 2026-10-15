from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris

from yikit.feature_selection import FilterSelector

SEED = 334


def _chain_data() -> np.ndarray:
    """Return 7 features with only positive strong correlations.

    The pairs (0, 1), (1, 2) and (3, 4) are strongly correlated, while
    (0, 2) is not, so feature 1 (two pairs) is removed before the others,
    and feature 3 is removed from the tie of (3, 4).
    """
    rng = np.random.RandomState(SEED)
    n_samples = 50
    z = rng.normal(size=n_samples)
    w = rng.normal(size=n_samples)
    a = z
    b = z + 0.4 * rng.normal(size=n_samples)
    c = b + 0.4 * rng.normal(size=n_samples)
    d = w
    e = w + 0.1 * rng.normal(size=n_samples)
    f = rng.normal(size=n_samples)
    g = rng.normal(size=n_samples)
    return np.column_stack([a, b, c, d, e, f, g])


def _factor_data() -> np.ndarray:
    """Return 30 features made from 6 factors with positive loadings."""
    rng = np.random.RandomState(SEED)
    n_samples, n_factors, n_features = 40, 6, 30
    factors = rng.normal(size=(n_samples, n_factors))
    which = rng.randint(n_factors, size=n_features)
    noise = rng.uniform(0.05, 0.6, size=n_features)
    return factors[:, which] + noise * rng.normal(size=(n_samples, n_features))


def test_filter_method():
    X = load_iris().data

    selector = FilterSelector(verbose=0)
    selector.fit(X)
    assert np.all(selector.support_ == np.array([True, True, False, True]))


def test_filter_selector_removes_strong_negative_correlation():
    rng = np.random.RandomState(SEED)
    x = rng.normal(size=50)
    X = np.column_stack(
        [x, -x + 0.05 * rng.normal(size=50), rng.normal(size=50)]
    )

    selector = FilterSelector(verbose=False).fit(X)

    assert selector.corr_[0, 1, 0] < -0.9
    assert selector.corr_[0, 1, 1] < 0.05
    # As for positive correlations, the first feature of the tie is removed.
    assert selector.support_.tolist() == [False, True, True]
    assert selector.transform(X).shape == (50, 2)


# Recorded from the recursive implementation before the correlation was
# judged by its absolute value. Without strong negative correlations the
# selected features must not change.
@pytest.mark.parametrize(
    "make_data, r, expected",
    [
        (_chain_data, 0.9, [1, 0, 1, 0, 1, 1, 1]),
        (
            _factor_data,
            0.8,
            [0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1]
            + [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1],
        ),
        (
            _factor_data,
            0.9,
            [1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1]
            + [1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1],
        ),
        (
            _factor_data,
            0.95,
            [1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1]
            + [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1],
        ),
    ],
)
def test_filter_selector_positive_correlation_keeps_previous_result(
    make_data, r, expected
):
    X = make_data()
    # The data must have no strong negative correlation.
    assert not np.any(np.corrcoef(X, rowvar=False) < -r)

    selector = FilterSelector(r=r, verbose=False).fit(X)

    assert selector.support_.astype(int).tolist() == expected


@pytest.mark.parametrize("verbose, n_jobs", [(True, None), (True, 2)])
def test_filter_selector_same_result_with_progress_and_jobs(verbose, n_jobs):
    X = _factor_data()
    expected = FilterSelector(verbose=False).fit(X)

    selector = FilterSelector(verbose=verbose, n_jobs=n_jobs).fit(X)

    np.testing.assert_array_equal(selector.support_, expected.support_)
    np.testing.assert_array_equal(selector.corr_, expected.corr_)


def test_filter_selector_feature_names_out_dataframe():
    X = pd.DataFrame(_chain_data(), columns=list("abcdefg"))

    selector = FilterSelector(verbose=False).fit(X)

    assert selector.n_features_in_ == 7
    assert selector.feature_names_in_.tolist() == list("abcdefg")
    assert selector.get_feature_names_out().tolist() == [
        "a",
        "c",
        "e",
        "f",
        "g",
    ]
    assert selector.transform(X).shape == (50, 5)


def test_filter_selector_feature_names_out_ndarray():
    X = _chain_data()
    selector = FilterSelector(verbose=False)

    # Refitting on an ndarray drops the names learned from a DataFrame.
    selector.fit(pd.DataFrame(X, columns=list("abcdefg")))
    selector.fit(X)

    assert selector.n_features_in_ == 7
    assert not hasattr(selector, "feature_names_in_")
    assert selector.get_feature_names_out().tolist() == [
        "x0",
        "x2",
        "x4",
        "x5",
        "x6",
    ]


def test_filter_selector_n_features_in():
    selector = FilterSelector(verbose=False).fit(_chain_data())

    assert selector.n_features_in_ == 7


def test_filter_selector_transform_different_n_features_raises():
    X = _chain_data()
    selector = FilterSelector(verbose=False).fit(X)

    with pytest.raises(ValueError):
        selector.transform(X[:, :-1])
    with pytest.raises(ValueError):
        selector.transform(np.hstack([X, X[:, :1]]))


def test_filter_selector_many_removals_without_recursion_error():
    # 1100 pairs of identical columns: one feature is removed per pair, so
    # there are more removals than the default recursion limit (1000).
    n_pairs = 1100
    rng = np.random.RandomState(SEED)
    X = np.repeat(rng.normal(size=(30, n_pairs)), 2, axis=1)

    start = time.perf_counter()
    selector = FilterSelector(verbose=False).fit(X)
    elapsed = time.perf_counter() - start

    assert selector.support_.tolist() == [False, True] * n_pairs
    # Generous bound; this takes well under a second on a laptop.
    assert elapsed < 30


@pytest.mark.parametrize(
    "X, expected",
    [
        (
            # Columns 0 to 2 are constant; 0.1 is not exactly its own mean.
            np.column_stack(
                [
                    np.ones(30),
                    np.full(30, 0.1),
                    np.full(30, 0.1),
                    np.arange(30.0),
                    np.arange(30.0) ** 1.01,
                    np.cos(np.arange(30.0)),
                ]
            ),
            [True, True, True, False, True, True],
        ),
        # With a single sample, every column is constant.
        (np.array([[1.0, 2.0, 3.0]]), [True, True, True]),
    ],
)
def test_filter_selector_constant_columns_without_warnings(X, expected):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        selector = FilterSelector(verbose=False).fit(X)

    assert selector.support_.tolist() == expected
    constant = np.all(X == X[0], axis=0)
    off_diagonal = ~np.eye(X.shape[1], dtype=bool)
    with_constant = (constant[:, None] | constant[None, :]) & off_diagonal
    # Pairs with a constant feature are treated as uncorrelated.
    assert np.all(selector.corr_[with_constant, 0] == 0.0)
    assert np.all(selector.corr_[with_constant, 1] == 1.0)
