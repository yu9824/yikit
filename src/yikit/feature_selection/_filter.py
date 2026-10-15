"""Filter method of feature selection: :class:`FilterSelector`.

The selector removes features that are strongly correlated with other
features, judging from their Pearson correlations alone (no estimator).

Copyright © 2021 yu9824

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import pearsonr
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from yikit.helpers import is_installed, tqdm_joblib
from yikit.helpers._validation import feature_names_of
from yikit.logging import get_child_logger

logger = get_child_logger(__name__)


class FilterSelector(SelectorMixin, BaseEstimator):
    """Remove features that are strongly correlated with other features.

    A pair of features is strongly correlated when the absolute value of
    their Pearson correlation coefficient is larger than ``r`` and its
    p-value is smaller than ``alpha``, so a strong negative correlation
    counts as well as a strong positive one. The feature in the largest
    number of strongly correlated pairs is removed (the first one on ties),
    and this is repeated with the remaining features until no strongly
    correlated pair is left.

    A constant feature is treated as uncorrelated with every other feature,
    so it is never removed by this selector.

    Parameters
    ----------
    r : float, default=0.9
        The threshold of the absolute value of the correlation coefficient.
        Pairs whose absolute correlation is larger than this are tested for
        significance and may be removed.
    alpha : float, default=0.05
        The significance level. The correlation of a pair is significant
        when its p-value is smaller than this. It is recommended not to
        change it without a specific reason.
    verbose : bool or int, default=True
        If positive, show the progress of computing the p-values: a tqdm
        progress bar when tqdm is installed, otherwise log messages.
    n_jobs : int or None, default=None
        The number of jobs to compute the p-values in parallel with joblib.
        None means 1 unless in a ``joblib.parallel_backend`` context; -1
        means all the processors.

    Attributes
    ----------
    corr_ : numpy.ndarray of shape (n_features_in_, n_features_in_, 2)
        ``corr_[i, j, 0]`` is the Pearson correlation coefficient of the
        features ``i`` and ``j`` and ``corr_[i, j, 1]`` is its p-value. The
        p-values are computed only for the pairs whose absolute correlation
        is larger than ``r``; the others are 1.0. The diagonal holds 1.0
        and 0.0, and a pair with a constant feature holds 0.0 and 1.0.
    support_ : numpy.ndarray of shape (n_features_in_,)
        The boolean mask of the selected (kept) features.
    n_features_in_ : int
        The number of features seen during ``fit``.
    feature_names_in_ : numpy.ndarray of shape (n_features_in_,)
        The names of the features seen during ``fit``. Defined only when
        ``X`` has feature names that are all strings.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from yikit.feature_selection import FilterSelector
    >>> rng = np.random.RandomState(334)
    >>> x = rng.normal(size=50)
    >>> X = pd.DataFrame(
    ...     {
    ...         "a": x,
    ...         "b": x + 0.1 * rng.normal(size=50),
    ...         "c": -x + 0.1 * rng.normal(size=50),
    ...         "d": rng.normal(size=50),
    ...     }
    ... )

    ``a``, ``b`` and ``c`` are strongly correlated with each other (``c``
    negatively), so only one of them is kept.

    >>> selector = FilterSelector(verbose=False).fit(X)
    >>> selector.get_support()
    array([False, False,  True,  True])
    >>> selector.get_feature_names_out()
    array(['c', 'd'], dtype=object)
    >>> selector.transform(X).shape
    (50, 2)
    """

    def __init__(
        self,
        r: float = 0.9,
        alpha: float = 0.05,
        verbose: int | bool = True,
        n_jobs: int | None = None,
    ):
        self.r = r
        self.alpha = alpha
        self.verbose = verbose
        self.n_jobs = n_jobs

    def fit(self, X: Any, y: Any = None) -> FilterSelector:
        """Find the features to keep.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The feature matrix, such as a pandas DataFrame or a 2D numpy
            array.
        y : None
            Ignored. Present for API consistency.

        Returns
        -------
        self : FilterSelector
            The fitted selector.
        """
        feature_names = feature_names_of(X)
        X_checked = check_array(X)
        n_features = X_checked.shape[1]

        use_tqdm = is_installed("tqdm") and self.verbose > 0
        use_logging = not is_installed("tqdm") and self.verbose > 0

        corr_matrix = _correlation_matrix(X_checked)
        # (i, j) with i < j, in row-major order.
        pairs = np.argwhere(np.triu(np.abs(corr_matrix) > self.r, k=1))

        if len(pairs) > 0:
            tasks = (
                delayed(_pearson_p_value)(X_checked[:, i], X_checked[:, j])
                for i, j in pairs
            )
            parallel = Parallel(n_jobs=self.n_jobs, verbose=0)
            if use_tqdm:
                with tqdm_joblib(
                    total=len(pairs), desc="computing p-values", silent=False
                ):
                    p_values = np.array(parallel(tasks), dtype=float)
            else:
                p_values = np.array(parallel(tasks), dtype=float)
        else:
            p_values = np.empty(0, dtype=float)
            if use_logging:
                logger.info(
                    "No pairs with an absolute correlation above the "
                    "threshold."
                )

        if use_logging:
            logger.info(
                f"Finished! Processed {n_features} features, "
                f"computed {len(pairs)} p-values."
            )

        corr = np.empty((n_features, n_features, 2), dtype=float)
        corr[:, :, 0] = corr_matrix
        corr[:, :, 1] = 1.0
        diagonal = np.arange(n_features)
        corr[diagonal, diagonal, 1] = 0.0
        corr[pairs[:, 0], pairs[:, 1], 1] = p_values
        corr[pairs[:, 1], pairs[:, 0], 1] = p_values

        support = _remove_correlated(pairs[p_values < self.alpha], n_features)

        self.corr_ = corr
        self.support_ = support
        self.n_features_in_ = n_features
        if feature_names is not None:
            self.feature_names_in_ = feature_names
        elif hasattr(self, "feature_names_in_"):
            # Do not keep the names of a previous fit.
            del self.feature_names_in_
        return self

    def _get_support_mask(self) -> np.ndarray:
        """Get the boolean mask of the selected features.

        Returns
        -------
        numpy.ndarray of shape (n_features_in_,)
            The mask; True means that the feature is selected.
        """
        check_is_fitted(self, "support_")
        return self.support_


def _correlation_matrix(X: np.ndarray) -> np.ndarray:
    """Return the Pearson correlation matrix of the columns of ``X``.

    Constant columns are found explicitly (with ``==``, like
    ``scipy.stats.pearsonr``) and get a correlation of 0.0 with every other
    column, because their correlation is undefined and ``numpy.corrcoef``
    returns NaN, or a meaningless value when the mean is rounded, for them.

    Parameters
    ----------
    X : numpy.ndarray of shape (n_samples, n_features)
        The checked feature matrix.

    Returns
    -------
    numpy.ndarray of shape (n_features, n_features)
        The correlation matrix with 1.0 on the diagonal.
    """
    n_features = X.shape[1]
    constant = np.all(X == X[0], axis=0)
    if np.all(constant):
        # Also covers a single sample, for which numpy.corrcoef warns.
        return np.eye(n_features)
    # There are at least two samples here. Dividing by a zero standard
    # deviation (of a constant column, or one whose variance underflows)
    # gives NaN, which is treated as no correlation.
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = np.atleast_2d(np.corrcoef(X, rowvar=False))
    corr = np.where(np.isnan(corr), 0.0, corr)
    corr[constant, :] = 0.0
    corr[:, constant] = 0.0
    np.fill_diagonal(corr, 1.0)
    return corr


def _pearson_p_value(x: np.ndarray, y: np.ndarray) -> float:
    """Return the p-value of the Pearson correlation of ``x`` and ``y``.

    Parameters
    ----------
    x, y : numpy.ndarray of shape (n_samples,)
        Two non-constant features.

    Returns
    -------
    float
        The two-sided p-value of ``scipy.stats.pearsonr``.
    """
    return float(pearsonr(x, y)[1])


def _remove_correlated(pairs: np.ndarray, n_features: int) -> np.ndarray:
    """Remove features until no strongly correlated pair is left.

    The feature in the largest number of the remaining pairs is removed
    (the one with the smallest index on ties), and its pairs are dropped.
    This is repeated until no pair is left. A loop is used instead of
    recursion so that any number of features can be removed.

    Parameters
    ----------
    pairs : numpy.ndarray of shape (n_pairs, 2)
        The indices of the strongly correlated pairs of features.
    n_features : int
        The number of features.

    Returns
    -------
    numpy.ndarray of shape (n_features,)
        The boolean mask of the kept features.
    """
    neighbors: list[set[int]] = [set() for _ in range(n_features)]
    for i, j in pairs.tolist():
        neighbors[i].add(j)
        neighbors[j].add(i)
    degrees = np.array([len(s) for s in neighbors], dtype=int)
    support = np.ones(n_features, dtype=bool)
    while np.any(degrees > 0):
        i_max = int(np.argmax(degrees))
        for j in neighbors[i_max]:
            neighbors[j].discard(i_max)
            degrees[j] -= 1
        neighbors[i_max] = set()
        degrees[i_max] = 0
        support[i_max] = False
    return support
