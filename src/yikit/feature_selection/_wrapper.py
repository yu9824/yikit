"""
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

from array import array
from typing import Literal

import boruta
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import pearsonr
from sklearn.cross_decomposition import PLSRegression
from sklearn.utils import check_random_state, check_X_y, shuffle
from sklearn.utils.validation import check_is_fitted

from yikit.helpers import is_installed, tqdm_joblib
from yikit.logging import get_child_logger

if is_installed("tqdm"):
    from tqdm.auto import tqdm
else:
    from yikit.helpers import dummy_tqdm as tqdm


logger = get_child_logger(__name__)


def _shuffled_pearsonrs(
    X: np.ndarray, y: np.ndarray, seed: int
) -> array[float]:
    """Return the Pearson correlations of the shuffled features with ``y``.

    The samples (rows) of ``X`` are shuffled with a RandomState made from
    ``seed``, so a job of ``BorutaPy._calc_auto_perc`` needs only the seed.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        The input samples.
    y : ndarray of shape (n_samples,)
        The target values.
    seed : int
        The seed of the shuffle.

    Returns
    -------
    array.array of shape (n_features,)
        The correlation of each shuffled feature with ``y`` (float32).
    """
    X_shuffled = np.asarray(shuffle(X, random_state=seed))
    return array(
        "f",
        [pearsonr(X_shuffled[:, i], y)[0] for i in range(X_shuffled.shape[1])],
    )


#: Fitted attributes that only a fit with ``perc="auto"`` sets.
_AUTO_PERC_ATTRIBUTES = ("pears_", "r_ccmax_")


class BorutaPy(boruta.BorutaPy):
    """Boruta all-relevant feature selection with an automatic percentile.

    This class extends ``boruta.BorutaPy`` of scikit-learn-contrib/boruta_py
    (BSD 3-Clause License), and this docstring is adapted from the one of
    boruta. Boruta is an all-relevant feature selection method: it tries to
    find all the features that carry information usable for prediction,
    rather than a compact subset on which a model has a minimal error. In
    each iteration, ``estimator`` is fitted on the features and on their
    shuffled copies (the shadow features), and a feature scores a hit when
    its importance is above the ``perc`` percentile of the importances of
    the shadow features. The features with significantly many hits are
    confirmed, and those with significantly few hits are rejected. With
    ``two_step=True``, the p-values are corrected with the Benjamini
    Hochberg FDR across the features still in the selection in each
    iteration, and then with Bonferroni across the iterations.

    Compared with boruta, this class adds:

    - ``perc="auto"`` (the default), which sets the percentile from the
      largest Pearson correlation between shuffled features and the target
      (see ``perc`` and ``max_shuf``), in both ``fit`` and
      ``fit_transform`` (and therefore in a
      :class:`~sklearn.pipeline.Pipeline`);
    - ``n_estimators="auto"`` as the default;
    - a progress bar (tqdm, when installed) and the logger of yikit instead
      of ``print`` (see ``verbose``);
    - ``transform`` of a :class:`pandas.DataFrame` without ``return_df``,
      which returns a NumPy array of the selected columns.

    It follows the conventions of scikit-learn: the parameters are stored
    as passed (``get_params`` returns them, also after a fit), and they are
    read in every fit. Pruned trees with a depth of 3 to 7 are recommended
    as ``estimator``.

    Parameters
    ----------
    estimator : object
        A supervised estimator with a ``fit`` method that sets
        ``feature_importances_``. Important features must have high absolute
        values in ``feature_importances_``. boruta sets ``n_estimators``
        and ``random_state`` of this estimator in each iteration.
    n_estimators : int or "auto", default="auto"
        The number of estimators of ``estimator``. If "auto", it is set in
        each iteration from the number of features that are not rejected
        and the ``max_depth`` of ``estimator`` (10 when it is None), so
        that fewer trees are used as features are rejected.
    perc : float or "auto", default="auto"
        The percentile of the importances of the shadow features that a
        feature must exceed to score a hit. 100 is the maximum, as in the
        original Boruta; it tends to be too strict. A lower ``perc``
        selects more features, including more irrelevant ones. If "auto",
        it is calculated from the data in every fit as
        ``100 * (1 - r_ccmax_)``, where ``r_ccmax_`` is the largest Pearson
        correlation between randomly shuffled features and the target (see
        ``max_shuf``). The value used is stored in ``perc_``; this
        parameter is never overwritten, so ``get_params`` and
        :func:`~sklearn.base.clone` keep "auto".
    alpha : float, default=0.05
        The level at which the corrected p-values are rejected in both
        correction steps.
    two_step : bool, default=True
        Whether to use the two-step correction. If False, only the
        Bonferroni correction of the original Boruta is used.
    max_iter : int, default=100
        The maximum number of iterations.
    random_state : int, RandomState instance or None, default=None
        Controls the shuffles of ``perc="auto"``, the shuffles of the
        shadow features and the ``random_state`` given to ``estimator``.
        The value is stored as passed, and each fit makes its random number
        generator from it with :func:`~sklearn.utils.check_random_state`.
        Pass an int to get the same result in every fit (also of a clone).
        A RandomState instance is used and advanced, and None uses the
        global random number generator of NumPy.
    verbose : int, default=1
        Controls what is shown during a fit. It is read at the start of
        each fit, so ``set_params(verbose=...)`` applies to the next fit.

        - 0: nothing.
        - 1: a progress bar of the iterations (and of the shuffles of
          ``perc="auto"``) when tqdm is installed, otherwise the iteration
          number is logged. The value of ``perc="auto"`` and a summary at
          the end are logged.
        - 2: as 1, and the numbers of confirmed, tentative and rejected
          features are logged in each iteration.
    max_shuf : int, default=10000
        The number of shuffles used to calculate ``r_ccmax_`` when ``perc``
        is "auto". Each shuffle has its own seed, drawn from
        ``random_state`` before the shuffles are split among the jobs, so
        ``perc_`` does not depend on ``n_jobs``.
    n_jobs : int or None, default=None
        The number of jobs that run the shuffles of ``perc="auto"`` in
        parallel with joblib. None means 1 unless in a
        ``joblib.parallel_backend`` context, and -1 means all processors.

    Attributes
    ----------
    n_features_ : int
        The number of selected (confirmed) features.
    support_ : ndarray of shape (n_features,)
        The mask of the selected features; only the confirmed ones are
        True.
    support_weak_ : ndarray of shape (n_features,)
        The mask of the tentative features, which have not gained enough
        support during the ``max_iter`` iterations.
    ranking_ : ndarray of shape (n_features,)
        The ranking of the features: 1 for the confirmed features, 2 for
        the tentative ones, and from 3 (2 when there are no tentative
        features) for the rejected ones, based on their importances through
        the iterations.
    importance_history_ : ndarray of shape (n_iterations + 1, n_features)
        The importances of the features in each iteration, after a first
        row of zeros. A feature has NaN in the iterations after it was
        rejected.
    perc_ : float
        The percentile used in the last fit: the value calculated from the
        data when ``perc="auto"``, otherwise ``perc`` itself.
    r_ccmax_ : float
        The largest Pearson correlation between the shuffled features and
        the target. Only set by a fit with ``perc="auto"``.
    pears_ : list of array.array of shape (n_features,)
        The Pearson correlations between the shuffled features and the
        target, one array for each of the ``max_shuf`` shuffles. Only set
        by a fit with ``perc="auto"``.

    See Also
    --------
    FilterSelector : Remove features that are highly correlated with
        another feature.

    References
    ----------
    Kursa M., Rudnicki W., "Feature Selection with the Boruta Package",
    Journal of Statistical Software, Vol. 36, Issue 11, Sep 2010.
    https://doi.org/10.18637/jss.v036.i11

    scikit-learn-contrib/boruta_py:
    https://github.com/scikit-learn-contrib/boruta_py

    Examples
    --------
    The first 3 of the 6 features are informative. The small ``max_iter``
    and ``max_shuf`` only keep this example fast:

    >>> from sklearn.datasets import make_regression
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from yikit.feature_selection import BorutaPy
    >>> X, y = make_regression(
    ...     n_samples=100,
    ...     n_features=6,
    ...     n_informative=3,
    ...     shuffle=False,
    ...     random_state=334,
    ... )
    >>> selector = BorutaPy(
    ...     RandomForestRegressor(max_depth=5, n_jobs=1, random_state=334),
    ...     max_iter=20,
    ...     max_shuf=100,
    ...     verbose=0,
    ...     random_state=334,
    ... )
    >>> X_selected = selector.fit_transform(X, y)
    >>> selector.support_
    array([ True,  True,  True, False, False, False])
    >>> X_selected.shape
    (100, 3)

    ``perc="auto"`` is resolved in ``perc_``, and the parameters keep the
    passed values, so a fit of a clone gives the same result:

    >>> from sklearn.base import clone
    >>> selector.get_params()["perc"], selector.get_params()["random_state"]
    ('auto', 334)
    >>> bool(0 < selector.perc_ < 100)
    True
    >>> bool(clone(selector).fit(X, y).perc_ == selector.perc_)
    True
    """

    def __init__(
        self,
        estimator,
        n_estimators: int | Literal["auto"] = "auto",
        perc: float | Literal["auto"] = "auto",
        alpha: float = 0.05,
        two_step: bool = True,
        max_iter: int = 100,
        random_state: np.random.RandomState | int | None = None,
        verbose: int = 1,
        max_shuf: int = 10000,
        n_jobs: int | None = None,
    ):
        self.max_shuf = max_shuf
        self.n_jobs = n_jobs
        super().__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            perc=perc,
            alpha=alpha,
            two_step=two_step,
            max_iter=max_iter,
            random_state=random_state,
            verbose=verbose,
        )

    def fit(self, X, y):
        """Fit the Boruta feature selection with ``estimator``.

        This docstring is adapted from the one of
        scikit-learn-contrib/boruta_py (BSD 3-Clause License).

        ``perc="auto"`` is resolved in ``_fit``, the same path that
        ``fit_transform`` takes. The resolved value is stored in ``perc_``
        and the ``perc`` parameter is left unchanged.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        self : BorutaPy
            The fitted selector.
        """
        return self._fit(X, y)

    def _fit(self, X, y):
        """Resolve ``perc`` and run the Boruta feature selection of boruta.

        boruta calls this method from both ``fit`` and ``fit_transform``, so
        ``perc="auto"`` is resolved on both paths, including inside a
        :class:`~sklearn.pipeline.Pipeline`. The resolved percentile is
        stored in ``perc_``. The display settings are read from ``verbose``
        and a random number generator is made from ``random_state`` in
        every fit. boruta reads ``self.perc`` and ``self.random_state``
        (and replaces ``self.random_state`` with a RandomState), so they
        hold ``perc_`` and the generator of this fit only while boruta's
        ``_fit`` runs. The passed values (for example "auto" and an int)
        are restored afterwards, and the progress bar is closed, also when
        an error is raised.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        self : BorutaPy
            The fitted selector.
        """
        X, y = check_X_y(X, y)
        # Decide on progress display based on availability of tqdm and
        # verbosity
        self._use_tqdm = is_installed("tqdm") and self.verbose > 0
        self._use_logging = (not self._use_tqdm) or self.verbose > 1

        random_state = check_random_state(self.random_state)
        if self.perc == "auto":
            self.perc_ = self._calc_auto_perc(X, y, random_state)
        else:
            self.perc_ = self.perc
            for name in _AUTO_PERC_ATTRIBUTES:
                if hasattr(self, name):  # left by a fit with perc="auto"
                    delattr(self, name)

        if self._use_tqdm:
            self._pbar = tqdm(total=self.max_iter, desc="BorutaPy")
        passed_perc, passed_random_state = self.perc, self.random_state
        self.perc, self.random_state = self.perc_, random_state
        try:
            return super()._fit(X, y)
        finally:
            self.perc, self.random_state = passed_perc, passed_random_state
            if self._use_tqdm:
                self._pbar.close()
                # A progress bar may not be picklable.
                del self._pbar

    def _transform(self, X, weak=False, return_df=False):
        """Reduce ``X`` to the selected features, also for a DataFrame.

        boruta calls this method from both ``transform`` and
        ``fit_transform``. Without ``return_df``, boruta selects the columns
        with ``X[:, mask]``, which a :class:`pandas.DataFrame` does not
        support, so a DataFrame is converted to a NumPy array first (as
        ``fit`` does). This is what a :class:`~sklearn.pipeline.Pipeline`
        fitted on a DataFrame calls. With ``return_df=True``, ``X`` is
        passed to boruta as it is and the selected columns of the DataFrame
        are returned.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The input samples.
        weak : bool, default=False
            If True, the tentative features are also selected.
        return_df : bool, default=False
            If True and ``X`` is a DataFrame, return a DataFrame.

        Returns
        -------
        X : array-like, shape = [n_samples, n_features_]
            ``X`` reduced to the selected features: a NumPy array for a
            DataFrame without ``return_df``, otherwise the type of boruta.
        """
        if return_df:
            return super()._transform(X, weak, return_df)
        if hasattr(X, "iloc"):  # pandas.DataFrame
            X = np.asarray(X)
        # Without return_df, which boruta < 0.4 does not take.
        return super()._transform(X, weak)

    def get_support(self, weak: bool = False) -> np.ndarray:
        """Get a mask, or integer index, of the features selected.

        Parameters
        ----------
        weak : bool, optional
            If set to True, the tentative features are also included in the support mask.
            If False, only confirmed features are included. By default False.

        Returns
        -------
        support : array of shape [n_features]
            A boolean mask of the features selected. If `weak=True`, includes both
            confirmed and tentative features. If `weak=False`, includes only confirmed
            features.
        """
        check_is_fitted(self, "support_")
        if weak:
            return self.support_weak_
        else:
            return self.support_

    def _calc_auto_perc(
        self, X: np.ndarray, y: np.ndarray, random_state: np.random.RandomState
    ) -> float:
        """Calculate ``perc`` from the data when it is "auto".

        This docstring is based on the one of scikit-learn-contrib/boruta_py
        (BSD 3-Clause License).

        The percentile is set from the largest Pearson correlation that
        occurs by chance between the features and the target: the samples
        of ``X`` are shuffled ``max_shuf`` times, and the correlations of
        the shuffled features with ``y`` are stored in ``pears_`` and their
        maximum in ``r_ccmax_``. One seed for each shuffle is drawn from
        ``random_state`` before the shuffles are split among the
        ``n_jobs`` jobs, and each job makes its own RandomState from the
        seed. So the result does not depend on ``n_jobs``, and no two jobs
        repeat the same shuffle.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The training input samples.
        y : ndarray of shape (n_samples,)
            The target values.
        random_state : RandomState
            The random number generator of this fit, which the seeds are
            drawn from.

        Returns
        -------
        float
            The percentile, ``100 * (1 - r_ccmax_)``.
        """
        seeds = random_state.randint(
            np.iinfo(np.int32).max, size=self.max_shuf
        )
        # Calculate correlations with randomly shuffled features to determine
        # the maximum spurious correlation that can occur by chance.
        with tqdm_joblib(
            total=self.max_shuf, desc="Calc r_ccmax", silent=not self._use_tqdm
        ):
            parallel = Parallel(n_jobs=self.n_jobs, verbose=0)
            self.pears_ = parallel(
                delayed(_shuffled_pearsonrs)(X, y, int(seed)) for seed in seeds
            )

        self.r_ccmax_ = np.nanmax(self.pears_, axis=None).item()
        perc = 100 * (1 - self.r_ccmax_)
        if self.verbose > 0:
            logger.info(f"Assigned perc = {perc:.1f}\n")
        return perc

    def _print_results(self, dec_reg, _iter, flag):
        """Print the results of the Boruta feature selection process.

        This method displays the current status of feature selection, including
        the number of confirmed, tentative, and rejected features at each iteration.
        The output format depends on the verbosity level and whether tqdm is available.

        Parameters
        ----------
        dec_reg : array-like
            Decision registry array indicating the status of each feature:
            - 1: confirmed features
            - 0: tentative features (still under consideration)
            - -1: rejected features
        _iter : int
            Current iteration number.
        flag : int
            Status flag:
            - 0: feature selection is still in progress
            - non-zero: Boruta has finished running and tentative features have been filtered
        """
        n_iter = str(_iter) + " / " + str(self.max_iter)
        n_confirmed = np.where(dec_reg == 1)[0].shape[0]
        n_rejected = np.where(dec_reg == -1)[0].shape[0]
        cols = ["Iteration: ", "Confirmed: ", "Tentative: ", "Rejected: "]

        # still in feature selection
        if flag == 0:
            n_tentative = np.where(dec_reg == 0)[0].shape[0]
            content = map(str, [n_iter, n_confirmed, n_tentative, n_rejected])
            if self._use_tqdm:
                self._pbar.update()
                output = None

            if self.verbose == 1:
                output = cols[0] + n_iter
            elif self.verbose > 1:
                output = "\n" + "\n".join(
                    [x[0] + "\t" + x[1] for x in zip(cols, content)]
                )

        # Boruta finished running and tentatives have been filtered
        else:
            n_tentative = np.sum(self.support_weak_)
            n_rejected = np.sum(~(self.support_ | self.support_weak_))
            content = map(str, [n_iter, n_confirmed, n_tentative, n_rejected])
            result = "\n".join(
                [x[0] + "\t" + x[1] for x in zip(cols, content)]
            )
            output = "\n\nBorutaPy finished running.\n\n" + result
            if self._use_tqdm:
                self._pbar.update(self.max_iter - _iter + 1)
                self._pbar.close()

        if (
            flag != 0  # finished
        ) or self._use_logging:
            logger.info(output + "\n")


def calc_vip(estimator: PLSRegression) -> np.ndarray:
    """Calculate VIP (The Variable Importance in Projection)

    References
    - Mukherjee, R., Sengupta, D., & Sikdar, S. K. (2015). Selection of Sustainable Processes Using Sustainability Footprint Method: A Case Study of Methanol Production from Carbon Dioxide. In Computer Aided Chemical Engineering (Vol. 36, pp. 311-329). Elsevier. DOI: [10.1016/B978-0-444-63472-6.00012-4](https://doi.org/10.1016/B978-0-444-63472-6.00012-4)
    - [Variable Importance in Projection](https://www.sciencedirect.com/topics/engineering/variable-importance-in-projection)

    Parameters
    ----------
    estimator : PLSRegression
        fitted pls regression model

    Returns
    -------
    np.ndarray
        VIP of each variable (n_features,)
    """
    # scores: latent variables, principal components
    t = estimator.x_scores_  # (n_samples, n_components)
    w = estimator.x_weights_  # (n_features, n_components)
    # loadings: factor loadings
    q = estimator.y_loadings_  # (n_targets, n_components)
    p, h = w.shape
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    return np.sqrt(
        p * (np.square(w / np.linalg.norm(w, axis=0)) @ s).ravel() / np.sum(s)
    )
