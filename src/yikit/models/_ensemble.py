"""Ensemble of regressors built on the ensembles of scikit-learn.

``EnsembleRegressor`` combines regressors with ``VotingRegressor`` (the
average of their predictions) or with ``StackingRegressor`` (a linear model
fitted on their cross-validated predictions), and can tune each regressor
with optuna (``OptunaSearchCV`` of optuna-integration) before combining
them. The modules of the tuning are imported only when the tuning is used,
so that ``yikit.models`` can be imported without optuna.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone, is_regressor
from sklearn.ensemble import (
    RandomForestRegressor,
    StackingRegressor,
    VotingRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from numpy.typing import ArrayLike, NDArray
    from sklearn.model_selection import BaseCrossValidator

#: The values of ``method`` (the ways to combine the models).
METHODS = ("average", "stacking", "blending")

#: Fitted attributes that only some fits define.
_OPTIONAL_FITTED_ATTRIBUTES = ("final_estimator_", "feature_names_in_")


def _is_named_pair(item: Any) -> bool:
    """Return whether ``item`` is a ``(name, estimator)`` pair."""
    return (
        isinstance(item, (tuple, list))
        and len(item) == 2
        and isinstance(item[0], str)
    )


def _named_estimators(estimators: Iterable[Any]) -> list[tuple[str, Any]]:
    """Return the models of ``estimators`` as ``(name, estimator)`` pairs.

    A ``(name, estimator)`` pair is kept as it is. A model without a name is
    named after its class in lower case, and the models that share a name
    get ``-1``, ``-2``, ... in their order, as in
    ``sklearn.pipeline.make_pipeline``.

    Parameters
    ----------
    estimators : iterable
        Models and ``(name, estimator)`` pairs.

    Returns
    -------
    list of (str, estimator)
        The named models, in the order of ``estimators``.
    """
    items = list(estimators)
    counts = Counter(
        type(item).__name__.lower()
        for item in items
        if not _is_named_pair(item)
    )
    numbers: Counter[str] = Counter()
    named = []
    for item in items:
        if _is_named_pair(item):
            named.append((item[0], item[1]))
            continue
        name = type(item).__name__.lower()
        if counts[name] > 1:
            numbers[name] += 1
            name = f"{name}-{numbers[name]}"
        named.append((name, item))
    return named


def _check_regressor(name: str, estimator: Any) -> None:
    """Raise a ``ValueError`` if ``estimator`` is not a regressor.

    A ``Pipeline`` is a regressor when its last step is one.

    Raises
    ------
    ValueError
        If ``estimator`` is not a regressor, with its name and type.
    """
    message = (
        f"The estimator {name!r} ({type(estimator).__name__}) is not a "
        "regressor; EnsembleRegressor combines regressors only."
    )
    try:
        regressor = is_regressor(estimator)
    except AttributeError as exc:  # not an estimator (scikit-learn >= 1.6)
        raise ValueError(message) from exc
    if not regressor:
        raise ValueError(message)


def _n_features(X: Any) -> int:
    """Return the number of columns of ``X``.

    Raises
    ------
    ValueError
        If ``X`` is not 2-dimensional.
    """
    shape = np.shape(X)
    if len(shape) != 2:
        raise ValueError(
            f"X must be 2-dimensional (n_samples, n_features), but its shape "
            f"is {shape}."
        )
    return int(shape[1])


def _import_search_regressor() -> tuple[Any, Any]:
    """Import the classes of the tuning (they need optuna).

    Returns
    -------
    tuple
        ``(ParamDistributions, OptunaSearchRegressor)``.
    """
    from yikit.models._optuna import ParamDistributions
    from yikit.models._search_cv import OptunaSearchRegressor

    return ParamDistributions, OptunaSearchRegressor


def _build_ensemble(
    method: str,
    estimators: list[tuple[str, Any]],
    cv: Any,
    n_jobs: int | None,
    verbose: int,
) -> VotingRegressor | StackingRegressor:
    """Build the unfitted scikit-learn ensemble of ``method``."""
    if method == "average":
        return VotingRegressor(
            estimators, n_jobs=n_jobs, verbose=bool(verbose)
        )
    final_estimator = (
        LinearRegression()
        if method == "stacking"
        else LinearRegression(positive=True, fit_intercept=False)
    )
    return StackingRegressor(
        estimators,
        final_estimator=final_estimator,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
    )


class EnsembleRegressor(RegressorMixin, BaseEstimator):
    """Regressor that combines regressors, optionally tuned with optuna.

    ``fit`` builds a ``VotingRegressor`` or a ``StackingRegressor`` of
    scikit-learn from the given models, fits it on ``X`` and ``y`` as they
    are (without converting them to arrays, so that pipelines that use the
    column names of a ``pandas.DataFrame`` work), and ``predict`` delegates
    to it. With ``opt=True``, each model is first wrapped in an
    ``OptunaSearchCV`` of optuna-integration that searches the parameters of
    ``yikit.models.ParamDistributions``. In all the methods, the predictions
    are made by the models fitted again on all the training data (never by
    the average of the models of the folds).

    Parameters
    ----------
    estimators : sequence of estimators or of (str, estimator) tuples, \
            default=(RandomForestRegressor(),)
        Regressors to combine. Each one is cloned, so the given objects are
        not changed. A ``(name, estimator)`` pair keeps its name. A model
        without a name is named after its class in lower case, and the
        models that share a name get ``-1``, ``-2``, ... in their order (as
        in ``sklearn.pipeline.make_pipeline``), e.g. ``[Ridge(), SVR(),
        Ridge()]`` gives ``"ridge-1"``, ``"svr"`` and ``"ridge-2"``. A
        ``Pipeline`` whose last step is a regressor is accepted, e.g. to
        select the features of a model with
        ``yikit.feature_selection.BorutaPy`` before it.
    method : {"blending", "average", "stacking"}, default="blending"
        How to combine the models.

        - ``"average"``: ``VotingRegressor``, the average of the predictions
          of the models.
        - ``"stacking"``: ``StackingRegressor`` whose final estimator is
          ``LinearRegression()``, fitted on the predictions of the models
          for the test part of each fold of ``cv``.
        - ``"blending"``: as ``"stacking"``, with
          ``LinearRegression(positive=True, fit_intercept=False)``: the
          weights of the models are non-negative and are not scaled to sum
          to 1.
    cv : int, cross-validation generator or iterable, default=5
        Splitting of the cross-validation of ``StackingRegressor`` (for
        ``"stacking"`` and ``"blending"``) and of the tuning of each model
        (when ``opt=True``). Pass an int or a splitter such as ``KFold``
        when both are used: an iterable of ``(train, test)`` indices refers
        to the samples of the whole data, which the tuning in the folds of
        ``StackingRegressor`` does not see.
    n_jobs : int or None, default=None
        Number of jobs of ``VotingRegressor`` or ``StackingRegressor``,
        which fit the models (and the folds) in parallel. None means 1
        unless in a ``joblib.parallel_backend`` context. The models keep
        their own ``n_jobs``, so the numbers of jobs multiply between the
        layers (this one, the tuning, and each model). To run the jobs in
        parallel here, set ``n_jobs=1`` on each model; otherwise keep None
        and let the models run in parallel.
    random_state : int, RandomState instance or None, default=None
        Seeds the tuning: an integer drawn from it for each model is the
        ``random_state`` of its ``OptunaSearchCV`` (the seed of the
        sampler). Pass an int for reproducible tuning. The models keep
        their own ``random_state``.
    scoring : str, callable or None, default="neg_mean_squared_error"
        Score that the tuning maximizes (the ``scoring`` of
        ``OptunaSearchCV``). Not used when ``opt=False``.
    verbose : int, default=0
        Verbosity of ``StackingRegressor`` (``VotingRegressor`` gets
        ``bool(verbose)``) and of the tuning. With 0, the logs of the
        trials of optuna are hidden while the models are tuned.
    opt : bool, default=True
        Whether to tune each model before combining them. Each model is
        wrapped in an ``OptunaSearchCV`` that runs ``n_trials`` trials on
        the parameters of ``ParamDistributions(model,
        n_features=X.shape[1])`` with ``cv`` and ``scoring``, and fits the
        model with the best parameters on its training data. It needs
        optuna and optuna-integration; with ``opt=False`` they are not
        imported. With ``"stacking"`` and ``"blending"``, each model is
        tuned ``cv + 1`` times (in each fold, on the training part of the
        fold only, and on all the data), so it is fitted about ``n_trials *
        cv * (cv + 1)`` times (3000 times with the defaults); with
        ``"average"`` it is tuned once (``n_trials * cv`` fits).
    n_trials : int, default=100
        Number of trials of the tuning of each model. Not used when
        ``opt=False``.

    Attributes
    ----------
    estimator_ : VotingRegressor or StackingRegressor
        The fitted ensemble of scikit-learn that ``predict`` uses.
    estimators_ : list of estimators
        The models fitted on all the training data, in the order of
        ``estimators``. With ``opt=True`` they are the fitted
        ``OptunaSearchCV``, whose ``study_``, ``best_params_`` and
        ``best_estimator_`` hold the results of the tuning.
    named_estimators_ : sklearn.utils.Bunch
        The models of ``estimators_`` by name.
    final_estimator_ : LinearRegression
        The final estimator fitted on the cross-validated predictions of the
        models. Only with ``"stacking"`` and ``"blending"``.
    weights_ : ndarray of shape (n_estimators,) or None
        The weights of the models (``final_estimator_.coef_``) with
        ``"stacking"`` and ``"blending"``; None with ``"average"``.
    n_features_in_ : int
        Number of features seen during ``fit``.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of the features seen during ``fit``. Defined only when ``X``
        has feature names that are all strings (such as the columns of a
        ``pandas.DataFrame``), with scikit-learn >= 1.0.

    See Also
    --------
    sklearn.ensemble.VotingRegressor : The ensemble of ``"average"``.
    sklearn.ensemble.StackingRegressor : The ensemble of ``"stacking"`` and
        ``"blending"``.
    yikit.models.ParamDistributions : The parameters searched with
        ``opt=True``.

    Notes
    -----
    yikit 0.4.0 removed the ``boruta`` parameter and the ``results_``
    attribute. To select features, pass the models in pipelines that select
    them first. To measure the scores of the ensemble, cross-validate it
    (e.g. with ``sklearn.model_selection.cross_validate``).

    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.linear_model import Ridge
    >>> from sklearn.svm import SVR
    >>> from yikit.models import EnsembleRegressor
    >>> X, y = make_regression(
    ...     n_samples=50, n_features=4, noise=1.0, random_state=334
    ... )
    >>> ensemble = EnsembleRegressor(
    ...     estimators=[Ridge(), SVR()], method="blending", cv=3, opt=False
    ... )
    >>> ensemble = ensemble.fit(X, y)
    >>> list(ensemble.named_estimators_)
    ['ridge', 'svr']
    >>> bool((ensemble.weights_ >= 0).all())
    True
    >>> ensemble.predict(X[:3]).shape
    (3,)
    """

    def __init__(
        self,
        estimators: Sequence[Any] = (RandomForestRegressor(),),
        method: str = "blending",
        cv: int | BaseCrossValidator | Iterable[Any] = 5,
        n_jobs: int | None = None,
        random_state: int | np.random.RandomState | None = None,
        scoring: str | Callable[..., float] | None = "neg_mean_squared_error",
        verbose: int = 0,
        opt: bool = True,
        n_trials: int = 100,
    ) -> None:
        self.estimators = estimators
        self.method = method
        self.cv = cv
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.scoring = scoring
        self.verbose = verbose
        self.opt = opt
        self.n_trials = n_trials

    def fit(self, X: ArrayLike, y: ArrayLike) -> EnsembleRegressor:
        """Fit the ensemble of the (tuned) models.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. It is passed to the models as it is.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : EnsembleRegressor
            The fitted ensemble.

        Raises
        ------
        ValueError
            If ``method`` is not one of ``"average"``, ``"stacking"`` and
            ``"blending"``, if ``estimators`` is empty, or if one of the
            models is not a regressor.
        """
        if self.method not in METHODS:
            raise ValueError(
                f"method must be one of {', '.join(map(repr, METHODS))}, "
                f"but got {self.method!r}."
            )
        named_estimators = _named_estimators(self.estimators)
        if not named_estimators:
            raise ValueError(
                "estimators is empty; pass at least one regressor."
            )
        for name, estimator in named_estimators:
            _check_regressor(name, estimator)

        rng = check_random_state(self.random_state)
        models = [
            (name, clone(estimator)) for name, estimator in named_estimators
        ]
        if self.opt:
            param_distributions, search_regressor = _import_search_regressor()
            n_features = _n_features(X)
            models = [
                (
                    name,
                    search_regressor(
                        model,
                        param_distributions(model, n_features=n_features),
                        n_trials=self.n_trials,
                        cv=self.cv,
                        scoring=self.scoring,
                        random_state=int(rng.randint(2**31 - 1)),
                        verbose=self.verbose,
                    ),
                )
                for name, model in models
            ]

        ensemble = _build_ensemble(
            self.method, models, self.cv, self.n_jobs, self.verbose
        )
        ensemble.fit(X, y)

        for attribute in _OPTIONAL_FITTED_ATTRIBUTES:
            if hasattr(self, attribute):  # left by a previous fit
                delattr(self, attribute)
        self.estimator_ = ensemble
        self.estimators_ = ensemble.estimators_
        self.named_estimators_ = ensemble.named_estimators_
        if isinstance(ensemble, StackingRegressor):
            self.final_estimator_ = ensemble.final_estimator_
            self.weights_ = self.final_estimator_.coef_
        else:
            self.weights_ = None
        try:
            self.n_features_in_ = ensemble.n_features_in_
        except AttributeError:  # the first model does not record it
            self.n_features_in_ = _n_features(X)
        if hasattr(ensemble, "feature_names_in_"):
            self.feature_names_in_ = ensemble.feature_names_in_
        return self

    def predict(self, X: ArrayLike) -> NDArray[Any]:
        """Predict with the ensemble of the models fitted on all the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict. They are passed to the models as they are.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted values.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the ensemble is not fitted yet.
        """
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(X)
