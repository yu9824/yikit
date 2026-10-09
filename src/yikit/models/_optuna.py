"""Hyperparameter optimization using Optuna.

``Objective`` is an objective function of optuna that cross-validates a
copy of an estimator with the parameters of each trial. It searches the
search space table of ``yikit.models._search_space`` (following the
estimators nested in ``Pipeline`` and ``TransformedTargetRegressor``) and
sets the parameters with ``yikit.models._params.apply_params``, so the
parameters of the estimator passed by the user are kept.
``ParamDistributions`` holds distributions to pass to ``OptunaSearchCV``.
"""

from __future__ import annotations

import sys
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import optuna
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import FitFailedWarning
from sklearn.metrics import check_scoring
from sklearn.model_selection import check_cv, cross_validate
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.utils import check_random_state, check_X_y

from yikit.helpers import is_installed
from yikit.models._linear import LinearModelRegressor
from yikit.models._params import apply_params, find_unspecified_random_states
from yikit.models._search_space import get_search_space, resolve_estimator
from yikit.models._svm import SupportVectorRegressor

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

    from numpy.typing import ArrayLike
    from sklearn.model_selection import BaseCrossValidator

if sys.version_info >= (3, 10):
    from types import NoneType
else:
    NoneType = type(None)  # type: ignore[assignment,misc]

if is_installed("lightgbm"):
    from lightgbm import LGBMRegressor  # type: ignore[reportMissingImports]

    from yikit.models._gbdt import GBDTRegressor

else:
    LGBMRegressor = NoneType  # type: ignore[assignment,misc]
    GBDTRegressor = NoneType  # type: ignore[assignment,misc]


if is_installed("ngboost"):
    from ngboost import NGBRegressor  # type: ignore[reportMissingImports]
else:
    NGBRegressor = NoneType


class ParamDistributions(
    dict
):  # HACK: should be MutableMapping but OptunaSearchCV does not support it
    """Parameter distributions for OptunaSearchCV.

    This class provides a dictionary-like interface for parameter distributions
    compatible with optuna.integration.OptunaSearchCV. It is designed to work
    similarly to the Objective class but returns BaseDistribution objects instead
    of using trial.suggest_* methods directly.

    Example
    -------
    >>> from yikit.models import ParamDistributions
    >>> from optuna.integration import OptunaSearchCV
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> param_distributions = ParamDistributions(RandomForestRegressor())
    >>> study = optuna.create_study()
    >>> optuna_search_cv = OptunaSearchCV(
    >>>     estimator=RandomForestRegressor(),
    >>>     param_distributions=param_distributions,
    >>>     study=study,
    >>>     cv=5,
    >>>     random_state=42,
    >>> )
    >>> optuna_search_cv.fit(X, y)
    """

    def __init__(
        self,
        estimator,
        custom_params=lambda trial: {},
        fixed_params={},
        random_state=None,
    ):
        """Initialize parameter distributions.

        Parameters
        ----------
        estimator : sklearn-based estimator instance
            e.g. sklearn.ensemble.RandomForestRegressor()
        custom_params : func, optional
            If you want to do your own custom range of optimization, you can define
            it here with a function that returns a dictionary of BaseDistribution
            objects, by default lambda trial:{}
        fixed_params : dict, optional
            If you have a fixed variable, you can specify it in the dictionary.,
            by default {}
        random_state : int or RandomState object, optional
            seed, by default None
        """
        self.estimator = estimator
        self.custom_params = custom_params
        self._fixed_params = fixed_params
        self.rng = check_random_state(random_state)
        self.distributions = self._build_distributions()
        for k, v in self.distributions.items():
            self[k] = v

    def _build_distributions(
        self,
    ) -> dict[str, optuna.distributions.BaseDistribution]:
        """Build parameter distributions based on estimator type."""
        if isinstance(self.estimator, (GBDTRegressor, LGBMRegressor)):
            return {
                "n_estimators": optuna.distributions.IntDistribution(
                    low=10, high=1000, log=True
                ),
                "min_child_weight": optuna.distributions.FloatDistribution(
                    low=0.001, high=10, log=True
                ),
                "colsample_bytree": optuna.distributions.FloatDistribution(
                    low=0.6, high=0.95
                ),
                "subsample": optuna.distributions.FloatDistribution(
                    low=0.6, high=0.95
                ),
                "num_leaves": optuna.distributions.IntDistribution(
                    low=2**3, high=2**9, log=True
                ),
            }
        elif isinstance(self.estimator, RandomForestRegressor):
            return {
                "min_samples_split": optuna.distributions.IntDistribution(
                    low=2, high=16
                ),
                "max_depth": optuna.distributions.IntDistribution(
                    low=10, high=100
                ),
                "n_estimators": optuna.distributions.IntDistribution(
                    low=10, high=1000, log=True
                ),
            }
        elif isinstance(self.estimator, (SupportVectorRegressor, SVR)):
            return {
                "C": optuna.distributions.FloatDistribution(
                    low=2**-5, high=2**10, log=True
                ),
                "epsilon": optuna.distributions.FloatDistribution(
                    low=2**-10, high=2**0, log=True
                ),
            }
        elif isinstance(self.estimator, LinearModelRegressor):
            return {
                "linear_model": optuna.distributions.CategoricalDistribution(
                    choices=["ridge", "lasso"]
                ),
                "alpha": optuna.distributions.FloatDistribution(
                    low=0.1, high=10, log=True
                ),
                "fit_intercept": optuna.distributions.CategoricalDistribution(
                    choices=[True, False]
                ),
                "max_iter": optuna.distributions.FloatDistribution(
                    low=100, high=10000, log=True
                ),
                "tol": optuna.distributions.FloatDistribution(
                    low=0.0001, high=0.01, log=True
                ),
            }
        elif isinstance(self.estimator, MLPRegressor):
            return {
                "hidden_layer_sizes": optuna.distributions.IntDistribution(
                    low=50, high=300
                ),
                "alpha": optuna.distributions.FloatDistribution(
                    low=1e-5, high=1e-3, log=True
                ),
                "learning_rate_init": optuna.distributions.FloatDistribution(
                    low=1e-5, high=1e-3, log=True
                ),
            }
        elif is_installed("ngboost") and isinstance(
            self.estimator, NGBRegressor
        ):
            return {
                "Base__max_depth": optuna.distributions.IntDistribution(
                    low=2, high=100
                ),
                "Base__criterion": optuna.distributions.CategoricalDistribution(
                    choices=["squared_error", "friedman_mse"]
                ),
                "n_estimators": optuna.distributions.IntDistribution(
                    low=10, high=1000, log=True
                ),
                "minibatch_frac": optuna.distributions.FloatDistribution(
                    low=0.5, high=1.0
                ),
            }
        else:
            # Try custom_params - it should return a dict of BaseDistribution objects
            # custom_params can be either:
            # 1. A dict of BaseDistribution objects directly
            # 2. A callable that returns a dict of BaseDistribution objects
            if isinstance(self.custom_params, dict):
                return self.custom_params
            elif callable(self.custom_params):
                try:
                    # Create a dummy study and trial to test custom_params
                    study = optuna.create_study()
                    trial = study.ask()
                    custom_result = self.custom_params(trial)
                    study.tell(trial, 0.0)
                    if custom_result and isinstance(custom_result, dict):
                        # Verify that values are BaseDistribution objects
                        from optuna.distributions import BaseDistribution

                        if all(
                            isinstance(v, BaseDistribution)
                            for v in custom_result.values()
                        ):
                            return custom_result
                except Exception:
                    pass
            raise NotImplementedError(
                f"ParamDistributions not implemented for {type(self.estimator)}. "
                "Please provide custom_params as a dict of BaseDistribution objects "
                "or a callable that returns such a dict."
            )

    def __repr__(self) -> str:
        """Return string representation of parameter distributions."""
        return (
            f"ParamDistributions({self.estimator}, "
            f"random_state={self.rng}, "
            f"fixed_params={self._fixed_params}, "
            f"custom_params={self.custom_params})"
        )


def _suggest(
    trial: optuna.trial.BaseTrial, name: str, distribution: BaseDistribution
) -> Any:
    """Suggest a value of ``name`` from ``distribution`` with ``trial``.

    Parameters
    ----------
    trial : optuna.trial.BaseTrial
        Trial that suggests the value, such as ``optuna.trial.Trial`` or
        ``optuna.trial.FixedTrial``.
    name : str
        Parameter name to suggest.
    distribution : optuna.distributions.BaseDistribution
        An ``IntDistribution``, ``FloatDistribution`` or
        ``CategoricalDistribution`` (including their deprecated
        subclasses), whose bounds, step, log scale or choices are passed to
        the matching ``suggest_*`` method.

    Returns
    -------
    object
        The value suggested by ``trial``.

    Raises
    ------
    TypeError
        If ``distribution`` is of another type.
    """
    if isinstance(distribution, IntDistribution):
        return trial.suggest_int(
            name,
            distribution.low,
            distribution.high,
            step=distribution.step,
            log=distribution.log,
        )
    if isinstance(distribution, FloatDistribution):
        return trial.suggest_float(
            name,
            distribution.low,
            distribution.high,
            step=distribution.step,
            log=distribution.log,
        )
    if isinstance(distribution, CategoricalDistribution):
        return trial.suggest_categorical(name, distribution.choices)
    raise TypeError(
        f"Cannot suggest {name!r} from a {type(distribution).__name__}. "
        "Use IntDistribution, FloatDistribution or CategoricalDistribution."
    )


def _no_search_space_message(estimator: Any) -> str:
    """Return the message telling that ``estimator`` has no search space."""
    _, model = resolve_estimator(estimator)
    return (
        f"No search space is registered for {type(model).__name__}. Pass "
        "custom_params, a function that takes an optuna trial and returns a "
        "non-empty dict of parameter values, to search it."
    )


class Objective:
    """Objective function of optuna that cross-validates an estimator.

    Each trial copies ``estimator`` and sets on the copy only the values
    searched in the trial, the integer of the ``random_state`` rule and
    ``fixed_params``, in this order of priority from the lowest. Every
    other parameter keeps the value of ``estimator``: in particular
    ``n_jobs`` and the values that yikit recommends (see
    ``RecommendedParams``) are never set unless they are in
    ``fixed_params``. The copy is cross-validated, and the mean of its test
    scores is the value of the trial.

    The searched parameters come from the search space table of yikit,
    which knows ``LGBMRegressor`` of LightGBM, ``GBDTRegressor`` of yikit,
    ``RandomForestRegressor``, ``SVR``, ``LinearSVR``, ``MLPRegressor``,
    ``NGBRegressor`` of ngboost, ``PLSRegression``, ``Ridge``, ``Lasso`` and
    ``ElasticNet`` (and their subclasses). The last step of a ``Pipeline``
    and the ``regressor`` of a ``TransformedTargetRegressor`` are followed,
    so a nested model is searched with prefixed names such as
    ``regressor__svr__C``. ``custom_params`` replaces the table.

    ``random_state`` rule: every ``random_state`` parameter of
    ``estimator`` and of the estimators nested in it whose value is None
    (or the global ``RandomState`` of NumPy, which ngboost puts in place of
    None) is set to ``model_random_state``, one integer drawn from
    ``random_state`` when the objective is created. The values given by the
    user are kept, and models without a ``random_state`` parameter get
    none.

    A trial whose model fails to fit in every fold of the cross-validation
    is reported with a ``FitFailedWarning`` and returns nan, as does a
    trial whose fit fails in some folds; optuna records such a trial as
    failed, never chooses it as the best trial, and goes on with the
    search.

    Parameters
    ----------
    estimator : object
        scikit-learn compatible estimator instance to search, e.g.
        ``sklearn.ensemble.RandomForestRegressor()`` or a ``Pipeline``. It
        is never modified.
    X : array-like of shape (n_samples, n_features)
        Features. Its number of columns bounds the parameters limited by
        the data, such as ``n_components`` of ``PLSRegression``.
    y : array-like of shape (n_samples,)
        Target.
    custom_params : callable or None, default=None
        Function that takes an optuna trial and returns a dict from
        parameter names to values, suggesting the values with the trial,
        e.g. ``lambda trial: {"C": trial.suggest_float("C", 0.1, 10)}``.
        When it returns a non-empty dict, the dict is used instead of the
        search space table, and its names are used as they are (no prefix
        is added). An empty dict means that the table is used.
    fixed_params : Mapping or None, default=None
        Parameter values to set on every model, prefixed like the searched
        names for nested estimators. They take priority over the searched
        values and the ``random_state`` rule, and the parameters of the
        table that they name are not searched. ``None`` and ``{}`` mean no
        fixed value.
    cv : int, cross-validation generator or iterable, default=5
        Cross-validation splitting strategy passed to ``check_cv``.
    random_state : int, RandomState instance or None, default=None
        Seed of the ``TPESampler`` exposed as ``sampler`` and of
        ``model_random_state``, drawn in this order.
    scoring : str, callable or None, default=None
        Scoring passed to ``check_scoring``. ``None`` means the ``score``
        method of ``estimator``.
    n_jobs : int or None, default=None
        Number of jobs of ``cross_validate``. It is not set on the model.

    Attributes
    ----------
    estimator : object
        The estimator passed by the user.
    X : numpy.ndarray of shape (n_samples, n_features)
        Features validated by ``check_X_y``.
    y : numpy.ndarray of shape (n_samples,)
        Target validated by ``check_X_y``.
    custom_params : callable or None
        The ``custom_params`` passed by the user.
    fixed_params : dict
        A copy of ``fixed_params`` (``{}`` when None was passed).
    cv : cross-validation generator
        Cross-validator returned by ``check_cv``.
    scoring : callable
        Scorer returned by ``check_scoring``.
    n_jobs : int or None
        Number of jobs of ``cross_validate``.
    sampler : optuna.samplers.TPESampler
        Sampler seeded from ``random_state``, to pass to
        ``optuna.create_study``.
    model_random_state : int
        Integer set on the unspecified ``random_state`` parameters, the same
        for every trial and for ``get_best_params``.
    param_distributions : dict or None
        Prefixed distributions of the search space table for ``estimator``
        and the number of columns of ``X``, without the names in
        ``fixed_params``. ``None`` when the table has no row for the model.
    estimator_ : object
        Unfitted model of the last trial, with the parameters of that trial
        set. It exists after the first trial.

    Raises
    ------
    NotImplementedError
        If the table has no search space for the (resolved) model and
        ``custom_params`` is None.
    ValueError
        If a name in ``fixed_params`` is not a parameter of the targeted
        estimator.

    Examples
    --------
    >>> import optuna
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.linear_model import ElasticNet
    >>> from yikit.models import Objective
    >>> X, y = make_regression(n_samples=50, n_features=5, random_state=334)
    >>> objective = Objective(ElasticNet(), X, y, random_state=334)
    >>> sorted(objective.param_distributions)
    ['alpha', 'l1_ratio']
    >>> study = optuna.create_study(
    ...     direction="maximize", sampler=objective.sampler
    ... )
    >>> study.optimize(objective, n_trials=10)
    >>> best_estimator = objective.get_best_estimator(study).fit(X, y)

    ``Ridge`` and ``Lasso`` are searched in the same way (they replace the
    removed ``LinearModelRegressor``):

    >>> from sklearn.linear_model import Lasso, Ridge
    >>> sorted(Objective(Ridge(), X, y).param_distributions)
    ['alpha']
    >>> sorted(Objective(Lasso(), X, y).param_distributions)
    ['alpha']

    An ``SVR`` scaled by a ``Pipeline`` and a ``TransformedTargetRegressor``
    (which replace the removed ``SupportVectorRegressor``) is searched with
    prefixed names. ``fixed_params`` sets the ``gamma`` that
    ``SupportVectorRegressor`` used:

    >>> from sklearn.compose import TransformedTargetRegressor
    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.svm import SVR
    >>> estimator = TransformedTargetRegressor(
    ...     regressor=make_pipeline(StandardScaler(), SVR()),
    ...     transformer=StandardScaler(),
    ... )
    >>> objective = Objective(
    ...     estimator,
    ...     X,
    ...     y,
    ...     fixed_params={"regressor__svr__gamma": "auto"},
    ...     random_state=334,
    ... )
    >>> sorted(objective.param_distributions)
    ['regressor__svr__C', 'regressor__svr__epsilon']
    >>> study = optuna.create_study(
    ...     direction="maximize", sampler=objective.sampler
    ... )
    >>> study.optimize(objective, n_trials=10)
    >>> best_params = objective.get_best_params(study)
    >>> sorted(best_params)
    ['regressor__svr__C', 'regressor__svr__epsilon', 'regressor__svr__gamma']
    >>> best_estimator = objective.get_best_estimator(study)
    >>> best_estimator.regressor.named_steps["svr"].gamma
    'auto'
    >>> best_estimator = best_estimator.fit(X, y)
    """

    def __init__(
        self,
        estimator: Any,
        X: ArrayLike,
        y: ArrayLike,
        custom_params: (
            Callable[[optuna.trial.BaseTrial], Mapping[str, Any]] | None
        ) = None,
        fixed_params: Mapping[str, Any] | None = None,
        cv: int | BaseCrossValidator | Iterable[Any] = 5,
        random_state: int | np.random.RandomState | None = None,
        scoring: str | Callable[..., float] | None = None,
        n_jobs: int | None = None,
    ) -> None:
        self.estimator = estimator
        self.X, self.y = check_X_y(X, y)
        self.custom_params = custom_params
        self.fixed_params: dict[str, Any] = (
            {} if fixed_params is None else dict(fixed_params)
        )
        self.cv = check_cv(cv)
        rng = check_random_state(random_state)
        self.scoring = check_scoring(estimator, scoring)
        self.n_jobs = n_jobs

        # The sampler seed is drawn first, as in yikit 0.4.0-rc.0.
        self.sampler = optuna.samplers.TPESampler(seed=rng.randint(2**31 - 1))
        self.model_random_state = int(rng.randint(2**31 - 1))
        # The original estimator is inspected: a copy would no longer hold
        # the global RandomState that ngboost puts in place of None.
        self._random_state_params: dict[str, Any] = {
            name: self.model_random_state
            for name in find_unspecified_random_states(estimator)
        }

        search_space = get_search_space(estimator, n_features=self.X.shape[1])
        self.param_distributions: dict[str, BaseDistribution] | None = (
            None
            if search_space is None
            else {
                name: distribution
                for name, distribution in search_space.items()
                if name not in self.fixed_params
            }
        )
        if self.param_distributions is None and self.custom_params is None:
            raise NotImplementedError(_no_search_space_message(estimator))
        # Raise a ValueError now for names that the estimator does not have.
        apply_params(
            estimator, {**self._random_state_params, **self.fixed_params}
        )

    def _make_params(self, trial: optuna.trial.BaseTrial) -> dict[str, Any]:
        """Return the parameters to set on the model of ``trial``.

        Parameters
        ----------
        trial : optuna.trial.BaseTrial
            Trial of the search, or a ``FixedTrial`` holding the values of
            a trial to build the same parameters again.

        Returns
        -------
        dict of str to object
            ``{**random_state rule values, **searched values,
            **fixed_params}``. The searched values come from
            ``custom_params`` when it returns a non-empty dict, and
            otherwise from ``param_distributions``.

        Raises
        ------
        NotImplementedError
            If ``custom_params`` returns an empty dict and there is no
            search space for the model.
        """
        custom = (
            None if self.custom_params is None else self.custom_params(trial)
        )
        searched: dict[str, Any]
        if custom:
            searched = dict(custom)
        elif self.param_distributions is not None:
            searched = {
                name: _suggest(trial, name, distribution)
                for name, distribution in self.param_distributions.items()
            }
        else:
            raise NotImplementedError(_no_search_space_message(self.estimator))
        return {**self._random_state_params, **searched, **self.fixed_params}

    def __call__(self, trial: optuna.trial.BaseTrial) -> float:
        """Evaluate the model of ``trial`` by cross-validation.

        The model of the trial is a copy of ``estimator`` with the
        parameters of the trial set, and is kept as ``estimator_``.

        Parameters
        ----------
        trial : optuna.trial.BaseTrial
            Trial given by ``optuna.study.Study.optimize``.

        Returns
        -------
        float
            Mean of the test scores of the cross-validation. nan when the
            fit fails in some folds, or in every fold (then a
            ``FitFailedWarning`` with the original error is emitted), so
            that optuna records the trial as failed.

        Raises
        ------
        NotImplementedError
            If ``custom_params`` returns an empty dict and there is no
            search space for the model.
        ValueError
            If ``custom_params`` returns a name that is not a parameter of
            the targeted estimator.
        """
        self.estimator_ = apply_params(
            self.estimator, self._make_params(trial)
        )
        try:
            scores = cross_validate(
                self.estimator_,
                self.X,
                self.y,
                cv=self.cv,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
            )
        except ValueError as error:
            warnings.warn(
                f"Trial {trial.number} returns nan because the "
                "cross-validation failed, and optuna records it as failed. "
                f"The original error was {type(error).__name__}: {error}",
                FitFailedWarning,
                stacklevel=2,
            )
            return float("nan")
        return float(np.mean(scores["test_score"]))

    def get_best_params(self, study: optuna.study.Study) -> dict[str, Any]:
        """Return the parameters of the model of the best trial.

        They are built again from the values of the best trial, in the same
        way as in the trial (through ``optuna.trial.FixedTrial``), so they
        hold the names returned by ``custom_params`` even when they differ
        from the names given to ``suggest_*``.

        Parameters
        ----------
        study : optuna.study.Study
            Study optimized with this objective. It must have a completed
            trial.

        Returns
        -------
        dict of str to object
            The values searched in the best trial, the integer of the
            ``random_state`` rule for the unspecified ``random_state``
            parameters, and ``fixed_params``, which win over the others.
            The other parameters of ``estimator`` are not included.

        Raises
        ------
        ValueError
            If ``study`` has no completed trial (raised by optuna).
        """
        return self._make_params(optuna.trial.FixedTrial(study.best_params))

    def get_best_estimator(self, study: optuna.study.Study) -> Any:
        """Return an unfitted copy of ``estimator`` with the best parameters.

        Parameters
        ----------
        study : optuna.study.Study
            Study optimized with this objective. It must have a completed
            trial.

        Returns
        -------
        object
            Unfitted copy of ``estimator`` with ``get_best_params(study)``
            set. The other parameters keep the values of ``estimator``.

        Raises
        ------
        ValueError
            If ``study`` has no completed trial (raised by optuna).
        """
        return apply_params(self.estimator, self.get_best_params(study))
