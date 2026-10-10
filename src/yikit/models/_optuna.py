"""Hyperparameter optimization using Optuna.

``Objective`` is an objective function of optuna that cross-validates a
copy of an estimator with the parameters of each trial. It searches the
search space table of ``yikit.models._search_space`` (following the
estimators nested in ``Pipeline`` and ``TransformedTargetRegressor``) and
sets the parameters with ``yikit.models._params.apply_params``, so the
parameters of the estimator passed by the user are kept.
``ParamDistributions`` holds the distributions of the same table, to pass
to ``OptunaSearchCV`` of optuna-integration.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import optuna
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from sklearn.exceptions import FitFailedWarning
from sklearn.metrics import check_scoring
from sklearn.model_selection import check_cv, cross_validate
from sklearn.utils import check_random_state, check_X_y

from yikit.models._params import (
    _is_estimator,
    apply_params,
    find_unspecified_random_states,
)
from yikit.models._search_space import (
    get_search_space,
    ignores_nested_set_params,
    resolve_estimator,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from numpy.typing import ArrayLike
    from sklearn.model_selection import BaseCrossValidator


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


def _is_covered(name: str, fixed_names: Iterable[str]) -> bool:
    """Return whether one of ``fixed_names`` sets the parameter ``name``.

    A fixed name covers itself and the names under it: ``Base`` covers
    ``Base`` and ``Base__max_depth``, because the estimator fixed as
    ``Base`` replaces the one that ``Base__max_depth`` would change.

    Parameters
    ----------
    name : str
        Prefixed parameter name, e.g. ``Base__max_depth``.
    fixed_names : Iterable of str
        Names of ``fixed_params``.

    Returns
    -------
    bool
        Whether ``name`` equals a fixed name or starts with a fixed name
        followed by ``"__"``.
    """
    return any(
        name == fixed_name or name.startswith(f"{fixed_name}__")
        for fixed_name in fixed_names
    )


def _random_state_rule_names(
    estimator: Any, fixed_params: Mapping[str, Any]
) -> list[str]:
    """Return the names that the ``random_state`` rule sets.

    The unspecified ``random_state`` parameters are looked up in
    ``estimator`` and in each estimator of ``fixed_params``, whose names
    get the fixed name as prefix (``Base__random_state`` for a fixed
    ``Base``). The objects given by the user are inspected, not copies, so
    that the global ``RandomState`` that ngboost puts in place of None is
    still found. A name found in one of these estimators is dropped when a
    fixed name inside that estimator covers it (see ``_is_covered``): the
    fixed value replaces what the rule would change, so the values of
    ``fixed_params``, including the parameters of a fixed estimator, are
    never overwritten.

    Parameters
    ----------
    estimator : object
        The estimator passed by the user.
    fixed_params : Mapping of str to object
        The ``fixed_params`` passed by the user.

    Returns
    -------
    list of str
        Prefixed names accepted by ``apply_params`` once ``fixed_params``
        are applied, each at most once.
    """
    sources = [("", estimator)] + [
        (f"{fixed_name}__", value)
        for fixed_name, value in fixed_params.items()
        if _is_estimator(value)
    ]
    return [
        prefix + name
        for prefix, source in sources
        for name in find_unspecified_random_states(source)
        if not _is_covered(
            prefix + name,
            [
                fixed_name
                for fixed_name in fixed_params
                if fixed_name.startswith(prefix)
            ],
        )
    ]


#: What ``custom_params`` of ``Objective`` must be to replace the table.
_OBJECTIVE_CUSTOM_PARAMS = (
    "a function that takes an optuna trial and returns a non-empty dict of "
    "parameter values"
)

#: What ``custom_params`` of ``ParamDistributions`` must be to replace the
#: table.
_PARAM_DISTRIBUTIONS_CUSTOM_PARAMS = (
    "a non-empty dict of optuna distributions or a function that returns one"
)


def _no_search_space_message(
    estimator: Any, custom_params: str = _OBJECTIVE_CUSTOM_PARAMS
) -> str:
    """Return the message telling that ``estimator`` has no search space.

    Parameters
    ----------
    estimator : object
        Estimator whose resolved model has no row in the table.
    custom_params : str, default=_OBJECTIVE_CUSTOM_PARAMS
        Description of the ``custom_params`` that would replace the table.

    Returns
    -------
    str
        Message naming the type of the resolved model and telling how to
        pass ``custom_params``.
    """
    _, model = resolve_estimator(estimator)
    return (
        f"No search space is registered for {type(model).__name__}. Pass "
        f"custom_params, {custom_params}, to search it."
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
    none. An estimator given in ``fixed_params`` follows the same rule in
    place of the one it replaces.

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
        values and the ``random_state`` rule. A fixed name covers itself
        and every name under it, and the names of the table that it covers
        are not searched. When it holds an estimator, such as
        ``{"Base": DecisionTreeRegressor(max_depth=5)}`` for
        ``NGBRegressor`` or ``{"svr": SVR(C=10.0)}`` for a ``Pipeline``,
        the ``random_state`` rule does not set the names under it either
        (``Base__max_depth``, ``svr__C``, ... are neither searched nor
        changed), so the parameters of the given estimator are kept. The
        rule is applied to that estimator instead: its ``random_state``
        parameters left as None get ``model_random_state`` (e.g.
        ``Base__random_state``), and those given are kept. Names under a
        fixed estimator that ``custom_params`` returns are still applied,
        to the copy of that estimator. ``None`` and ``{}`` mean no fixed
        value.
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
        and the number of columns of ``X``, without the names covered by
        ``fixed_params`` (the fixed names and the names under them).
        ``None`` when the table has no row for the model.
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
        self._random_state_params: dict[str, Any] = {
            name: self.model_random_state
            for name in _random_state_rule_names(estimator, self.fixed_params)
        }

        search_space = get_search_space(estimator, n_features=self.X.shape[1])
        self.param_distributions: dict[str, BaseDistribution] | None = (
            None
            if search_space is None
            else {
                name: distribution
                for name, distribution in search_space.items()
                if not _is_covered(name, self.fixed_params)
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


def _custom_distributions(
    custom_params: (
        Mapping[str, BaseDistribution]
        | Callable[[optuna.trial.BaseTrial], Mapping[str, BaseDistribution]]
        | None
    ),
) -> dict[str, BaseDistribution]:
    """Return the distributions given by ``custom_params``.

    Parameters
    ----------
    custom_params : Mapping, callable or None
        ``custom_params`` of ``ParamDistributions``: a mapping from
        parameter names to optuna distributions, a function that takes an
        optuna trial and returns one, or None. The function is called once
        with ``optuna.trial.FixedTrial({})``, and what it raises propagates.

    Returns
    -------
    dict of str to optuna.distributions.BaseDistribution
        New dict with the given names and distributions. Empty when
        ``custom_params`` is None or gives no distribution.

    Raises
    ------
    TypeError
        If ``custom_params``, or what its function returns, is not a
        mapping, or if one of its values is not an optuna distribution.
    """
    if custom_params is None:
        return {}
    custom: Mapping[str, Any]
    if isinstance(custom_params, Mapping):
        custom = custom_params
    elif callable(custom_params):
        custom = custom_params(optuna.trial.FixedTrial({}))
        if not isinstance(custom, Mapping):
            raise TypeError(
                "custom_params of ParamDistributions must return a mapping "
                "from parameter names to optuna distributions, got "
                f"{type(custom).__name__}."
            )
    else:
        raise TypeError(
            "custom_params of ParamDistributions must be a mapping from "
            "parameter names to optuna distributions, a function that "
            f"returns one, or None, got {type(custom_params).__name__}."
        )
    invalid = [
        f"{name!r} ({type(value).__name__})"
        for name, value in custom.items()
        if not isinstance(value, BaseDistribution)
    ]
    if invalid:
        raise TypeError(
            "custom_params of ParamDistributions must give optuna "
            "distributions such as optuna.distributions.FloatDistribution, "
            f"but got other values for {', '.join(invalid)}."
        )
    return dict(custom)


def _set_params_warning(estimator: Any, names: Iterable[str]) -> str | None:
    """Return a warning about the names ``OptunaSearchCV`` cannot apply.

    ``OptunaSearchCV`` sets the parameters with ``estimator.set_params``.
    When the resolved model ignores nested names in its own ``set_params``
    (``NGBRegressor`` of ngboost 0.4.0 or later), the names that go below
    the model, such as ``Base__max_depth``, have no effect.

    Parameters
    ----------
    estimator : object
        The estimator passed to ``ParamDistributions``.
    names : Iterable of str
        Names of the distributions, prefixed for ``estimator``.

    Returns
    -------
    str or None
        Message listing the names that the model's own ``set_params``
        ignores, or None when there is none.
    """
    prefix, model = resolve_estimator(estimator)
    if not ignores_nested_set_params(model):
        return None
    ignored = [
        name
        for name in names
        if name.startswith(prefix) and "__" in name[len(prefix) :]
    ]
    if not ignored:
        return None
    model_name = type(model).__name__
    return (
        f"OptunaSearchCV cannot apply {ignored}: set_params of {model_name} "
        "ignores nested parameter names, so these values never reach the "
        f"estimators inside {model_name}. Search with yikit.models.Objective "
        "instead, which applies them to a copy of the model."
    )


class ParamDistributions(dict):  # a dict that OptunaSearchCV keeps as is
    """Distributions of the search space to pass to ``OptunaSearchCV``.

    A dict from parameter names to optuna distributions, built from the
    same search space table as ``Objective``: for the same estimator and the
    same number of features, it holds the same names and the same
    distributions as the trials of ``Objective``. It is passed as it is as
    ``param_distributions`` of ``OptunaSearchCV`` of optuna-integration,
    and keeps its contents and type through ``copy.deepcopy`` and
    ``sklearn.base.clone``.

    The table knows ``LGBMRegressor`` of LightGBM, ``GBDTRegressor`` of
    yikit, ``RandomForestRegressor``, ``SVR``, ``LinearSVR``,
    ``MLPRegressor``, ``NGBRegressor`` of ngboost, ``PLSRegression``,
    ``Ridge``, ``Lasso`` and ``ElasticNet`` (and their subclasses). The
    last step of a ``Pipeline`` and the ``regressor`` of a
    ``TransformedTargetRegressor`` are followed, so a nested model gets
    prefixed names such as ``regressor__svr__C``, which ``OptunaSearchCV``
    passes to ``set_params`` of the estimator. Only searched parameters are
    included: fixed values, such as ``n_jobs``, ``random_state`` or the
    values of ``RecommendedParams``, are set on the estimator itself, whose
    other parameters ``OptunaSearchCV`` keeps.

    ``set_params`` of ``NGBRegressor`` (ngboost 0.4.0 or later) ignores
    nested names such as ``Base__max_depth``, so ``OptunaSearchCV`` cannot
    apply them. A ``UserWarning`` lists such names; ``Objective`` applies
    them.

    Parameters
    ----------
    estimator : object
        scikit-learn compatible estimator instance to search, e.g.
        ``sklearn.svm.SVR()`` or a ``Pipeline`` whose last step is one. It
        is not modified.
    custom_params : Mapping, callable or None, default=None
        Mapping from parameter names to optuna distributions, e.g.
        ``{"C": FloatDistribution(0.1, 10.0, log=True)}``, or a function
        that takes an optuna trial and returns such a mapping. The function
        is called once with ``optuna.trial.FixedTrial({})``, so it must
        return distributions instead of suggesting values (a function that
        suggests values, as ``custom_params`` of ``Objective`` does, raises
        the error of ``FixedTrial``). A non-empty mapping is used instead of
        the search space table, and its names are used as they are (no
        prefix is added). None or an empty mapping means that the table is
        used.
    n_features : int or None, default=None
        Number of features of the data passed to ``estimator`` (the
        outermost one when nested). Keyword-only. When given, the upper
        bound of the parameters limited by the data, ``n_components`` of
        ``PLSRegression``, becomes ``min(10, n_features)``. When None, the
        bound of the table (10) is kept, and ``OptunaSearchCV`` records the
        trials that exceed the number of features as failed and goes on
        with the search.

    Attributes
    ----------
    estimator : object
        The estimator passed by the user.
    custom_params : Mapping, callable or None
        The ``custom_params`` passed by the user.
    n_features : int or None
        The ``n_features`` passed by the user.

    Raises
    ------
    NotImplementedError
        If the table has no search space for the (resolved) model and
        ``custom_params`` gives no distribution.
    TypeError
        If ``custom_params``, or what its function returns, is not a
        mapping, or if one of its values is not an optuna distribution.
        The ``fixed_params`` and ``random_state`` arguments of yikit
        0.4.0-rc.0 were removed, and passing them raises a ``TypeError``
        too.
    ValueError
        If ``n_features`` is smaller than 1, or if a ``Pipeline`` on the
        way has no steps.

    Warns
    -----
    UserWarning
        If the resolved model ignores nested names in its own
        ``set_params`` (``NGBRegressor`` of ngboost 0.4.0 or later) and the
        distributions hold such names, e.g. ``Base__max_depth``.

    See Also
    --------
    Objective : Objective function of optuna that searches the same table.

    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.linear_model import ElasticNet
    >>> from yikit.models import ParamDistributions
    >>> try:
    ...     from optuna_integration import OptunaSearchCV
    ... except ImportError:  # old optuna that bundles the integration
    ...     from optuna.integration import OptunaSearchCV
    >>> X, y = make_regression(n_samples=50, n_features=5, random_state=334)
    >>> estimator = ElasticNet()
    >>> param_distributions = ParamDistributions(
    ...     estimator, n_features=X.shape[1]
    ... )
    >>> sorted(param_distributions)
    ['alpha', 'l1_ratio']
    >>> search = OptunaSearchCV(
    ...     estimator,
    ...     param_distributions=param_distributions,
    ...     n_trials=10,
    ...     random_state=334,
    ... ).fit(X, y)
    >>> sorted(search.best_params_)
    ['alpha', 'l1_ratio']

    ``Ridge`` and ``Lasso`` are searched in the same way (they replace the
    removed ``LinearModelRegressor``):

    >>> from sklearn.linear_model import Lasso, Ridge
    >>> sorted(ParamDistributions(Ridge()))
    ['alpha']
    >>> sorted(ParamDistributions(Lasso()))
    ['alpha']

    An ``SVR`` scaled by a ``Pipeline`` and a ``TransformedTargetRegressor``
    (which replace the removed ``SupportVectorRegressor``) gets prefixed
    names. The ``gamma`` that ``SupportVectorRegressor`` used is set on the
    ``SVR`` itself, and ``OptunaSearchCV`` keeps it:

    >>> from sklearn.compose import TransformedTargetRegressor
    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.svm import SVR
    >>> estimator = TransformedTargetRegressor(
    ...     regressor=make_pipeline(StandardScaler(), SVR(gamma="auto")),
    ...     transformer=StandardScaler(),
    ... )
    >>> param_distributions = ParamDistributions(
    ...     estimator, n_features=X.shape[1]
    ... )
    >>> sorted(param_distributions)
    ['regressor__svr__C', 'regressor__svr__epsilon']
    >>> search = OptunaSearchCV(
    ...     estimator,
    ...     param_distributions=param_distributions,
    ...     n_trials=5,
    ...     random_state=334,
    ... ).fit(X, y)
    >>> sorted(search.best_params_)
    ['regressor__svr__C', 'regressor__svr__epsilon']
    >>> search.best_estimator_.regressor_.named_steps["svr"].gamma
    'auto'

    The upper bound of ``n_components`` of ``PLSRegression`` follows
    ``n_features``, and ``custom_params`` replaces the table:

    >>> from optuna.distributions import FloatDistribution
    >>> from sklearn.cross_decomposition import PLSRegression
    >>> ParamDistributions(PLSRegression())["n_components"].high
    10
    >>> ParamDistributions(PLSRegression(), n_features=3)["n_components"].high
    3
    >>> sorted(
    ...     ParamDistributions(
    ...         SVR(), {"C": FloatDistribution(0.1, 10.0, log=True)}
    ...     )
    ... )
    ['C']
    """

    def __init__(
        self,
        estimator: Any,
        custom_params: (
            Mapping[str, BaseDistribution]
            | Callable[
                [optuna.trial.BaseTrial], Mapping[str, BaseDistribution]
            ]
            | None
        ) = None,
        *,
        n_features: int | None = None,
    ) -> None:
        search_space = get_search_space(estimator, n_features=n_features)
        custom = _custom_distributions(custom_params)
        distributions: dict[str, BaseDistribution]
        if custom:
            distributions = custom
        elif search_space is not None:
            distributions = search_space
        else:
            raise NotImplementedError(
                _no_search_space_message(
                    estimator, _PARAM_DISTRIBUTIONS_CUSTOM_PARAMS
                )
            )
        super().__init__(distributions)
        self.estimator = estimator
        self.custom_params = custom_params
        self.n_features = n_features

        message = _set_params_warning(estimator, self)
        if message is not None:
            warnings.warn(message, UserWarning, stacklevel=2)

    def __repr__(self) -> str:
        """Return the arguments that built the distributions."""
        return (
            f"{type(self).__name__}({self.estimator!r}, "
            f"custom_params={self.custom_params!r}, "
            f"n_features={self.n_features!r})"
        )
