"""Search spaces and recommended values of the models registered in yikit.

This module holds the single table of optuna distributions that both
``Objective`` and ``ParamDistributions`` search. Each row of the table maps
model types to the distributions of their parameters. The table holds only
the searched parameters: fixed values such as ``n_jobs`` and
``random_state`` are never part of it, so the parameters of the estimator
passed by the user are kept as they are.

It also holds the table of the values yikit recommends, which are used only
when the user passes them explicitly (``RecommendedParams``).

Both tables are looked up through ``resolve_estimator``: the last step of a
``Pipeline`` and the ``regressor`` of a ``TransformedTargetRegressor`` are
followed, and the names returned are prefixed so that they can be passed to
``set_params`` of the estimator as they are.
"""

from __future__ import annotations

import copy
import re
from typing import TYPE_CHECKING, Any

import sklearn
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from sklearn.base import BaseEstimator
from sklearn.compose import TransformedTargetRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR, LinearSVR

from yikit.helpers import is_installed

if TYPE_CHECKING:
    #: ``(model types, {name: distribution}, names bounded by n_features)``
    _SearchSpaceRow = tuple[
        type | tuple[type, ...],
        dict[str, BaseDistribution],
        tuple[str, ...],
    ]
    #: ``(model types, {name: recommended value})``
    _RecommendedParamsRow = tuple[type | tuple[type, ...], dict[str, Any]]


def _criterion_choices(version: str | None = None) -> list[str]:
    """Return the tree criteria usable without warnings in scikit-learn.

    ``"squared_error"`` was added in scikit-learn 1.0 (replacing ``"mse"``),
    and ``"friedman_mse"`` is deprecated since scikit-learn 1.9.

    Parameters
    ----------
    version : str or None, default=None
        Version string of scikit-learn such as ``"1.9.1"``. ``None`` means the
        installed scikit-learn. Only the first two numbers are compared.

    Returns
    -------
    list of str
        ``["mse", "friedman_mse"]`` before 1.0,
        ``["squared_error", "friedman_mse"]`` before 1.9, and
        ``["squared_error"]`` otherwise.

    Raises
    ------
    ValueError
        If ``version`` does not start with two dot-separated numbers.
    """
    version = sklearn.__version__ if version is None else version
    match = re.match(r"(\d+)\.(\d+)", version)
    if match is None:
        raise ValueError(
            f"Cannot parse the scikit-learn version {version!r}; expected "
            "it to start with '<major>.<minor>'."
        )
    major_minor = (int(match.group(1)), int(match.group(2)))
    if major_minor < (1, 0):
        return ["mse", "friedman_mse"]
    if major_minor < (1, 9):
        return ["squared_error", "friedman_mse"]
    return ["squared_error"]


def _svr_distributions() -> dict[str, BaseDistribution]:
    """Return the distributions shared by SVR and LinearSVR."""
    return {
        "C": FloatDistribution(low=2**-5, high=2**10, log=True),
        "epsilon": FloatDistribution(low=2**-10, high=2**0, log=True),
    }


def _alpha_distributions() -> dict[str, BaseDistribution]:
    """Return the regularization strength shared by the linear models."""
    return {"alpha": FloatDistribution(low=0.1, high=10, log=True)}


def _build_search_spaces() -> tuple[_SearchSpaceRow, ...]:
    """Build the search space table in its matching order.

    The rows of lightgbm and ngboost are included only when the package is
    installed. ``Lasso`` comes before ``ElasticNet`` because ``Lasso`` is a
    subclass of ``ElasticNet`` and does not have ``l1_ratio``.

    Returns
    -------
    tuple of tuple
        Rows of ``(model types, {name: distribution}, names whose upper bound
        is clipped by the number of features)``.
    """
    rows: list[_SearchSpaceRow] = []
    if is_installed("lightgbm"):
        from lightgbm import (  # type: ignore[reportMissingImports]
            LGBMRegressor,
        )

        from yikit.models._gbdt import GBDTRegressor

        rows.append(
            (
                (GBDTRegressor, LGBMRegressor),
                {
                    "n_estimators": IntDistribution(
                        low=10, high=1000, log=True
                    ),
                    "min_child_weight": FloatDistribution(
                        low=0.001, high=10, log=True
                    ),
                    "colsample_bytree": FloatDistribution(low=0.6, high=0.95),
                    "subsample": FloatDistribution(low=0.6, high=0.95),
                    "num_leaves": IntDistribution(
                        low=2**3, high=2**9, log=True
                    ),
                },
                (),
            )
        )
    rows += [
        (
            RandomForestRegressor,
            {
                "min_samples_split": IntDistribution(low=2, high=16),
                "max_depth": IntDistribution(low=10, high=100),
                "n_estimators": IntDistribution(low=10, high=1000, log=True),
            },
            (),
        ),
        (SVR, _svr_distributions(), ()),
        (LinearSVR, _svr_distributions(), ()),
        (
            MLPRegressor,
            {
                "hidden_layer_sizes": IntDistribution(low=50, high=300),
                "alpha": FloatDistribution(low=1e-5, high=1e-3, log=True),
                "learning_rate_init": FloatDistribution(
                    low=1e-5, high=1e-3, log=True
                ),
            },
            (),
        ),
    ]
    if is_installed("ngboost"):
        from ngboost import (  # type: ignore[reportMissingImports]
            NGBRegressor,
        )

        rows.append(
            (
                NGBRegressor,
                {
                    "Base__max_depth": IntDistribution(low=2, high=100),
                    "Base__criterion": CategoricalDistribution(
                        choices=_criterion_choices()
                    ),
                    "n_estimators": IntDistribution(
                        low=10, high=1000, log=True
                    ),
                    "minibatch_frac": FloatDistribution(low=0.5, high=1.0),
                },
                (),
            )
        )
    rows += [
        (
            PLSRegression,
            {"n_components": IntDistribution(low=1, high=10)},
            ("n_components",),
        ),
        (Lasso, _alpha_distributions(), ()),
        (
            ElasticNet,
            {
                **_alpha_distributions(),
                "l1_ratio": FloatDistribution(low=0.1, high=1.0),
            },
            (),
        ),
        (Ridge, _alpha_distributions(), ()),
    ]
    return tuple(rows)


#: The search space table. Rows are matched from the top with ``isinstance``.
_SEARCH_SPACES: tuple[_SearchSpaceRow, ...] = _build_search_spaces()

#: The recommended values table. Rows are matched from the top with
#: ``isinstance``. The values are applied only when the user passes them.
_RECOMMENDED_PARAMS: tuple[_RecommendedParamsRow, ...] = (
    (SVR, {"gamma": "auto"}),
)


def _build_ignores_nested_set_params() -> tuple[type, ...]:
    """Return the model types whose own ``set_params`` ignores nested names.

    Since ngboost 0.4.0, ``set_params`` of ``NGBRegressor`` only calls
    ``setattr``, so a name such as ``Base__max_depth`` does not reach its
    ``Base``. Older ngboost inherits ``set_params`` from scikit-learn and
    applies nested names. The type is included only when ngboost is
    installed and overrides ``set_params``.

    Returns
    -------
    tuple of type
        The model types, or an empty tuple when none is installed.
    """
    if is_installed("ngboost"):
        from ngboost import (  # type: ignore[reportMissingImports]
            NGBRegressor,
        )

        if NGBRegressor.set_params is not BaseEstimator.set_params:
            return (NGBRegressor,)
    return ()


#: Model types whose own ``set_params`` does not apply nested names.
_IGNORES_NESTED_SET_PARAMS: tuple[type, ...] = (
    _build_ignores_nested_set_params()
)


def resolve_estimator(estimator: Any) -> tuple[str, Any]:
    """Find the model to search inside nested estimators.

    The last step of a ``Pipeline`` and the ``regressor`` of a
    ``TransformedTargetRegressor`` are followed recursively, and the names
    on the way are joined into a prefix of the parameter names. The steps
    before the last one and the ``transformer`` are not followed. Any other
    estimator is the model itself.

    Parameters
    ----------
    estimator : object
        Estimator instance to resolve, e.g.
        ``TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(),
        SVR()))``.

    Returns
    -------
    prefix : str
        Prefix to add to the parameter names of the model so that
        ``estimator.set_params`` accepts them, e.g. ``"regressor__svr__"``.
        ``""`` when ``estimator`` is not nested.
    model : object
        The innermost model, not a copy. ``None`` when the ``regressor`` of
        a ``TransformedTargetRegressor`` is ``None``.

    Raises
    ------
    ValueError
        If a ``Pipeline`` on the way has no steps.

    Examples
    --------
    >>> from sklearn.compose import TransformedTargetRegressor
    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.svm import SVR
    >>> resolve_estimator(
    ...     TransformedTargetRegressor(
    ...         regressor=make_pipeline(StandardScaler(), SVR())
    ...     )
    ... )
    ('regressor__svr__', SVR())
    """
    if isinstance(estimator, Pipeline):
        if len(estimator.steps) == 0:
            raise ValueError(
                "Cannot find the model to search in a Pipeline with no steps."
            )
        step_name, step = estimator.steps[-1]
        prefix, model = resolve_estimator(step)
        return f"{step_name}__{prefix}", model
    if isinstance(estimator, TransformedTargetRegressor):
        prefix, model = resolve_estimator(estimator.regressor)
        return f"regressor__{prefix}", model
    return "", estimator


def _find_search_space_row(model: Any) -> _SearchSpaceRow | None:
    """Return the first row whose model types match ``model``.

    Parameters
    ----------
    model : object
        Model to match. Instances of subclasses match the row of their
        parent class.

    Returns
    -------
    tuple or None
        The first matching row of the table, or ``None`` when no row matches.
    """
    return next(
        (row for row in _SEARCH_SPACES if isinstance(model, row[0])), None
    )


def _clip_high(
    distribution: BaseDistribution, n_features: int
) -> IntDistribution:
    """Return a new distribution whose upper bound is at most ``n_features``.

    Parameters
    ----------
    distribution : optuna.distributions.IntDistribution
        Distribution to clip.
    n_features : int
        Number of features.

    Returns
    -------
    optuna.distributions.IntDistribution
        New distribution with ``high = min(high, n_features)``.

    Raises
    ------
    TypeError
        If ``distribution`` is not an ``IntDistribution``.
    """
    if not isinstance(distribution, IntDistribution):
        raise TypeError(
            "Only an IntDistribution can be bounded by the number of "
            f"features, got {type(distribution).__name__}."
        )
    return IntDistribution(
        low=distribution.low,
        high=min(distribution.high, n_features),
        log=distribution.log,
        step=distribution.step,
    )


def get_search_space(
    estimator: Any, *, n_features: int | None = None
) -> dict[str, BaseDistribution] | None:
    """Return the distributions to search for ``estimator``.

    The model to search is found with ``resolve_estimator``, so a model
    nested in a ``Pipeline`` or a ``TransformedTargetRegressor`` is
    searched with prefixed names. The rows of the search space table are
    matched from the top with ``isinstance``, and the first matching row is
    used. Subclasses of a registered model therefore get the row of their
    parent class.

    Parameters
    ----------
    estimator : object
        Estimator instance to look up, e.g. ``sklearn.svm.SVR()`` or a
        ``Pipeline`` whose last step is one.
    n_features : int or None, default=None
        Number of features of the data passed to ``estimator`` (the
        outermost one when nested). When given, the upper bound of the
        parameters limited by the data (``n_components`` of
        ``PLSRegression``) becomes ``min(high, n_features)``. When ``None``,
        the bounds of the table are used as they are.

    Returns
    -------
    dict of str to optuna.distributions.BaseDistribution or None
        Newly created mapping from the prefixed parameter names to
        distributions, or ``None`` when no row matches the resolved model.
        Neither the dict nor the distributions are shared with the table or
        with other calls.

    Raises
    ------
    ValueError
        If ``n_features`` is smaller than 1, or if a ``Pipeline`` on the way
        has no steps.

    Examples
    --------
    >>> from sklearn.linear_model import ElasticNet
    >>> sorted(get_search_space(ElasticNet()))
    ['alpha', 'l1_ratio']
    >>> from sklearn.compose import TransformedTargetRegressor
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.svm import SVR
    >>> sorted(
    ...     get_search_space(
    ...         TransformedTargetRegressor(
    ...             regressor=Pipeline(
    ...                 [("scaler", StandardScaler()), ("svr", SVR())]
    ...             )
    ...         )
    ...     )
    ... )
    ['regressor__svr__C', 'regressor__svr__epsilon']
    """
    if n_features is not None and n_features < 1:
        raise ValueError(
            f"n_features must be a positive integer, got {n_features!r}."
        )
    prefix, model = resolve_estimator(estimator)
    row = _find_search_space_row(model)
    if row is None:
        return None
    _, distributions, bounded_names = row
    return {
        f"{prefix}{name}": (
            _clip_high(distribution, n_features)
            if n_features is not None and name in bounded_names
            else copy.deepcopy(distribution)
        )
        for name, distribution in distributions.items()
    }


def get_recommended_params(estimator: Any) -> dict[str, Any]:
    """Return the values yikit recommends to fix for ``estimator``.

    The model is found with ``resolve_estimator`` and the rows of the
    recommended values table are matched from the top with ``isinstance``,
    in the same way as ``get_search_space``. The values are never applied
    unless the caller passes them, e.g. to ``set_params``.

    Parameters
    ----------
    estimator : object
        Estimator instance to look up, e.g. ``sklearn.svm.SVR()`` or a
        ``Pipeline`` whose last step is one.

    Returns
    -------
    dict of str to object
        Newly created mapping from the prefixed parameter names to the
        recommended values. Empty when no row matches the resolved model,
        including models without a search space.

    Raises
    ------
    ValueError
        If a ``Pipeline`` on the way has no steps.

    Examples
    --------
    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.svm import SVR
    >>> get_recommended_params(SVR())
    {'gamma': 'auto'}
    >>> get_recommended_params(make_pipeline(StandardScaler(), SVR()))
    {'svr__gamma': 'auto'}
    >>> from sklearn.linear_model import Ridge
    >>> get_recommended_params(Ridge())
    {}
    """
    prefix, model = resolve_estimator(estimator)
    row = next(
        (row for row in _RECOMMENDED_PARAMS if isinstance(model, row[0])),
        None,
    )
    if row is None:
        return {}
    _, values = row
    return {
        f"{prefix}{name}": copy.deepcopy(value)
        for name, value in values.items()
    }


def ignores_nested_set_params(model: Any) -> bool:
    """Return whether ``model.set_params`` ignores nested parameter names.

    Since ngboost 0.4.0, ``set_params`` of ``NGBRegressor`` only calls
    ``setattr``, so a name such as ``Base__max_depth`` does not reach its
    ``Base``. Such names are applied by ``Objective`` but not by
    ``OptunaSearchCV``.

    Parameters
    ----------
    model : object
        Model to check, usually the model returned by ``resolve_estimator``.
        Nested estimators are not resolved here.

    Returns
    -------
    bool
        ``True`` for ``NGBRegressor`` (and its subclasses) when the
        installed ngboost overrides ``set_params``, ``False`` otherwise.
    """
    return isinstance(model, _IGNORES_NESTED_SET_PARAMS)
