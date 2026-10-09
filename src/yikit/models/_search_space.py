"""Search spaces of the models registered in yikit.

This module holds the single table of optuna distributions that both
``Objective`` and ``ParamDistributions`` search. Each row of the table maps
model types to the distributions of their parameters. The table holds only
the searched parameters: fixed values such as ``n_jobs`` and
``random_state`` are never part of it, so the parameters of the estimator
passed by the user are kept as they are.
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
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR, LinearSVR

from yikit.helpers import is_installed

if TYPE_CHECKING:
    #: ``(model types, {name: distribution}, names bounded by n_features)``
    _SearchSpaceRow = tuple[
        type | tuple[type, ...],
        dict[str, BaseDistribution],
        tuple[str, ...],
    ]


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

    The rows of the search space table are matched from the top with
    ``isinstance``, and the first matching row is used. Subclasses of a
    registered model therefore get the row of their parent class.

    Parameters
    ----------
    estimator : object
        Estimator instance to look up, e.g. ``sklearn.svm.SVR()``.
    n_features : int or None, default=None
        Number of features of the data. When given, the upper bound of the
        parameters limited by the data (``n_components`` of
        ``PLSRegression``) becomes ``min(high, n_features)``. When ``None``,
        the bounds of the table are used as they are.

    Returns
    -------
    dict of str to optuna.distributions.BaseDistribution or None
        Newly created mapping from parameter names to distributions, or
        ``None`` when no row matches ``estimator``. Neither the dict nor the
        distributions are shared with the table or with other calls.

    Raises
    ------
    ValueError
        If ``n_features`` is smaller than 1.

    Examples
    --------
    >>> from sklearn.linear_model import ElasticNet
    >>> sorted(get_search_space(ElasticNet()))
    ['alpha', 'l1_ratio']
    """
    if n_features is not None and n_features < 1:
        raise ValueError(
            f"n_features must be a positive integer, got {n_features!r}."
        )
    row = _find_search_space_row(estimator)
    if row is None:
        return None
    _, distributions, bounded_names = row
    return {
        name: (
            _clip_high(distribution, n_features)
            if n_features is not None and name in bounded_names
            else copy.deepcopy(distribution)
        )
        for name, distribution in distributions.items()
    }
