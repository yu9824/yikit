"""Tests of ``yikit.models.EnsembleRegressor``."""

from __future__ import annotations

import inspect
import logging
import subprocess
import sys
import textwrap
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest
import sklearn
from joblib import parallel_backend
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.datasets import make_regression
from sklearn.ensemble import (
    RandomForestRegressor,
    StackingRegressor,
    VotingRegressor,
)
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

from yikit.models import EnsembleRegressor

if TYPE_CHECKING:
    from collections.abc import Iterator

SEED = 334
N_SAMPLES = 60
N_FEATURES = 4
CV = 3
N_TRIALS = 2
METHODS = ("average", "stacking", "blending")

#: ``feature_names_in_`` exists from scikit-learn 1.0.
HAS_FEATURE_NAMES = tuple(
    int(part) for part in sklearn.__version__.split(".")[:2]
) >= (1, 0)

#: Small data shared by the tests.
X, y = make_regression(
    n_samples=N_SAMPLES,
    n_features=N_FEATURES,
    noise=1.0,
    random_state=SEED,
)
COLUMNS = [f"x{i}" for i in range(N_FEATURES)]


def _models() -> list[BaseEstimator]:
    """Return new models whose results do not depend on a random state."""
    return [
        Ridge(random_state=SEED),
        SVR(),
        RandomForestRegressor(n_estimators=20, random_state=0),
    ]


def _named_models() -> list[tuple[str, BaseEstimator]]:
    """Return ``_models`` with the names that EnsembleRegressor gives them."""
    return list(zip(["ridge", "svr", "randomforestregressor"], _models()))


def _sklearn_ensemble(method: str) -> BaseEstimator:
    """Build by hand the scikit-learn ensemble of ``method``."""
    if method == "average":
        return VotingRegressor(_named_models())
    if method == "stacking":
        return StackingRegressor(
            _named_models(), final_estimator=LinearRegression(), cv=CV
        )
    return StackingRegressor(
        _named_models(),
        final_estimator=LinearRegression(positive=True, fit_intercept=False),
        cv=CV,
    )


def _random_models() -> list[BaseEstimator]:
    """Return new models with given and unspecified ``random_state``.

    EnsembleRegressor names them ``"randomforestregressor-1"``,
    ``"randomforestregressor-2"``, ``"pipeline"`` and ``"svr"``.
    """
    return [
        RandomForestRegressor(n_estimators=10, n_jobs=1, random_state=7),
        RandomForestRegressor(n_estimators=10, max_depth=3),
        make_pipeline(
            QuantileTransformer(n_quantiles=10),
            RandomForestRegressor(n_estimators=10, min_samples_leaf=2),
        ),
        SVR(C=2.0),
    ]


#: The names of ``_random_models`` that the ``random_state`` rule sets.
_UNSPECIFIED_RANDOM_STATES = (
    (),
    ("random_state",),
    (
        "quantiletransformer__random_state",
        "randomforestregressor__random_state",
    ),
    (),
)


def _holds_estimator(value: object) -> bool:
    """Return whether ``value`` is an estimator or a sequence holding one."""
    if hasattr(value, "get_params") and not isinstance(value, type):
        return True
    return isinstance(value, (list, tuple)) and any(
        _holds_estimator(item) for item in value
    )


def _plain_params(estimator: BaseEstimator) -> dict[str, object]:
    """Return the deep parameters of ``estimator`` that hold no estimator."""
    return {
        name: value
        for name, value in estimator.get_params(deep=True).items()
        if not _holds_estimator(value)
    }


def _skip_without_search_cv() -> None:
    """Skip the test when the tuning (optuna-integration) is unavailable."""
    pytest.importorskip("optuna")
    try:
        import yikit.models._search_cv  # noqa: F401
    except ImportError:
        pytest.skip("OptunaSearchCV (optuna-integration) is not installed")


def _tuning_classes() -> tuple[type, type, type]:
    """Return the classes of the tuning, or skip when they are unavailable.

    Returns
    -------
    tuple of type
        ``OptunaSearchCV`` of optuna-integration, ``OptunaSearchRegressor``
        (which ``EnsembleRegressor`` wraps each model in) and
        ``ParamDistributions``.
    """
    _skip_without_search_cv()
    try:
        from optuna_integration import OptunaSearchCV
    except ImportError:  # old optuna that still bundles the integration
        from optuna.integration import OptunaSearchCV

    from yikit.models._optuna import ParamDistributions
    from yikit.models._search_cv import OptunaSearchRegressor

    return OptunaSearchCV, OptunaSearchRegressor, ParamDistributions


class _RecordsHandler(logging.Handler):
    """Logging handler that keeps the records it receives."""

    def __init__(self) -> None:
        super().__init__(level=logging.NOTSET)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def optuna_records() -> Iterator[list[logging.LogRecord]]:
    """Collect the records that reach the root logger of optuna.

    optuna does not propagate its records to the root logger (so ``caplog``
    does not see them); its loggers (``optuna.*``) all pass their records
    to the handlers of the ``optuna`` logger.
    """
    handler = _RecordsHandler()
    logger = logging.getLogger("optuna")
    logger.addHandler(handler)
    try:
        yield handler.records
    finally:
        logger.removeHandler(handler)


def _block_imports(
    monkeypatch: pytest.MonkeyPatch,
    blocked: tuple[str, ...],
    reimported: tuple[str, ...],
) -> None:
    """Make the modules ``blocked`` fail to import during the test.

    The modules ``reimported`` are removed from ``sys.modules``, so that the
    next import runs them again (and fails on a blocked module).
    ``monkeypatch`` puts back ``sys.modules`` and the modules of the tuning
    as attributes of ``yikit.models`` (which a successful import would
    replace) after the test.
    """
    package = sys.modules["yikit.models"]
    for attribute in ("_optuna", "_search_cv"):
        if hasattr(package, attribute):
            monkeypatch.setattr(
                package, attribute, getattr(package, attribute)
            )
    for name in reimported:
        monkeypatch.delitem(sys.modules, name, raising=False)
    for name in blocked:
        monkeypatch.setitem(sys.modules, name, None)


class _NoFeatureCountRegressor(RegressorMixin, BaseEstimator):
    """Regressor that predicts the mean and records no ``n_features_in_``.

    Some regressors (e.g. ``NGBRegressor`` of ngboost) do not record it.
    """

    def fit(self, X, y):
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self.mean_)


# --- Ensembles of scikit-learn -----------------------------------------------


@pytest.mark.parametrize("method", METHODS)
def test_predictions_match_the_sklearn_ensemble(method):
    ensemble = EnsembleRegressor(
        estimators=_models(), method=method, cv=CV, opt=False
    )
    ensemble.fit(X, y)
    expected = _sklearn_ensemble(method).fit(X, y)

    assert_allclose(ensemble.predict(X), expected.predict(X))


@pytest.mark.parametrize("method", METHODS)
def test_predictions_use_the_models_refitted_on_all_the_data(method):
    ensemble = EnsembleRegressor(
        estimators=_models(), method=method, cv=CV, opt=False
    ).fit(X, y)

    for model, refitted in zip(_models(), ensemble.estimators_):
        assert_allclose(refitted.predict(X), clone(model).fit(X, y).predict(X))


@pytest.mark.parametrize("method", METHODS)
def test_one_model(method):
    ensemble = EnsembleRegressor(
        estimators=[Ridge(random_state=SEED)], method=method, cv=CV, opt=False
    ).fit(X, y)

    assert list(ensemble.named_estimators_) == ["ridge"]
    assert ensemble.predict(X).shape == (N_SAMPLES,)
    if method == "average":
        assert_allclose(
            ensemble.predict(X),
            Ridge(random_state=SEED).fit(X, y).predict(X),
        )


def test_stacking_final_estimator_is_a_linear_regression():
    ensemble = EnsembleRegressor(
        estimators=_models(), method="stacking", cv=CV, opt=False
    ).fit(X, y)

    assert type(ensemble.final_estimator_) is LinearRegression
    assert ensemble.final_estimator_.get_params() == (
        LinearRegression().get_params()
    )
    assert ensemble.weights_ is ensemble.final_estimator_.coef_


def test_blending_weights_are_non_negative_without_intercept():
    ensemble = EnsembleRegressor(
        estimators=_models(), method="blending", cv=CV, opt=False
    ).fit(X, y)

    assert type(ensemble.final_estimator_) is LinearRegression
    assert ensemble.final_estimator_.positive is True
    assert ensemble.final_estimator_.fit_intercept is False
    assert ensemble.weights_ is ensemble.final_estimator_.coef_
    assert ensemble.weights_.shape == (len(_models()),)
    assert np.all(ensemble.weights_ >= 0)


def test_average_has_no_weights_and_no_final_estimator():
    ensemble = EnsembleRegressor(
        estimators=_models(), method="average", opt=False
    ).fit(X, y)

    assert ensemble.weights_ is None
    assert not hasattr(ensemble, "final_estimator_")


def test_refitting_with_average_removes_the_final_estimator():
    ensemble = EnsembleRegressor(
        estimators=_models(), method="stacking", cv=CV, opt=False
    ).fit(X, y)
    ensemble.set_params(method="average").fit(X, y)

    assert ensemble.weights_ is None
    assert not hasattr(ensemble, "final_estimator_")


# --- Names of the models -----------------------------------------------------


def test_models_are_named_as_in_make_pipeline():
    ensemble = EnsembleRegressor(
        estimators=[Ridge(), SVR(), Ridge(alpha=2.0)],
        method="average",
        opt=False,
    ).fit(X, y)

    assert list(ensemble.named_estimators_) == ["ridge-1", "svr", "ridge-2"]
    assert ensemble.named_estimators_["ridge-2"].alpha == 2.0
    assert [name for name, _ in ensemble.estimator_.estimators] == [
        "ridge-1",
        "svr",
        "ridge-2",
    ]


def test_named_pairs_are_used_as_they_are():
    ensemble = EnsembleRegressor(
        estimators=[("first", Ridge()), ("second", SVR()), Ridge()],
        method="stacking",
        cv=CV,
        opt=False,
    ).fit(X, y)

    assert list(ensemble.named_estimators_) == ["first", "second", "ridge"]
    assert isinstance(ensemble.named_estimators_["second"], SVR)


# --- Fitted attributes -------------------------------------------------------


@pytest.mark.parametrize(
    "method, ensemble_type",
    [
        ("average", VotingRegressor),
        ("stacking", StackingRegressor),
        ("blending", StackingRegressor),
    ],
)
def test_fitted_attributes(method, ensemble_type):
    models = _models()
    ensemble = EnsembleRegressor(
        estimators=models, method=method, cv=CV, opt=False
    ).fit(X, y)

    assert type(ensemble.estimator_) is ensemble_type
    assert ensemble.estimators_ is ensemble.estimator_.estimators_
    assert len(ensemble.estimators_) == len(models)
    for model, fitted in zip(models, ensemble.estimators_):
        assert type(fitted) is type(model)
        assert fitted is not model
        assert fitted.get_params() == model.get_params()
    assert list(ensemble.named_estimators_) == [
        name for name, _ in _named_models()
    ]
    assert ensemble.n_features_in_ == N_FEATURES
    assert not hasattr(ensemble, "feature_names_in_")
    assert not hasattr(ensemble, "results_")
    if method != "average":
        assert (
            ensemble.final_estimator_ is ensemble.estimator_.final_estimator_
        )
    assert_allclose(ensemble.predict(X), ensemble.estimator_.predict(X))


@pytest.mark.skipif(
    not HAS_FEATURE_NAMES, reason="feature_names_in_ needs scikit-learn 1.0"
)
@pytest.mark.parametrize("method", METHODS)
def test_feature_names_of_a_dataframe(method):
    X_frame = pd.DataFrame(X, columns=COLUMNS)
    ensemble = EnsembleRegressor(
        estimators=_models(), method=method, cv=CV, opt=False
    ).fit(X_frame, y)

    assert list(ensemble.feature_names_in_) == COLUMNS
    assert ensemble.n_features_in_ == N_FEATURES
    assert_allclose(
        ensemble.predict(X_frame),
        _sklearn_ensemble(method).fit(X_frame, y).predict(X_frame),
    )

    # They are removed when it is fitted again on data without names.
    ensemble.fit(X, y)
    assert not hasattr(ensemble, "feature_names_in_")


@pytest.mark.parametrize("method", METHODS)
def test_number_of_features_without_the_attribute_in_the_models(method):
    ensemble = EnsembleRegressor(
        estimators=[_NoFeatureCountRegressor()],
        method=method,
        cv=CV,
        opt=False,
    ).fit(X, y)

    assert ensemble.n_features_in_ == N_FEATURES
    assert ensemble.predict(X).shape == (N_SAMPLES,)


def test_predict_before_fit_raises_not_fitted_error():
    ensemble = EnsembleRegressor(estimators=[Ridge()], opt=False)

    with pytest.raises(NotFittedError):
        ensemble.predict(X)


# --- Parameters --------------------------------------------------------------


def test_default_parameters():
    parameters = inspect.signature(EnsembleRegressor).parameters

    assert list(parameters) == [
        "estimators",
        "method",
        "cv",
        "n_jobs",
        "random_state",
        "scoring",
        "verbose",
        "opt",
        "n_trials",
    ]
    default_estimators = parameters["estimators"].default
    assert isinstance(default_estimators, tuple)
    assert len(default_estimators) == 1
    assert type(default_estimators[0]) is RandomForestRegressor
    assert {
        name: parameter.default
        for name, parameter in parameters.items()
        if name != "estimators"
    } == {
        "method": "blending",
        "cv": 5,
        "n_jobs": None,
        "random_state": None,
        "scoring": "neg_mean_squared_error",
        "verbose": 0,
        "opt": True,
        "n_trials": 100,
    }


def test_boruta_argument_is_removed():
    with pytest.raises(TypeError, match="boruta"):
        EnsembleRegressor(boruta=False)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("n_jobs", [None, 2])
def test_n_jobs_is_passed_to_the_sklearn_ensemble(method, n_jobs):
    ensemble = EnsembleRegressor(
        estimators=_models(), method=method, cv=CV, n_jobs=n_jobs, opt=False
    )
    with parallel_backend("threading"):
        ensemble.fit(X, y)

    assert ensemble.estimator_.n_jobs == n_jobs
    assert_allclose(
        ensemble.predict(X), _sklearn_ensemble(method).fit(X, y).predict(X)
    )


@pytest.mark.parametrize(
    "method, verbose, expected",
    [
        ("average", 0, False),
        ("average", 2, True),
        ("stacking", 0, 0),
        ("stacking", 2, 2),
        ("blending", 2, 2),
    ],
)
def test_verbose_is_passed_to_the_sklearn_ensemble(method, verbose, expected):
    ensemble = EnsembleRegressor(
        estimators=[Ridge()], method=method, cv=CV, verbose=verbose, opt=False
    ).fit(X, y)

    assert ensemble.estimator_.verbose == expected
    assert type(ensemble.estimator_.verbose) is type(expected)


def test_cv_is_passed_to_the_stacking():
    ensemble = EnsembleRegressor(
        estimators=[Ridge()], method="stacking", cv=4, opt=False
    ).fit(X, y)

    assert ensemble.estimator_.cv == 4


def test_get_params_set_params_and_clone_keep_the_parameters():
    models = [Ridge(alpha=0.5), SVR(C=2.0)]
    params = {
        "estimators": models,
        "method": "stacking",
        "cv": CV,
        "n_jobs": 2,
        "random_state": SEED,
        "scoring": "r2",
        "verbose": 1,
        "opt": False,
        "n_trials": 7,
    }
    ensemble = EnsembleRegressor(**params)

    assert ensemble.get_params(deep=False) == params
    assert ensemble.estimators is models

    cloned = clone(ensemble)
    cloned_params = cloned.get_params(deep=False)
    assert {k: v for k, v in cloned_params.items() if k != "estimators"} == {
        k: v for k, v in params.items() if k != "estimators"
    }
    assert [type(model) for model in cloned_params["estimators"]] == [
        Ridge,
        SVR,
    ]
    assert [model.get_params() for model in cloned_params["estimators"]] == [
        model.get_params() for model in models
    ]

    assert ensemble.set_params(method="average", n_trials=3) is ensemble
    assert ensemble.method == "average"
    assert ensemble.n_trials == 3


# --- Parameters of the models and the random_state rule ----------------------


@pytest.mark.parametrize("method", METHODS)
def test_given_n_jobs_and_random_state_of_a_model_are_kept(method):
    ensemble = EnsembleRegressor(
        estimators=[
            RandomForestRegressor(n_estimators=10, n_jobs=1, random_state=7)
        ],
        method=method,
        cv=CV,
        random_state=SEED,
        opt=False,
    ).fit(X, y)

    for fitted in (
        ensemble.estimators_[0],
        ensemble.named_estimators_["randomforestregressor"],
    ):
        assert fitted.n_jobs == 1
        assert fitted.random_state == 7


@pytest.mark.parametrize("method", METHODS)
def test_unspecified_random_states_get_model_random_state(method):
    ensemble = EnsembleRegressor(
        estimators=_random_models(),
        method=method,
        cv=CV,
        random_state=SEED,
        opt=False,
    ).fit(X, y)
    seed = ensemble.model_random_state_

    assert type(seed) is int
    named = ensemble.named_estimators_
    assert list(named) == [
        "randomforestregressor-1",
        "randomforestregressor-2",
        "pipeline",
        "svr",
    ]
    assert named["randomforestregressor-1"].random_state == 7
    assert named["randomforestregressor-2"].random_state == seed
    pipeline_steps = named["pipeline"].named_steps
    assert pipeline_steps["quantiletransformer"].random_state == seed
    assert pipeline_steps["randomforestregressor"].random_state == seed
    assert "random_state" not in named["svr"].get_params()


@pytest.mark.parametrize("method", METHODS)
def test_other_parameters_are_the_given_ones(method):
    models = _random_models()
    ensemble = EnsembleRegressor(
        estimators=models, method=method, cv=CV, random_state=SEED, opt=False
    ).fit(X, y)

    for model, fitted, names in zip(
        models, ensemble.estimators_, _UNSPECIFIED_RANDOM_STATES
    ):
        expected = {
            **_plain_params(model),
            **{name: ensemble.model_random_state_ for name in names},
        }
        assert _plain_params(fitted) == expected


def test_models_without_random_state_are_used_as_given():
    models = [
        SVR(C=2.0),
        KNeighborsRegressor(n_neighbors=3),
        LinearRegression(fit_intercept=False),
    ]
    ensemble = EnsembleRegressor(
        estimators=models, method="average", random_state=SEED, opt=False
    ).fit(X, y)

    for model, fitted in zip(models, ensemble.estimators_):
        assert fitted.get_params() == model.get_params()
    expected = VotingRegressor(
        [(type(model).__name__.lower(), clone(model)) for model in models]
    ).fit(X, y)
    assert_allclose(ensemble.predict(X), expected.predict(X))


@pytest.mark.parametrize("method", METHODS)
def test_same_random_state_gives_the_same_predictions(method):
    models = [
        RandomForestRegressor(n_estimators=10),
        make_pipeline(
            QuantileTransformer(n_quantiles=10),
            RandomForestRegressor(n_estimators=10),
        ),
    ]

    def fit(random_state: int) -> EnsembleRegressor:
        return EnsembleRegressor(
            estimators=models,
            method=method,
            cv=CV,
            random_state=random_state,
            opt=False,
        ).fit(X, y)

    first, second, other = fit(SEED), fit(SEED), fit(SEED + 1)

    assert first.model_random_state_ == second.model_random_state_
    assert_array_equal(first.predict(X), second.predict(X))
    assert other.model_random_state_ != first.model_random_state_
    assert not np.allclose(other.predict(X), first.predict(X))


@pytest.mark.parametrize("random_state_type", [int, np.random.RandomState])
def test_model_random_state_is_the_first_integer_drawn(random_state_type):
    expected = int(check_random_state(SEED).randint(2**31 - 1))
    ensemble = EnsembleRegressor(
        estimators=[RandomForestRegressor(n_estimators=10)],
        method="average",
        random_state=random_state_type(SEED),
        opt=False,
    ).fit(X, y)

    assert ensemble.model_random_state_ == expected
    assert ensemble.estimators_[0].random_state == expected


def test_given_models_are_not_changed():
    models = _random_models()
    given = list(models)
    params_before = [model.get_params(deep=True) for model in models]
    ensemble = EnsembleRegressor(
        estimators=models,
        method="stacking",
        cv=CV,
        random_state=SEED,
        opt=False,
    )
    ensemble.fit(X, y)

    assert ensemble.estimators is models
    assert all(model is before for model, before in zip(models, given))
    # The nested estimators are the same objects with the same parameters.
    assert [model.get_params(deep=True) for model in models] == params_before
    assert models[1].random_state is None
    for _, step in models[2].steps:
        assert step.random_state is None
    for fitted, model in zip(ensemble.estimators_, models):
        assert fitted is not model
    for unfitted in (models[0], models[1], *models[2].named_steps.values()):
        with pytest.raises(NotFittedError):
            check_is_fitted(unfitted)


def test_random_states_of_ngboost_are_set_on_a_copy():
    ngboost = pytest.importorskip("ngboost")
    base = DecisionTreeRegressor(max_depth=3)
    ngb = ngboost.NGBRegressor(Base=base, n_estimators=20, verbose=False)
    # ngboost may put the global RandomState of NumPy in place of None.
    random_state_before = ngb.random_state
    # NGBRegressor of ngboost >= 0.5 lacks the tags of a regressor that
    # scikit-learn >= 1.6 reads, so it is passed inside a regressor.
    model = TransformedTargetRegressor(regressor=ngb)
    ensemble = EnsembleRegressor(
        estimators=[model], method="average", random_state=SEED, opt=False
    ).fit(X, y)
    seed = ensemble.model_random_state_

    # The copy given to VotingRegressor (which clones it before fitting).
    prepared = ensemble.estimator_.estimators[0][1].regressor
    assert prepared is not ngb
    assert prepared.Base is not base
    assert prepared.Base.random_state == seed
    assert_array_equal(
        check_random_state(prepared.random_state).get_state()[1],
        np.random.RandomState(seed).get_state()[1],
    )
    assert ensemble.estimators_[0].regressor_.Base.random_state == seed
    assert model.regressor is ngb
    assert ngb.random_state is random_state_before
    assert ngb.Base is base
    assert base.random_state is None


def test_model_random_state_is_drawn_before_the_seeds_of_the_tuning():
    _skip_without_search_cv()
    models = [Ridge(), SVR()]
    ensemble = EnsembleRegressor(
        estimators=models,
        method="average",
        cv=CV,
        random_state=SEED,
        n_trials=1,
    ).fit(X, y)
    rng = check_random_state(SEED)
    expected_model_random_state = int(rng.randint(2**31 - 1))
    expected_seeds = [int(rng.randint(2**31 - 1)) for _ in models]

    assert ensemble.model_random_state_ == expected_model_random_state
    searches = ensemble.estimators_
    assert [search.random_state for search in searches] == expected_seeds
    assert all(type(search.random_state) is int for search in searches)
    # The rule is applied to the model tuned in each search.
    ridge, svr = (search.estimator for search in searches)
    assert ridge.random_state == ensemble.model_random_state_
    assert svr.get_params() == models[1].get_params()
    assert models[0].random_state is None


# --- Tuning ------------------------------------------------------------------


def _tuned_models() -> list[BaseEstimator]:
    """Return new models that have a search space, one of them nested.

    EnsembleRegressor names them ``"ridge"``, ``"pipeline"`` and ``"svr"``.
    """
    return [Ridge(), make_pipeline(StandardScaler(), Ridge()), SVR()]


@pytest.mark.parametrize("method", METHODS)
def test_tuned_models_are_optuna_search_cv(method):
    search_cv, search_regressor, param_distributions = _tuning_classes()
    models = _tuned_models()
    ensemble = EnsembleRegressor(
        estimators=models,
        method=method,
        cv=CV,
        random_state=SEED,
        n_trials=N_TRIALS,
    ).fit(X, y)

    assert list(ensemble.named_estimators_) == ["ridge", "pipeline", "svr"]
    for model, search, named in zip(
        models, ensemble.estimators_, ensemble.named_estimators_.values()
    ):
        assert type(search) is search_regressor
        assert isinstance(search, search_cv)
        assert named is search
        # The search space of the model for the columns of X.
        expected = param_distributions(model, n_features=N_FEATURES)
        assert dict(search.param_distributions) == dict(expected)
        assert search.param_distributions.n_features == N_FEATURES
        assert len(search.study_.trials) == N_TRIALS
        assert set(search.best_params_) == set(expected)
        # The model refitted with the best parameters.
        best_estimator_params = search.best_estimator_.get_params()
        for name, value in search.best_params_.items():
            assert best_estimator_params[name] == value
    # A nested model gets the names prefixed with its step.
    assert set(ensemble.named_estimators_["pipeline"].best_params_) == {
        "ridge__alpha"
    }
    assert ensemble.n_features_in_ == N_FEATURES
    assert_allclose(ensemble.predict(X), ensemble.estimator_.predict(X))


def test_n_trials_defaults_to_100():
    ensemble = EnsembleRegressor()

    assert ensemble.n_trials == 100
    assert ensemble.get_params()["n_trials"] == 100


@pytest.mark.parametrize(
    "scoring_params, expected_scoring",
    [({}, "neg_mean_squared_error"), ({"scoring": "r2"}, "r2")],
    ids=["default", "r2"],
)
def test_cv_scoring_and_n_trials_are_passed_to_each_search(
    scoring_params, expected_scoring
):
    _tuning_classes()
    cv = KFold(n_splits=CV, shuffle=True, random_state=0)
    ensemble = EnsembleRegressor(
        estimators=[Ridge(), SVR()],
        method="average",
        cv=cv,
        random_state=SEED,
        n_trials=N_TRIALS,
        **scoring_params,
    ).fit(X, y)

    # The searches that EnsembleRegressor gave to VotingRegressor.
    for _, search in ensemble.estimator_.estimators:
        assert search.cv is cv
    # The fitted copies of the searches.
    for search in ensemble.estimators_:
        params = search.get_params(deep=False)
        assert params["scoring"] == expected_scoring
        assert params["n_trials"] == N_TRIALS
        assert params["verbose"] == 0
        assert len(search.study_.trials) == N_TRIALS
        # Each trial is scored with ``scoring`` on the splits of ``cv``.
        for trial in search.study_.trials:
            assert sorted(
                key for key in trial.user_attrs if key.endswith("_test_score")
            ) == [
                "mean_test_score",
                *[f"split{i}_test_score" for i in range(CV)],
                "std_test_score",
            ]
        best_model = clone(search.estimator).set_params(**search.best_params_)
        assert search.study_.best_value == pytest.approx(
            cross_val_score(
                best_model, X, y, cv=cv, scoring=expected_scoring
            ).mean()
        )


@pytest.mark.parametrize("method", METHODS)
def test_stacking_tunes_on_the_training_part_of_each_split(
    method, monkeypatch
):
    _, search_regressor, _ = _tuning_classes()
    row_numbers = {row.tobytes(): i for i, row in enumerate(X)}
    fitted_rows: list[tuple[int, ...]] = []
    original_fit = search_regressor.fit

    def recording_fit(self, X_fit, y_fit=None, groups=None, **fit_params):
        fitted_rows.append(
            tuple(row_numbers[row.tobytes()] for row in np.asarray(X_fit))
        )
        return original_fit(self, X_fit, y_fit, groups=groups, **fit_params)

    monkeypatch.setattr(search_regressor, "fit", recording_fit)
    EnsembleRegressor(
        estimators=[Ridge()],
        method=method,
        cv=CV,
        random_state=SEED,
        n_trials=N_TRIALS,
    ).fit(X, y)

    all_rows = tuple(range(N_SAMPLES))
    if method == "average":
        assert fitted_rows == [all_rows]
        return
    # Tuned on the training part of each split of cv (never on its test
    # part), and once on all the data.
    split_rows = [tuple(train) for train, _ in KFold(n_splits=CV).split(X)]
    assert sorted(fitted_rows) == sorted([*split_rows, all_rows])
    assert sorted(map(len, fitted_rows)) == [
        N_SAMPLES * (CV - 1) // CV
    ] * CV + [N_SAMPLES]


@pytest.mark.parametrize("verbose, hidden", [(0, True), (1, False)])
def test_verbose_0_hides_the_trial_logs_of_optuna(
    verbose, hidden, optuna_records
):
    _tuning_classes()
    import optuna

    previous = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.INFO)
    try:
        EnsembleRegressor(
            estimators=[Ridge()],
            method="stacking",
            cv=CV,
            random_state=SEED,
            verbose=verbose,
            n_trials=N_TRIALS,
        ).fit(X, y)

        info_messages = [
            record.getMessage()
            for record in optuna_records
            if record.levelno < logging.WARNING
        ]
        assert optuna.logging.get_verbosity() == optuna.logging.INFO
    finally:
        optuna.logging.set_verbosity(previous)

    if hidden:
        assert info_messages == []
    else:  # the trial logs reach the handler when they are not hidden
        assert any("Trial" in message for message in info_messages)


@pytest.mark.parametrize("method", ["average", "stacking"])
def test_same_random_state_gives_the_same_tuning(method):
    _tuning_classes()

    def fit(random_state: int) -> EnsembleRegressor:
        return EnsembleRegressor(
            estimators=[Ridge(), SVR()],
            method=method,
            cv=CV,
            random_state=random_state,
            n_trials=N_TRIALS,
        ).fit(X, y)

    def best_params(ensemble: EnsembleRegressor) -> list[dict[str, object]]:
        return [search.best_params_ for search in ensemble.estimators_]

    first, second, other = fit(SEED), fit(SEED), fit(SEED + 1)

    for ensemble in (first, second):
        seeds = [search.random_state for search in ensemble.estimators_]
        assert all(type(seed) is int for seed in seeds)
    assert [search.random_state for search in first.estimators_] == [
        search.random_state for search in second.estimators_
    ]
    assert best_params(first) == best_params(second)
    assert_array_equal(first.predict(X), second.predict(X))
    assert best_params(other) != best_params(first)


@pytest.mark.parametrize(
    "estimator, name, type_name",
    [
        (KNeighborsRegressor(), "kneighborsregressor", "KNeighborsRegressor"),
        (("knn", KNeighborsRegressor()), "knn", "KNeighborsRegressor"),
        (
            ("pipe", make_pipeline(StandardScaler(), KNeighborsRegressor())),
            "pipe",
            "Pipeline",
        ),
    ],
    ids=["unnamed", "named", "pipeline"],
)
def test_model_without_search_space_raises_not_implemented_error(
    estimator, name, type_name
):
    _tuning_classes()
    ensemble = EnsembleRegressor(
        estimators=[Ridge(), estimator], cv=CV, n_trials=N_TRIALS
    )

    with pytest.raises(
        NotImplementedError, match=f"'{name}' \\({type_name}\\)"
    ) as excinfo:
        ensemble.fit(X, y)

    # Chained from the error of ParamDistributions, whose message is kept.
    cause = excinfo.value.__cause__
    assert isinstance(cause, NotImplementedError)
    assert "KNeighborsRegressor" in str(cause)
    assert str(cause) in str(excinfo.value)
    assert "opt=False" in str(excinfo.value)
    assert not hasattr(ensemble, "estimator_")


# --- Errors ------------------------------------------------------------------


@pytest.mark.parametrize(
    "estimators, name, type_name",
    [
        ([Ridge(), StandardScaler()], "standardscaler", "StandardScaler"),
        ([("clf", LogisticRegression())], "clf", "LogisticRegression"),
        ([Ridge(), ("nothing", None)], "nothing", "NoneType"),
        ([("pipe", Pipeline([("scaler", StandardScaler())]))], "pipe", ""),
    ],
    ids=["transformer", "classifier", "none", "pipeline-of-a-transformer"],
)
def test_non_regressor_raises_value_error(estimators, name, type_name):
    ensemble = EnsembleRegressor(estimators=estimators, opt=False)

    with pytest.raises(ValueError, match=f"'{name}'.*{type_name}"):
        ensemble.fit(X, y)


def test_pipeline_ending_with_a_regressor_is_accepted():
    pipeline = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())])
    ensemble = EnsembleRegressor(
        estimators=[pipeline], method="stacking", cv=CV, opt=False
    ).fit(X, y)

    assert list(ensemble.named_estimators_) == ["pipeline"]
    assert_allclose(
        ensemble.predict(X),
        StackingRegressor(
            [("pipeline", clone(pipeline))],
            final_estimator=LinearRegression(),
            cv=CV,
        )
        .fit(X, y)
        .predict(X),
    )


@pytest.mark.parametrize("estimators", [[], ()])
def test_empty_estimators_raise_value_error(estimators):
    ensemble = EnsembleRegressor(estimators=estimators, opt=False)

    with pytest.raises(ValueError, match="estimators"):
        ensemble.fit(X, y)


def test_unknown_method_raises_value_error_listing_the_methods():
    # The constructor does not check its parameters.
    ensemble = EnsembleRegressor(
        estimators=[Ridge()], method="median", opt=False
    )

    with pytest.raises(ValueError, match="median") as excinfo:
        ensemble.fit(X, y)
    for method in METHODS:
        assert repr(method) in str(excinfo.value)


# --- Optional dependencies ---------------------------------------------------

#: Fit an ensemble without tuning, and list the tuning modules imported.
_IMPORT_CHECK = textwrap.dedent(
    """
    import sys

    if sys.argv[1] == "blocked":
        # Make optuna and optuna-integration impossible to import.
        sys.modules["optuna"] = None
        sys.modules["optuna_integration"] = None

    import numpy as np
    from sklearn.linear_model import Ridge

    from yikit.models import EnsembleRegressor

    X = np.arange(40.0).reshape(20, 2)
    y = X.sum(axis=1)
    for method in ("average", "stacking", "blending"):
        EnsembleRegressor([Ridge()], method=method, cv=3, opt=False).fit(
            X, y
        ).predict(X)

    names = ["yikit.models._search_cv"]
    if sys.argv[1] == "blocked":
        names += ["optuna", "optuna_integration", "yikit.models._optuna"]
    print([name for name in names if sys.modules.get(name) is not None])
    """
)


@pytest.mark.parametrize("optuna", ["blocked", "installed"])
def test_tuning_modules_are_imported_only_when_tuning(optuna):
    result = subprocess.run(
        [sys.executable, "-c", _IMPORT_CHECK, optuna],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


@pytest.mark.parametrize(
    "blocked, reimported",
    [
        (
            ("optuna_integration", "optuna.integration"),
            ("yikit.models._search_cv",),
        ),
        (("yikit.models._optuna",), ()),
        (("optuna",), ("yikit.models._optuna", "yikit.models._search_cv")),
    ],
    ids=["optuna-integration", "yikit.models._optuna", "optuna"],
)
def test_tuning_without_its_modules_raises_import_error(
    monkeypatch, blocked, reimported
):
    _tuning_classes()
    _block_imports(monkeypatch, blocked, reimported)
    ensemble = EnsembleRegressor(
        estimators=[Ridge()], method="average", n_trials=N_TRIALS
    )

    with pytest.raises(ImportError, match="opt=False") as excinfo:
        ensemble.fit(X, y)

    message = str(excinfo.value)
    assert "optuna" in message
    assert "optuna-integration" in message
    # Chained from the error of the import that failed.
    cause = excinfo.value.__cause__
    assert isinstance(cause, ImportError)
    assert cause.name in blocked
    assert not hasattr(ensemble, "estimator_")
    # Without the tuning, the ensemble does not need these modules.
    for method in METHODS:
        fitted = EnsembleRegressor(
            estimators=[Ridge()], method=method, cv=CV, opt=False
        ).fit(X, y)
        assert fitted.predict(X).shape == (N_SAMPLES,)
