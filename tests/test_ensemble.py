"""Tests of ``yikit.models.EnsembleRegressor``."""

from __future__ import annotations

import inspect
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest
import sklearn
from joblib import parallel_backend
from numpy.testing import assert_allclose
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.datasets import make_regression
from sklearn.ensemble import (
    RandomForestRegressor,
    StackingRegressor,
    VotingRegressor,
)
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from yikit.models import EnsembleRegressor

SEED = 334
N_SAMPLES = 60
N_FEATURES = 4
CV = 3
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
