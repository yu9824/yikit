from __future__ import annotations

import ast
import copy
import importlib
import inspect
import math
import pickle
import re
import textwrap
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import optuna
import pytest
from optuna.distributions import FloatDistribution, IntDistribution
from optuna.trial import TrialState
from sklearn.base import BaseEstimator, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import FitFailedWarning, NotFittedError
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

import yikit.models
from yikit.models import (
    EnsembleRegressor,
    Objective,
    ParamDistributions,
    RecommendedParams,
    _optuna,
    _search_space,
)

if TYPE_CHECKING:
    from collections.abc import Callable

try:
    from optuna_integration import OptunaSearchCV
except ImportError:  # old optuna that still bundles the integration
    from optuna.integration import OptunaSearchCV

SEED = 334

#: Small data for the tests that fit models or only inspect parameters.
N_SAMPLES = 40
N_FEATURES = 3

#: Expected values of ``test_optuna_search_cv``. The tolerance covers the
#: differences of the random forests between scikit-learn versions.
OPTUNA_SEARCH_CV_BEST_SCORE = -65.64
OPTUNA_SEARCH_CV_ABS_TOL = 0.2
OPTUNA_SEARCH_CV_BEST_TRIAL_NUMBER = 3

#: Expected values of ``test_create_study``. The tolerance covers the
#: differences of the random forests between scikit-learn versions (about
#: 0.15 between 0.24.1 and 1.9.1).
CREATE_STUDY_BEST_SCORE = -65.64
CREATE_STUDY_ABS_TOL = 0.2
CREATE_STUDY_BEST_TRIAL_NUMBER = 3

#: Number of trials of ``test_failed_trials_are_recorded_and_skipped``, all
#: drawn at random by the TPESampler (its first 10 trials). With the sampler
#: seeded from ``SEED``, trials 4 and 6 draw ``n_components=3``, which
#: cannot be fitted on the 2 selected features.
FAILING_SEARCH_N_TRIALS = 10


@pytest.fixture(scope="module")
def small_data() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_regression(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        noise=1.0,
        random_state=SEED,
    )
    return X, y


def _patch_cross_validate(
    monkeypatch: pytest.MonkeyPatch,
    score: Callable[[Any], float] = lambda estimator: 0.0,
) -> list[Any]:
    """Replace ``cross_validate`` of ``Objective`` with a fake one.

    The fake one fits nothing: it records the estimator it receives and
    returns ``score(estimator)`` as the only test score.

    Returns
    -------
    list
        The estimators passed to ``cross_validate``, in the order of the
        calls.
    """
    evaluated: list[Any] = []

    def fake_cross_validate(
        estimator: Any, X: Any, y: Any, **kwargs: Any
    ) -> dict[str, np.ndarray]:
        evaluated.append(estimator)
        return {"test_score": np.array([score(estimator)])}

    monkeypatch.setattr(_optuna, "cross_validate", fake_cross_validate)
    return evaluated


def _optimize(
    objective: Objective, n_trials: int, direction: str = "maximize"
) -> optuna.study.Study:
    study = optuna.create_study(direction=direction, sampler=objective.sampler)
    study.optimize(objective, n_trials=n_trials)
    return study


def test_create_study(X_regression, y_regression):
    x = X_regression
    y = y_regression

    estimator = RandomForestRegressor(random_state=334, n_jobs=-1)
    objective = Objective(
        estimator,
        x,
        y,
        scoring="neg_mean_absolute_error",
        cv=KFold(n_splits=5, shuffle=True, random_state=334),
        random_state=334,
    )
    study = optuna.create_study(
        direction="maximize", sampler=objective.sampler
    )
    study.optimize(objective, n_trials=10)

    assert math.isclose(
        study.best_value, CREATE_STUDY_BEST_SCORE, abs_tol=CREATE_STUDY_ABS_TOL
    )
    assert study.best_trial.number == CREATE_STUDY_BEST_TRIAL_NUMBER
    # random_state and n_jobs are given, so only the searched values are
    # added, and the best estimator keeps the values given.
    assert objective.get_best_params(study) == study.best_params
    best_estimator = objective.get_best_estimator(study)
    assert best_estimator.random_state == 334
    assert best_estimator.n_jobs == -1


def test_optuna_search_cv(X_regression, y_regression):
    x = X_regression
    y = y_regression

    estimator = RandomForestRegressor(random_state=334, n_jobs=-1)
    param_distributions = ParamDistributions(estimator)

    ocv = OptunaSearchCV(
        estimator,
        param_distributions=param_distributions,
        scoring="neg_mean_absolute_error",
        cv=KFold(n_splits=5, shuffle=True, random_state=334),
        random_state=334,
    )
    ocv.fit(x, y)
    assert math.isclose(
        ocv.best_score_,
        OPTUNA_SEARCH_CV_BEST_SCORE,
        abs_tol=OPTUNA_SEARCH_CV_ABS_TOL,
    )
    assert ocv.study_.best_trial.number == OPTUNA_SEARCH_CV_BEST_TRIAL_NUMBER
    # OptunaSearchCV keeps the parameters that are not searched.
    assert ocv.best_estimator_.random_state == 334
    assert ocv.best_estimator_.n_jobs == -1


def test_model_random_state_is_drawn_after_the_sampler_seed(small_data):
    X, y = small_data
    rng = np.random.RandomState(SEED)
    rng.randint(2**31 - 1)  # seed of the TPESampler
    expected = int(rng.randint(2**31 - 1))

    objective = Objective(RandomForestRegressor(), X, y, random_state=SEED)

    assert type(objective.model_random_state) is int
    assert objective.model_random_state == expected


def test_public_attributes(small_data):
    X, y = small_data
    estimator = RandomForestRegressor()
    fixed_params = {"max_depth": 3}

    objective = Objective(estimator, X, y, fixed_params=fixed_params)

    assert objective.estimator is estimator
    assert objective.fixed_params == fixed_params
    assert isinstance(objective.fixed_params, dict)
    assert objective.custom_params is None
    assert set(objective.param_distributions) == {
        "min_samples_split",
        "n_estimators",
    }
    assert isinstance(objective.sampler, optuna.samplers.TPESampler)
    for removed in ("model", "fixed_params_", "rng"):
        assert not hasattr(objective, removed)


def test_same_random_state_gives_same_results(small_data):
    X, y = small_data

    def search(random_state: int) -> tuple[Objective, optuna.study.Study]:
        objective = Objective(
            RandomForestRegressor(),
            X,
            y,
            cv=KFold(n_splits=3, shuffle=True, random_state=SEED),
            random_state=random_state,
            scoring="neg_mean_absolute_error",
        )
        return objective, _optimize(objective, n_trials=3)

    objective1, study1 = search(SEED)
    objective2, study2 = search(SEED)

    assert objective1.model_random_state == objective2.model_random_state
    # The integer of the random_state rule is the one used for fitting.
    assert objective1.estimator_.random_state == objective1.model_random_state
    assert study1.best_params == study2.best_params
    assert study1.best_value == study2.best_value
    assert objective1.get_best_params(study1) == objective2.get_best_params(
        study2
    )
    other = Objective(RandomForestRegressor(), X, y, random_state=SEED + 1)
    assert other.model_random_state != objective1.model_random_state


def test_fixed_params_are_not_searched_and_are_used(monkeypatch, small_data):
    X, y = small_data
    evaluated = _patch_cross_validate(monkeypatch)
    fixed_params = {"n_estimators": 20, "random_state": 5}

    objective = Objective(
        RandomForestRegressor(),
        X,
        y,
        fixed_params=fixed_params,
        random_state=SEED,
    )
    study = _optimize(objective, n_trials=3)

    assert set(objective.param_distributions) == {
        "min_samples_split",
        "max_depth",
    }
    assert all(
        set(trial.params) == {"min_samples_split", "max_depth"}
        for trial in study.trials
    )
    assert len(evaluated) == 3
    assert objective.estimator_ is evaluated[-1]
    for estimator in evaluated:
        # fixed_params win over the random_state rule.
        assert estimator.n_estimators == 20
        assert estimator.random_state == 5
    assert objective.get_best_params(study) == {
        **study.best_params,
        **fixed_params,
    }


def test_custom_params_override_registered_space(monkeypatch, small_data):
    X, y = small_data
    evaluated = _patch_cross_validate(monkeypatch)

    def custom_params(trial: optuna.trial.BaseTrial) -> dict[str, Any]:
        # The returned names may differ from the names given to suggest.
        return {
            "max_features": trial.suggest_float("max_features", 0.1, 1.0),
            "max_depth": trial.suggest_int("depth", 2, 8),
        }

    objective = Objective(
        RandomForestRegressor(),
        X,
        y,
        custom_params=custom_params,
        random_state=SEED,
    )
    study = _optimize(objective, n_trials=3)

    assert all(
        set(trial.params) == {"max_features", "depth"}
        for trial in study.trials
    )
    for trial, estimator in zip(study.trials, evaluated):
        assert estimator.max_features == trial.params["max_features"]
        assert estimator.max_depth == trial.params["depth"]
        # The registered space of RandomForestRegressor is not used.
        assert estimator.n_estimators == RandomForestRegressor().n_estimators
        assert estimator.random_state == objective.model_random_state
    assert objective.get_best_params(study) == {
        "random_state": objective.model_random_state,
        "max_features": study.best_params["max_features"],
        "max_depth": study.best_params["depth"],
    }


def test_custom_params_names_are_not_prefixed(monkeypatch, small_data):
    X, y = small_data
    evaluated = _patch_cross_validate(monkeypatch)

    objective = Objective(
        make_pipeline(StandardScaler(), SVR()),
        X,
        y,
        custom_params=lambda trial: {
            "svr__C": trial.suggest_float("C", 0.5, 2.0)
        },
    )
    study = _optimize(objective, n_trials=2)

    for trial, estimator in zip(study.trials, evaluated):
        assert estimator.named_steps["svr"].C == trial.params["C"]
    assert objective.get_best_params(study) == {
        "svr__C": study.best_params["C"]
    }


def test_old_default_arguments_mean_nothing(monkeypatch, small_data):
    X, y = small_data
    evaluated = _patch_cross_validate(monkeypatch)

    objective = Objective(
        SVR(), X, y, custom_params=lambda trial: {}, fixed_params={}
    )
    study = _optimize(objective, n_trials=2)

    assert objective.fixed_params == {}
    assert set(objective.param_distributions) == {"C", "epsilon"}
    assert all(set(trial.params) == {"C", "epsilon"} for trial in study.trials)
    for estimator in evaluated:
        # No implicit fixed value such as gamma="auto" is set.
        assert estimator.gamma == SVR().gamma
        assert estimator.kernel == SVR().kernel


@pytest.mark.parametrize(
    "estimator",
    [
        KNeighborsRegressor(),
        make_pipeline(StandardScaler(), KNeighborsRegressor()),
    ],
)
def test_unregistered_model_raises_at_construction(estimator, small_data):
    X, y = small_data
    with pytest.raises(
        NotImplementedError, match=r"KNeighborsRegressor.*custom_params"
    ):
        Objective(estimator, X, y)


def test_unregistered_model_with_custom_params(monkeypatch, small_data):
    X, y = small_data
    evaluated = _patch_cross_validate(monkeypatch)

    objective = Objective(
        KNeighborsRegressor(),
        X,
        y,
        custom_params=lambda trial: {
            "n_neighbors": trial.suggest_int("n_neighbors", 1, 5)
        },
    )
    study = _optimize(objective, n_trials=2)

    assert objective.param_distributions is None
    for trial, estimator in zip(study.trials, evaluated):
        assert estimator.n_neighbors == trial.params["n_neighbors"]


def test_empty_custom_params_without_search_space_raises(
    monkeypatch, small_data
):
    X, y = small_data
    evaluated = _patch_cross_validate(monkeypatch)
    objective = Objective(
        KNeighborsRegressor(), X, y, custom_params=lambda trial: {}
    )

    with pytest.raises(
        NotImplementedError, match=r"KNeighborsRegressor.*custom_params"
    ):
        objective(optuna.trial.FixedTrial({}))
    assert evaluated == []


def test_invalid_fixed_params_name_raises_at_construction(small_data):
    X, y = small_data
    with pytest.raises(ValueError, match="no_such_param"):
        Objective(SVR(), X, y, fixed_params={"no_such_param": 1})


def test_failed_trials_are_recorded_and_skipped(small_data):
    X, y = small_data
    # PLSRegression after 2 selected features cannot have 3 components,
    # while its upper bound comes from the 3 columns of X.
    estimator = Pipeline(
        [
            ("select", SelectKBest(f_regression, k=2)),
            ("pls", PLSRegression()),
        ]
    )
    objective = Objective(estimator, X, y, cv=3, random_state=SEED)
    assert objective.param_distributions == {
        "pls__n_components": IntDistribution(1, N_FEATURES)
    }

    with pytest.warns(FitFailedWarning) as records:
        study = _optimize(objective, n_trials=FAILING_SEARCH_N_TRIALS)

    assert len(study.trials) == FAILING_SEARCH_N_TRIALS
    failed = [t for t in study.trials if t.state == TrialState.FAIL]
    assert failed
    assert all(t.params["pls__n_components"] == 3 for t in failed)
    messages = [
        str(record.message)
        for record in records
        if issubclass(record.category, FitFailedWarning)
    ]
    assert any(
        f"Trial {failed[0].number}" in message and "n_components" in message
        for message in messages
    )
    assert study.best_trial.state == TrialState.COMPLETE
    assert study.best_params["pls__n_components"] != 3


def test_get_best_params_and_estimator(monkeypatch, small_data):
    X, y = small_data
    # The best trial is the one whose alpha is the closest to 1.
    _patch_cross_validate(
        monkeypatch, score=lambda estimator: -abs(math.log(estimator.alpha))
    )
    estimator = Ridge(max_iter=500)

    objective = Objective(
        estimator,
        X,
        y,
        fixed_params={"fit_intercept": False},
        random_state=SEED,
    )
    study = _optimize(objective, n_trials=5)

    best_params = objective.get_best_params(study)
    assert best_params == {
        "random_state": objective.model_random_state,
        "alpha": study.best_params["alpha"],
        "fit_intercept": False,
    }
    best_estimator = objective.get_best_estimator(study)
    assert isinstance(best_estimator, Ridge)
    assert best_estimator is not estimator
    with pytest.raises(NotFittedError):
        check_is_fitted(best_estimator)
    assert best_estimator.get_params() == {
        **estimator.get_params(),
        **best_params,
    }
    # The estimator passed to Objective is not modified.
    assert estimator.get_params() == Ridge(max_iter=500).get_params()


def test_get_best_estimator_of_nested_estimator(monkeypatch, small_data):
    X, y = small_data
    _patch_cross_validate(monkeypatch)
    estimator = TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), SVR())
    )

    objective = Objective(estimator, X, y, random_state=SEED)
    study = _optimize(objective, n_trials=2)

    assert set(objective.param_distributions) == {
        "regressor__svr__C",
        "regressor__svr__epsilon",
    }
    assert objective.get_best_params(study) == study.best_params
    best_svr = objective.get_best_estimator(study).regressor.named_steps["svr"]
    assert best_svr.C == study.best_params["regressor__svr__C"]
    assert best_svr.epsilon == study.best_params["regressor__svr__epsilon"]
    assert best_svr.gamma == SVR().gamma
    assert estimator.regressor.named_steps["svr"].C == SVR().C


# --- The case of licond: the parameters of the given model are kept ------


def _make_lightgbm(**params: Any) -> Any:
    lightgbm = pytest.importorskip("lightgbm")
    return lightgbm.LGBMRegressor(**params)


def _make_ngboost(**params: Any) -> Any:
    """Return a small ``NGBRegressor`` with an explicit ``Base``.

    ``Base`` is always given: the default ``Base`` of ngboost is one
    module-global object shared by every ``NGBRegressor``.
    """
    ngboost = pytest.importorskip("ngboost")
    params.setdefault("Base", DecisionTreeRegressor(max_depth=3))
    return ngboost.NGBRegressor(n_estimators=10, verbose=False, **params)


#: Factories of the models whose ``n_jobs`` and ``random_state`` licond
#: chooses itself.
LICOND_MODELS = [
    pytest.param(RandomForestRegressor, id="RandomForestRegressor"),
    pytest.param(_make_lightgbm, id="LGBMRegressor"),
]


def _assert_same_state(
    random_state: np.random.RandomState, expected: np.random.RandomState
) -> None:
    state, expected_state = random_state.get_state(), expected.get_state()
    assert state[0] == expected_state[0]
    np.testing.assert_array_equal(state[1], expected_state[1])
    assert state[2:] == expected_state[2:]


@pytest.mark.parametrize("make_estimator", LICOND_MODELS)
def test_n_jobs_is_kept(make_estimator, monkeypatch, small_data):
    X, y = small_data
    evaluated = _patch_cross_validate(monkeypatch)
    estimator = make_estimator(n_jobs=2)
    original_params = estimator.get_params()

    objective = Objective(estimator, X, y, random_state=SEED)
    study = _optimize(objective, n_trials=3)

    assert len(evaluated) == 3
    assert objective.estimator_ is evaluated[-1]
    assert all(model.n_jobs == 2 for model in evaluated)
    best_params = objective.get_best_params(study)
    assert "n_jobs" not in best_params
    assert best_params == {
        **study.best_params,
        "random_state": objective.model_random_state,
    }
    best_estimator = objective.get_best_estimator(study)
    assert best_estimator.n_jobs == 2
    # The parameters that are not searched keep the values of the input.
    assert best_estimator.get_params() == {**original_params, **best_params}
    assert estimator.get_params() == original_params


@pytest.mark.parametrize("make_estimator", LICOND_MODELS)
def test_given_random_state_is_kept(make_estimator, monkeypatch, small_data):
    X, y = small_data
    evaluated = _patch_cross_validate(monkeypatch)

    objective = Objective(
        make_estimator(random_state=7), X, y, random_state=SEED
    )
    study = _optimize(objective, n_trials=3)

    assert all(model.random_state == 7 for model in evaluated)
    # The rule adds nothing: only the searched values are returned.
    assert objective.get_best_params(study) == study.best_params
    assert objective.get_best_estimator(study).random_state == 7


@pytest.mark.parametrize("make_estimator", LICOND_MODELS)
def test_unspecified_random_state_gets_the_drawn_integer(
    make_estimator, monkeypatch, small_data
):
    X, y = small_data
    evaluated = _patch_cross_validate(monkeypatch)
    estimator = make_estimator()
    assert estimator.random_state is None

    objective = Objective(estimator, X, y, random_state=SEED)
    study = _optimize(objective, n_trials=3)

    expected = objective.model_random_state
    assert type(expected) is int
    assert all(model.random_state == expected for model in evaluated)
    assert objective.estimator_.random_state == expected
    assert objective.get_best_params(study)["random_state"] == expected
    assert objective.get_best_estimator(study).random_state == expected
    assert estimator.random_state is None


@pytest.mark.parametrize(
    "base_random_state", ["unspecified", "int", "RandomState"]
)
def test_ngboost_random_states(base_random_state, monkeypatch, small_data):
    X, y = small_data
    evaluated = _patch_cross_validate(monkeypatch)
    given = {
        "unspecified": None,
        "int": 11,
        "RandomState": np.random.RandomState(1),
    }[base_random_state]
    base = DecisionTreeRegressor(max_depth=3, random_state=given)
    base_params = base.get_params()
    estimator = _make_ngboost(Base=base)
    # ngboost turns random_state=None into the global RandomState.
    assert estimator.random_state is check_random_state(None)

    objective = Objective(estimator, X, y, random_state=SEED)
    study = _optimize(objective, n_trials=3)

    model_random_state = objective.model_random_state
    expected_params = {**study.best_params, "random_state": model_random_state}
    if given is None:
        expected_params["Base__random_state"] = model_random_state
    assert objective.get_best_params(study) == expected_params
    best_estimator = objective.get_best_estimator(study)
    models = [*evaluated, best_estimator]
    for model in models:
        # The constructor of ngboost turns the integer into a RandomState.
        _assert_same_state(
            model.random_state, np.random.RandomState(model_random_state)
        )
        if given is None:
            assert model.Base.random_state == model_random_state
        elif isinstance(given, int):
            assert model.Base.random_state == given
        else:
            _assert_same_state(model.Base.random_state, given)
    assert (
        best_estimator.Base.max_depth == study.best_params["Base__max_depth"]
    )

    # No RandomState or Base is shared between the models, nor with the
    # input estimator or the global RandomState.
    held = [
        value
        for model in models
        for value in (model.random_state, model.Base.random_state)
        if isinstance(value, np.random.RandomState)
    ]
    held_ids = [id(value) for value in held]
    assert len(set(held_ids)) == len(held_ids)
    assert not set(held_ids) & {
        id(estimator.random_state),
        id(base.random_state),
        id(check_random_state(None)),
    }
    base_ids = [id(model.Base) for model in models]
    assert len(set(base_ids)) == len(base_ids)
    assert id(base) not in base_ids
    # The input estimator is not modified.
    assert estimator.random_state is check_random_state(None)
    assert estimator.Base is base
    assert base.get_params() == base_params


def test_ngboost_trials_and_best_estimator_can_be_fitted(small_data):
    X, y = small_data
    # minibatch_frac < 1 draws from the RandomState of the model, which
    # fails if an integer random_state was not turned into a RandomState.
    objective = Objective(
        _make_ngboost(),
        X,
        y,
        fixed_params={"n_estimators": 10, "minibatch_frac": 0.5},
        cv=3,
        random_state=SEED,
    )
    study = _optimize(objective, n_trials=2)

    assert all(trial.state == TrialState.COMPLETE for trial in study.trials)
    best_estimator = objective.get_best_estimator(study).fit(X, y)
    assert best_estimator.predict(X).shape == (N_SAMPLES,)


@pytest.mark.parametrize("make_estimator", LICOND_MODELS)
def test_fixed_n_jobs_and_random_state_are_used(
    make_estimator, monkeypatch, small_data
):
    X, y = small_data
    evaluated = _patch_cross_validate(monkeypatch)
    fixed_params = {"n_jobs": 3, "random_state": 5}
    estimator = make_estimator(n_jobs=2)

    objective = Objective(
        estimator, X, y, fixed_params=fixed_params, random_state=SEED
    )
    study = _optimize(objective, n_trials=3)

    assert all(
        not set(fixed_params) & set(trial.params) for trial in study.trials
    )
    for model in evaluated:
        assert model.n_jobs == 3
        assert model.random_state == 5
    assert objective.get_best_params(study) == {
        **study.best_params,
        **fixed_params,
    }
    best_estimator = objective.get_best_estimator(study)
    assert best_estimator.n_jobs == 3
    assert best_estimator.random_state == 5
    assert estimator.n_jobs == 2
    assert estimator.random_state is None


def test_svr_kernel_and_gamma_are_kept(monkeypatch, small_data):
    X, y = small_data
    evaluated = _patch_cross_validate(monkeypatch)
    estimator = SVR(kernel="linear")

    objective = Objective(estimator, X, y, random_state=SEED)
    study = _optimize(objective, n_trials=3)

    best_params = objective.get_best_params(study)
    # SVR has no random_state, so only the searched values are returned.
    assert best_params == study.best_params
    assert set(best_params) == {"C", "epsilon"}
    best_estimator = objective.get_best_estimator(study)
    for model in [*evaluated, best_estimator]:
        assert model.kernel == "linear"
        assert model.gamma == SVR().gamma
    assert best_estimator.get_params() == {
        **estimator.get_params(),
        **best_params,
    }


# --- fixed_params holding an estimator win over the names under it --------


@pytest.mark.parametrize(
    "base_random_state", [3, None], ids=["explicit", "unspecified"]
)
def test_fixed_ngboost_base_is_not_overwritten(
    base_random_state, monkeypatch, small_data
):
    X, y = small_data
    evaluated = _patch_cross_validate(monkeypatch)
    # The Base of the input has random_state=None.
    estimator = _make_ngboost()
    fixed_base = DecisionTreeRegressor(
        random_state=base_random_state, max_depth=5
    )
    fixed_base_params = fixed_base.get_params()

    objective = Objective(
        estimator,
        X,
        y,
        fixed_params={"Base": fixed_base},
        random_state=SEED,
    )
    study = _optimize(objective, n_trials=3)

    # Nothing under the fixed Base is searched.
    assert set(objective.param_distributions) == {
        "n_estimators",
        "minibatch_frac",
    }
    assert all(
        set(trial.params) == {"n_estimators", "minibatch_frac"}
        for trial in study.trials
    )
    model_random_state = objective.model_random_state
    expected_params = {
        **study.best_params,
        "random_state": model_random_state,
        "Base": fixed_base,
    }
    if base_random_state is None:
        # The rule applies to the fixed Base itself.
        expected_params["Base__random_state"] = model_random_state
    assert objective.get_best_params(study) == expected_params
    expected_base_params = {
        **fixed_base_params,
        "random_state": (
            model_random_state
            if base_random_state is None
            else base_random_state
        ),
    }
    best_estimator = objective.get_best_estimator(study)
    for model in [*evaluated, best_estimator]:
        assert model.Base is not fixed_base
        assert model.Base.get_params() == expected_base_params
    assert fixed_base.get_params() == fixed_base_params


@pytest.mark.parametrize(
    "step_random_state", [3, None], ids=["explicit", "unspecified"]
)
def test_fixed_pipeline_step_is_not_overwritten(
    step_random_state, monkeypatch, small_data
):
    X, y = small_data
    evaluated = _patch_cross_validate(monkeypatch)
    estimator = make_pipeline(
        QuantileTransformer(n_quantiles=10), RandomForestRegressor()
    )
    fixed_step = RandomForestRegressor(
        random_state=step_random_state, n_estimators=5
    )
    fixed_step_params = fixed_step.get_params()

    objective = Objective(
        estimator,
        X,
        y,
        fixed_params={"randomforestregressor": fixed_step},
        random_state=SEED,
    )
    study = _optimize(objective, n_trials=2)

    # Every searched name is under the fixed step.
    assert objective.param_distributions == {}
    assert all(trial.params == {} for trial in study.trials)
    model_random_state = objective.model_random_state
    expected_params = {
        # The steps that are not fixed still follow the rule.
        "quantiletransformer__random_state": model_random_state,
        "randomforestregressor": fixed_step,
    }
    if step_random_state is None:
        expected_params["randomforestregressor__random_state"] = (
            model_random_state
        )
    assert objective.get_best_params(study) == expected_params
    expected_step_params = {
        **fixed_step_params,
        "random_state": (
            model_random_state
            if step_random_state is None
            else step_random_state
        ),
    }
    best_estimator = objective.get_best_estimator(study)
    for model in [*evaluated, best_estimator]:
        step = model.named_steps["randomforestregressor"]
        assert step is not fixed_step
        assert step.get_params() == expected_step_params
        assert (
            model.named_steps["quantiletransformer"].random_state
            == model_random_state
        )
    assert fixed_step.get_params() == fixed_step_params


def test_fixed_estimator_inside_a_fixed_estimator_wins(
    monkeypatch, small_data
):
    X, y = small_data
    evaluated = _patch_cross_validate(monkeypatch)
    estimator = TransformedTargetRegressor(regressor=RandomForestRegressor())
    # The fixed regressor__randomforestregressor replaces the step of the
    # fixed regressor, so the rule does not set the random_state of the
    # replaced step.
    fixed_params = {
        "regressor": make_pipeline(
            QuantileTransformer(n_quantiles=10), RandomForestRegressor()
        ),
        "regressor__randomforestregressor": RandomForestRegressor(
            random_state=3
        ),
    }

    objective = Objective(
        estimator, X, y, fixed_params=fixed_params, random_state=SEED
    )
    study = _optimize(objective, n_trials=2)

    assert objective.param_distributions == {}
    model_random_state = objective.model_random_state
    assert objective.get_best_params(study) == {
        "regressor__quantiletransformer__random_state": model_random_state,
        **fixed_params,
    }
    for model in [*evaluated, objective.get_best_estimator(study)]:
        steps = model.regressor.named_steps
        assert steps["quantiletransformer"].random_state == model_random_state
        assert steps["randomforestregressor"].random_state == 3


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("regressor", True),
        ("regressor__C", True),
        ("regressor__svr__C", True),
        ("regressor2__C", False),
        ("regressor2", False),
        ("reg__C", False),
    ],
)
def test_fixed_name_covers_only_its_own_path(name, expected):
    assert _optuna._is_covered(name, ["regressor"]) is expected


# --- ParamDistributions ---------------------------------------------------


def _make_gbdt(**params: Any) -> Any:
    pytest.importorskip("lightgbm")
    from yikit.models._gbdt import GBDTRegressor

    return GBDTRegressor(**params)


#: Factories of one model of each type of the search space table. The id of
#: each factory is the name of the type.
REGISTERED_MODELS = [
    pytest.param(SVR, id="SVR"),
    pytest.param(LinearSVR, id="LinearSVR"),
    pytest.param(RandomForestRegressor, id="RandomForestRegressor"),
    pytest.param(MLPRegressor, id="MLPRegressor"),
    pytest.param(PLSRegression, id="PLSRegression"),
    pytest.param(Ridge, id="Ridge"),
    pytest.param(Lasso, id="Lasso"),
    pytest.param(ElasticNet, id="ElasticNet"),
    pytest.param(_make_lightgbm, id="LGBMRegressor"),
    pytest.param(_make_gbdt, id="GBDTRegressor"),
    pytest.param(_make_ngboost, id="NGBRegressor"),
]

#: Factories of registered models nested in meta-estimators.
NESTED_MODELS = [
    pytest.param(
        lambda: TransformedTargetRegressor(
            regressor=make_pipeline(StandardScaler(), SVR())
        ),
        id="TransformedTargetRegressor-Pipeline-SVR",
    ),
    pytest.param(
        lambda: make_pipeline(StandardScaler(), PLSRegression()),
        id="Pipeline-PLSRegression",
    ),
    pytest.param(
        lambda: TransformedTargetRegressor(regressor=_make_ngboost()),
        id="TransformedTargetRegressor-NGBRegressor",
    ),
]

#: Start of the warning about the names that ``OptunaSearchCV`` cannot set.
SET_PARAMS_WARNING = "OptunaSearchCV cannot apply"


def test_parity_covers_every_registered_type():
    registered = {
        model_type.__name__
        for model_types, _, _ in _search_space._SEARCH_SPACES
        for model_type in (
            model_types if isinstance(model_types, tuple) else (model_types,)
        )
    }

    assert registered <= {param.id for param in REGISTERED_MODELS}


@pytest.mark.filterwarnings(f"ignore:{SET_PARAMS_WARNING}:UserWarning")
@pytest.mark.parametrize(
    "make_estimator", [*REGISTERED_MODELS, *NESTED_MODELS]
)
def test_param_distributions_match_objective(
    make_estimator, monkeypatch, small_data
):
    X, y = small_data
    _patch_cross_validate(monkeypatch)
    estimator = make_estimator()

    objective = Objective(estimator, X, y, random_state=SEED)
    study = _optimize(objective, n_trials=1)
    param_distributions = ParamDistributions(estimator, n_features=X.shape[1])

    assert isinstance(param_distributions, dict)
    distributions = study.trials[0].distributions
    assert distributions == dict(param_distributions)
    assert list(distributions) == list(param_distributions)


def test_param_distributions_attributes():
    estimator = PLSRegression()

    def custom_params(trial: optuna.trial.BaseTrial) -> dict[str, Any]:
        return {}

    param_distributions = ParamDistributions(
        estimator, custom_params, n_features=N_FEATURES
    )

    assert param_distributions == {
        "n_components": IntDistribution(1, N_FEATURES)
    }
    assert param_distributions.estimator is estimator
    assert param_distributions.custom_params is custom_params
    assert param_distributions.n_features == N_FEATURES
    for removed in ("rng", "random_state", "fixed_params", "distributions"):
        assert not hasattr(param_distributions, removed)
    text = repr(param_distributions)
    assert text.startswith("ParamDistributions(PLSRegression()")
    assert f"custom_params={custom_params!r}" in text
    assert f"n_features={N_FEATURES}" in text


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"fixed_params": {}}, id="fixed_params"),
        pytest.param({"random_state": SEED}, id="random_state"),
    ],
)
def test_removed_arguments_raise_type_error(kwargs):
    (name,) = kwargs
    with pytest.raises(TypeError, match=name):
        ParamDistributions(SVR(), **kwargs)


def test_third_positional_argument_raises_type_error():
    # The third positional argument used to be fixed_params, and n_features
    # is keyword-only.
    with pytest.raises(TypeError, match="positional"):
        ParamDistributions(PLSRegression(), None, N_FEATURES)


@pytest.mark.parametrize(
    ("n_features", "expected_high"),
    [(None, 10), (1, 1), (N_FEATURES, N_FEATURES), (10, 10), (20, 10)],
)
def test_n_features_bounds_pls(n_features, expected_high):
    expected = {"n_components": IntDistribution(1, expected_high)}

    assert (
        ParamDistributions(PLSRegression(), n_features=n_features) == expected
    )
    assert ParamDistributions(
        make_pipeline(StandardScaler(), PLSRegression()),
        n_features=n_features,
    ) == {
        f"plsregression__{name}": distribution
        for name, distribution in expected.items()
    }


def test_n_features_defaults_to_the_bound_of_the_table():
    assert ParamDistributions(PLSRegression()) == {
        "n_components": IntDistribution(1, 10)
    }


#: Distributions given as ``custom_params`` of a RandomForestRegressor.
CUSTOM_DISTRIBUTIONS = {
    "max_features": FloatDistribution(0.1, 1.0),
    "max_depth": IntDistribution(2, 8),
}


@pytest.mark.parametrize(
    "as_function", [False, True], ids=["dict", "function"]
)
def test_custom_params_replace_the_table(as_function):
    calls: list[Any] = []

    def custom_params(trial: optuna.trial.BaseTrial) -> dict[str, Any]:
        calls.append(trial)
        return CUSTOM_DISTRIBUTIONS

    param_distributions = ParamDistributions(
        RandomForestRegressor(),
        custom_params if as_function else CUSTOM_DISTRIBUTIONS,
    )

    assert param_distributions == CUSTOM_DISTRIBUTIONS
    assert list(param_distributions) == list(CUSTOM_DISTRIBUTIONS)
    if as_function:
        assert len(calls) == 1
        assert isinstance(calls[0], optuna.trial.FixedTrial)


def test_custom_params_names_are_used_as_given():
    custom_params = {"svr__C": FloatDistribution(0.5, 2.0)}

    param_distributions = ParamDistributions(
        TransformedTargetRegressor(
            regressor=make_pipeline(StandardScaler(), SVR())
        ),
        custom_params,
    )

    assert param_distributions == custom_params


@pytest.mark.parametrize(
    "custom_params",
    [
        pytest.param(None, id="None"),
        pytest.param({}, id="empty-dict"),
        pytest.param(lambda trial: {}, id="old-default"),
    ],
)
def test_empty_custom_params_use_the_table(custom_params):
    estimator = make_pipeline(StandardScaler(), SVR())

    assert ParamDistributions(
        estimator, custom_params
    ) == _search_space.get_search_space(estimator)


@pytest.mark.parametrize(
    "custom_params",
    [
        pytest.param({"C": 1.0}, id="dict-of-values"),
        pytest.param(lambda trial: {"C": 1.0}, id="function-of-values"),
        pytest.param(
            lambda trial: [("C", FloatDistribution(0.1, 1.0))],
            id="function-returning-a-list",
        ),
        pytest.param([("C", FloatDistribution(0.1, 1.0))], id="list"),
    ],
)
def test_invalid_custom_params_raise_type_error(custom_params):
    with pytest.raises(TypeError, match="custom_params"):
        ParamDistributions(SVR(), custom_params)


def test_type_error_names_the_values_that_are_not_distributions():
    custom_params = {
        "C": FloatDistribution(0.1, 1.0),
        "epsilon": 0.1,
        "gamma": "auto",
    }

    with pytest.raises(TypeError) as error:
        ParamDistributions(SVR(), custom_params)

    message = str(error.value)
    assert "'epsilon' (float)" in message
    assert "'gamma' (str)" in message
    assert "'C'" not in message


class _CustomParamsError(Exception):
    """Error raised by a ``custom_params`` function of a test."""


def test_errors_of_custom_params_function_are_not_swallowed():
    def failing(trial: optuna.trial.BaseTrial) -> dict[str, Any]:
        raise _CustomParamsError("raised by custom_params")

    def suggesting(trial: optuna.trial.BaseTrial) -> dict[str, Any]:
        # A function for Objective suggests values, which FixedTrial({})
        # cannot do.
        return {"n_neighbors": trial.suggest_int("n_neighbors", 1, 5)}

    with pytest.raises(_CustomParamsError, match="raised by custom_params"):
        ParamDistributions(KNeighborsRegressor(), failing)
    with pytest.raises(ValueError, match="n_neighbors"):
        ParamDistributions(KNeighborsRegressor(), suggesting)


@pytest.mark.parametrize(
    "estimator",
    [
        KNeighborsRegressor(),
        make_pipeline(StandardScaler(), KNeighborsRegressor()),
    ],
)
@pytest.mark.parametrize(
    "custom_params",
    [
        pytest.param(None, id="None"),
        pytest.param({}, id="empty-dict"),
        pytest.param(lambda trial: {}, id="old-default"),
    ],
)
def test_param_distributions_of_unregistered_model_raise(
    estimator, custom_params
):
    with pytest.raises(
        NotImplementedError, match=r"KNeighborsRegressor.*custom_params"
    ):
        ParamDistributions(estimator, custom_params)


def test_param_distributions_of_unregistered_model_with_custom_params():
    custom_params = {"n_neighbors": IntDistribution(1, 5)}

    assert (
        ParamDistributions(KNeighborsRegressor(), custom_params)
        == custom_params
    )


def _ngboost_set_params_ignores_nested_names() -> bool:
    ngboost = pytest.importorskip("ngboost")
    return ngboost.NGBRegressor.set_params is not BaseEstimator.set_params


def _user_warnings(records: list[warnings.WarningMessage]) -> list[str]:
    return [
        str(record.message)
        for record in records
        if record.category is UserWarning
    ]


@pytest.mark.parametrize(
    ("wrap", "prefix"),
    [
        pytest.param(lambda model: model, "", id="NGBRegressor"),
        pytest.param(
            lambda model: TransformedTargetRegressor(
                regressor=make_pipeline(StandardScaler(), model)
            ),
            "regressor__ngbregressor__",
            id="nested",
        ),
    ],
)
def test_ngboost_nested_names_warn(wrap, prefix):
    ignores_nested_names = _ngboost_set_params_ignores_nested_names()
    estimator = wrap(_make_ngboost())

    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        param_distributions = ParamDistributions(estimator)

    nested_names = [f"{prefix}Base__max_depth", f"{prefix}Base__criterion"]
    assert set(nested_names) <= set(param_distributions)
    messages = _user_warnings(records)
    if not ignores_nested_names:
        # ngboost before 0.4.0 applies nested names in set_params.
        assert messages == []
        return
    assert len(messages) == 1
    (message,) = messages
    assert message.startswith(SET_PARAMS_WARNING)
    assert "NGBRegressor" in message
    assert "Objective" in message
    for name in nested_names:
        assert repr(name) in message
    # The names that set_params of NGBRegressor applies are not listed.
    assert repr(f"{prefix}n_estimators") not in message
    assert repr(f"{prefix}minibatch_frac") not in message


def test_ngboost_warning_follows_custom_params():
    if not _ngboost_set_params_ignores_nested_names():
        pytest.skip("set_params of this ngboost applies nested names")

    estimator = _make_ngboost()
    custom_params = {"n_estimators": IntDistribution(10, 100)}
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        ParamDistributions(estimator, custom_params)

    assert _user_warnings(records) == []
    with pytest.warns(UserWarning, match=r"'Base__max_depth'"):
        ParamDistributions(
            _make_ngboost(), {"Base__max_depth": IntDistribution(2, 10)}
        )


@pytest.mark.parametrize(
    "estimator",
    [
        SVR(),
        RandomForestRegressor(),
        TransformedTargetRegressor(
            regressor=make_pipeline(StandardScaler(), SVR())
        ),
    ],
)
def test_no_warning_for_models_that_apply_nested_names(estimator):
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        ParamDistributions(estimator)

    assert _user_warnings(records) == []


def test_param_distributions_survive_deepcopy_and_clone():
    estimator = make_pipeline(StandardScaler(), PLSRegression())
    param_distributions = ParamDistributions(estimator, n_features=N_FEATURES)

    copied = copy.deepcopy(param_distributions)

    assert type(copied) is ParamDistributions
    assert copied is not param_distributions
    assert copied == param_distributions
    assert copied.n_features == N_FEATURES
    assert copied.custom_params is None
    assert repr(copied) == repr(param_distributions)

    search = clone(
        OptunaSearchCV(estimator, param_distributions=param_distributions)
    )
    assert type(search.param_distributions) is ParamDistributions
    assert search.param_distributions == param_distributions


def test_removed_wrappers_are_not_imported_by_optuna_module():
    for name in ("LinearModelRegressor", "SupportVectorRegressor", "NoneType"):
        assert not hasattr(_optuna, name)


# --- RecommendedParams ------------------------------------------------------


class _SubSVR(SVR):
    """Subclass of SVR, which the recommended values table also matches."""


def _nested_svr() -> TransformedTargetRegressor:
    return TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), SVR()),
        transformer=StandardScaler(),
    )


@pytest.mark.parametrize(
    ("make_estimator", "expected"),
    [
        (SVR, {"gamma": "auto"}),
        (lambda: SVR(kernel="linear", gamma=0.1), {"gamma": "auto"}),
        (_SubSVR, {"gamma": "auto"}),
        (
            lambda: make_pipeline(StandardScaler(), SVR()),
            {"svr__gamma": "auto"},
        ),
        (
            lambda: TransformedTargetRegressor(regressor=SVR()),
            {"regressor__gamma": "auto"},
        ),
        (_nested_svr, {"regressor__svr__gamma": "auto"}),
    ],
    ids=["svr", "svr-given-gamma", "subclass", "pipeline", "ttr", "nested"],
)
def test_recommended_params_of_svr(make_estimator, expected):
    estimator = make_estimator()
    params_before = estimator.get_params()

    recommended = RecommendedParams(estimator)

    assert isinstance(recommended, dict)
    assert recommended == expected
    assert recommended.estimator is estimator
    # The values are only returned, never set on the estimator.
    assert estimator.get_params() == params_before


@pytest.mark.parametrize(
    "make_estimator",
    [
        RandomForestRegressor,
        KNeighborsRegressor,
        Ridge,
        LinearSVR,
        lambda: make_pipeline(StandardScaler(), RandomForestRegressor()),
        lambda: TransformedTargetRegressor(regressor=KNeighborsRegressor()),
        TransformedTargetRegressor,
    ],
    ids=[
        "random-forest",
        "kneighbors",
        "ridge",
        "linear-svr",
        "pipeline-random-forest",
        "ttr-kneighbors",
        "ttr-none",
    ],
)
def test_recommended_params_of_other_models_are_empty(make_estimator):
    estimator = make_estimator()

    recommended = RecommendedParams(estimator)

    assert type(recommended) is RecommendedParams
    assert recommended == {}
    assert recommended.estimator is estimator


def test_recommended_params_behave_as_a_plain_dict():
    recommended = RecommendedParams(SVR())

    assert recommended == {"gamma": "auto"}
    assert {"gamma": "auto"} == recommended
    assert recommended != {"gamma": "scale"}
    assert {**recommended} == {"gamma": "auto"}
    assert dict(recommended) == {"gamma": "auto"}
    assert repr(recommended) == "RecommendedParams({'gamma': 'auto'})"
    assert repr(RecommendedParams(Ridge())) == "RecommendedParams({})"


def test_recommended_params_are_new_each_time():
    recommended = RecommendedParams(SVR())
    recommended["gamma"] = 0.1
    recommended["C"] = 10.0

    assert RecommendedParams(SVR()) == {"gamma": "auto"}


@pytest.mark.parametrize(
    "copy_function",
    [copy.deepcopy, lambda obj: pickle.loads(pickle.dumps(obj))],
    ids=["deepcopy", "pickle"],
)
def test_recommended_params_survive_deepcopy_and_pickle(copy_function):
    recommended = RecommendedParams(_nested_svr())

    copied = copy_function(recommended)

    assert type(copied) is RecommendedParams
    assert copied is not recommended
    assert copied == recommended == {"regressor__svr__gamma": "auto"}
    assert type(copied.estimator) is TransformedTargetRegressor
    assert copied.estimator is not recommended.estimator
    assert repr(copied) == repr(recommended)


@pytest.mark.parametrize(
    "make_estimator", [SVR, _SubSVR, _nested_svr], ids=["svr", "sub", "nested"]
)
def test_recommended_params_can_be_passed_to_set_params(make_estimator):
    estimator = make_estimator()
    recommended = RecommendedParams(estimator)

    returned = estimator.set_params(**recommended)

    assert returned is estimator
    params = estimator.get_params()
    assert recommended
    for name, value in recommended.items():
        assert params[name] == value == "auto"


@pytest.mark.parametrize(
    ("make_estimator", "name"),
    [(SVR, "gamma"), (_nested_svr, "regressor__svr__gamma")],
    ids=["svr", "nested"],
)
def test_objective_uses_recommended_params_as_fixed_params(
    make_estimator, name, monkeypatch, small_data
):
    X, y = small_data
    evaluated = _patch_cross_validate(monkeypatch)
    estimator = make_estimator()

    objective = Objective(
        estimator,
        X,
        y,
        fixed_params=RecommendedParams(estimator),
        random_state=SEED,
    )
    study = _optimize(objective, n_trials=3)

    assert objective.fixed_params == {name: "auto"}
    assert objective.param_distributions is not None
    assert name not in objective.param_distributions
    assert all(name not in trial.params for trial in study.trials)
    assert objective.get_best_params(study)[name] == "auto"
    best_estimator = objective.get_best_estimator(study)
    assert len(evaluated) == 3
    for model in [*evaluated, best_estimator]:
        assert model.get_params()[name] == "auto"
    # The estimator passed by the user keeps its own gamma.
    assert estimator.get_params()[name] == "scale"


@pytest.mark.parametrize(
    ("make_estimator", "name", "gamma"),
    [
        (SVR, "gamma", "scale"),
        (lambda: SVR(gamma=0.1), "gamma", 0.1),
        (_nested_svr, "regressor__svr__gamma", "scale"),
    ],
    ids=["svr", "svr-given-gamma", "nested"],
)
def test_objective_keeps_gamma_without_recommended_params(
    make_estimator, name, gamma, monkeypatch, small_data
):
    X, y = small_data
    evaluated = _patch_cross_validate(monkeypatch)

    objective = Objective(make_estimator(), X, y, random_state=SEED)
    study = _optimize(objective, n_trials=3)

    assert objective.fixed_params == {}
    assert name not in objective.get_best_params(study)
    best_estimator = objective.get_best_estimator(study)
    assert len(evaluated) == 3
    for model in [*evaluated, best_estimator]:
        assert model.get_params()[name] == gamma


def test_objective_and_param_distributions_do_not_use_recommended_params(
    monkeypatch, small_data
):
    X, y = small_data
    _patch_cross_validate(monkeypatch)

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the recommended values must not be looked up")

    monkeypatch.setattr(_optuna, "RecommendedParams", fail)
    monkeypatch.setattr(_optuna, "get_recommended_params", fail)
    monkeypatch.setattr(_search_space, "get_recommended_params", fail)
    estimator = _nested_svr()

    objective = Objective(estimator, X, y, random_state=SEED)
    study = _optimize(objective, n_trials=2)
    objective.get_best_params(study)
    best_estimator = objective.get_best_estimator(study)
    param_distributions = ParamDistributions(estimator, n_features=N_FEATURES)

    assert best_estimator.regressor.named_steps["svr"].gamma == "scale"
    assert "regressor__svr__gamma" not in param_distributions


# --- Public names of yikit.models and EnsembleRegressor ----------------------


def test_linear_model_regressor_is_not_provided():
    with pytest.raises(
        ImportError, match="cannot import name 'LinearModelRegressor'"
    ):
        from yikit.models import LinearModelRegressor  # noqa: F401

    assert "LinearModelRegressor" not in yikit.models.__all__


def test_support_vector_regressor_is_not_provided():
    with pytest.raises(
        ImportError, match="cannot import name 'SupportVectorRegressor'"
    ):
        from yikit.models import SupportVectorRegressor  # noqa: F401

    assert "SupportVectorRegressor" not in yikit.models.__all__


@pytest.mark.parametrize(
    "module", ["yikit.models._linear", "yikit.models._svm"]
)
def test_removed_wrapper_modules_do_not_exist(module):
    with pytest.raises(ModuleNotFoundError, match=re.escape(module)):
        importlib.import_module(module)


def test_optuna_classes_are_public():
    from yikit.models import Objective as PublicObjective
    from yikit.models import ParamDistributions as PublicParamDistributions
    from yikit.models import RecommendedParams as PublicRecommendedParams

    assert PublicObjective is _optuna.Objective
    assert PublicParamDistributions is _optuna.ParamDistributions
    assert PublicRecommendedParams is _optuna.RecommendedParams
    for name in ("Objective", "ParamDistributions", "RecommendedParams"):
        assert name in yikit.models.__all__
    for name in yikit.models.__all__:
        assert hasattr(yikit.models, name)


def test_ensemble_builds_the_tuned_estimator_with_get_best_estimator():
    # EnsembleRegressor cannot be fitted on scikit-learn >= 1.4 (fixed by
    # ensemble-on-sklearn), so the source of ``fit`` is checked instead.
    tree = ast.parse(textwrap.dedent(inspect.getsource(EnsembleRegressor.fit)))

    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get_best_estimator"
    ]
    assert len(calls) == 1
    (call,) = calls
    assert isinstance(call.func, ast.Attribute)
    assert isinstance(call.func.value, ast.Name)
    assert call.func.value.id == "objective"
    assert len(call.args) == 1
    assert isinstance(call.args[0], ast.Name)
    assert call.args[0].id == "study"
    assert call.keywords == []

    # It is assigned to the name of the estimator fitted afterwards.
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and node.value is call
    ]
    assert len(assignments) == 1
    (target,) = assignments[0].targets
    assert isinstance(target, ast.Name)
    assert target.id == "_best_estimator_"

    used_attributes = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "objective"
    }
    assert used_attributes.isdisjoint({"model", "fixed_params_", "rng"})
