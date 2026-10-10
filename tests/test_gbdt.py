from __future__ import annotations

import inspect
import warnings
from itertools import product

import numpy as np
import optuna
import pytest
from sklearn.base import clone
from sklearn.datasets import make_regression
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils import check_random_state

lightgbm = pytest.importorskip("lightgbm")

from yikit.models import GBDTRegressor, Objective  # noqa: E402
from yikit.models._search_space import get_search_space  # noqa: E402

SEED = 334
N_SAMPLES = 120
N_FEATURES = 5


@pytest.fixture(scope="module")
def data() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_regression(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        noise=10.0,
        random_state=SEED,
    )
    return X, y


def test_fit_and_predict(data):
    X, y = data
    model = GBDTRegressor(random_state=SEED).fit(X, y)

    assert model.predict(X).shape == (N_SAMPLES,)
    assert model.n_features_in_ == N_FEATURES
    assert model.feature_importances_.shape == (N_FEATURES,)


def test_learns_only_on_the_training_split(data):
    """The validation split decides early stopping and is not trained on."""
    X, y = data
    model = GBDTRegressor(n_estimators=500, random_state=SEED).fit(X, y)

    rng = check_random_state(SEED)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=rng
    )
    expected = lightgbm.LGBMRegressor(
        n_estimators=500, random_state=rng, n_jobs=-1, verbosity=-1
    )
    if "eval_X" in inspect.signature(lightgbm.LGBMRegressor.fit).parameters:
        validation = {"eval_X": (X_valid,), "eval_y": (y_valid,)}
    else:
        validation = {"eval_set": [(X_valid, y_valid)]}
    expected.fit(
        X_train,
        y_train,
        eval_metric=["mse", "mae"],
        callbacks=[lightgbm.early_stopping(20, verbose=False)],
        **validation,
    )

    assert model.best_iteration_ == expected.booster_.best_iteration
    np.testing.assert_allclose(model.predict(X), expected.predict(X))


def test_stops_early(data):
    X, y = data
    model = GBDTRegressor(n_estimators=1000, random_state=SEED).fit(X, y)

    assert 0 < model.best_iteration_ < 1000
    assert model.best_iteration_ == model.estimator_.booster_.best_iteration


def test_same_random_state_gives_same_predictions(data):
    X, y = data
    predictions = [
        GBDTRegressor(subsample=0.7, subsample_freq=1, random_state=SEED)
        .fit(X, y)
        .predict(X)
        for _ in range(2)
    ]

    np.testing.assert_array_equal(predictions[0], predictions[1])


def test_unknown_keyword_arguments_are_rejected():
    with pytest.raises(TypeError):
        GBDTRegressor(verbosity=-1)  # type: ignore[call-arg]


def test_clone_keeps_all_parameters():
    model = GBDTRegressor(num_leaves=15, subsample=0.8, random_state=3)
    cloned = clone(model)

    assert cloned.get_params() == model.get_params()
    assert "kwargs" not in model.get_params()


def test_fit_emits_no_lightgbm_deprecation_warning(data):
    X, y = data
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        GBDTRegressor(random_state=SEED).fit(X, y)

    deprecations = [
        record
        for record in records
        if "lightgbm" in record.filename
        and (
            issubclass(record.category, (DeprecationWarning, FutureWarning))
            or "Deprecation" in record.category.__name__
            or "deprecated" in str(record.message)
        )
    ]
    assert deprecations == []


def _fit_parameters(fit):
    return inspect.signature(fit).parameters


def _fit_3_2(self, X, y, eval_set=None, verbose=True, callbacks=None): ...


def _fit_3_3(self, X, y, eval_set=None, verbose="warn", callbacks=None): ...


def _fit_4_6(self, X, y, eval_set=None, callbacks=None): ...


def _fit_4_7(
    self, X, y, eval_set=None, callbacks=None, eval_X=None, eval_y=None
): ...


@pytest.mark.parametrize(
    ("fit", "uses_eval_xy", "passes_verbose"),
    [
        pytest.param(_fit_3_2, False, True, id="lightgbm<3.3"),
        pytest.param(_fit_3_3, False, False, id="lightgbm3.3"),
        pytest.param(_fit_4_6, False, False, id="lightgbm4.0-4.6"),
        pytest.param(_fit_4_7, True, False, id="lightgbm>=4.7"),
    ],
)
def test_validation_fit_params_follow_the_fit_signature(
    monkeypatch, fit, uses_eval_xy, passes_verbose
):
    from yikit.models import _gbdt

    monkeypatch.setattr(_gbdt, "_FIT_PARAMETERS", _fit_parameters(fit))
    X_valid = np.zeros((4, 2))
    y_valid = np.zeros(4)
    params = _gbdt._validation_fit_params(X_valid, y_valid)

    if uses_eval_xy:
        assert "eval_set" not in params
        assert params["eval_X"][0] is X_valid
        assert params["eval_y"][0] is y_valid
    else:
        assert "eval_X" not in params
        assert params["eval_set"] == [(X_valid, y_valid)]
    if passes_verbose:
        assert params["verbose"] is False
    else:
        assert "verbose" not in params
    assert params["eval_metric"] == ["mse", "mae"]
    assert len(params["callbacks"]) == 1


def test_silent_model_prints_nothing(data, capfd):
    X, y = data
    GBDTRegressor(random_state=SEED).fit(X, y)

    captured = capfd.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_predict_before_fit_raises():
    from sklearn.exceptions import NotFittedError

    with pytest.raises(NotFittedError):
        GBDTRegressor().predict(np.zeros((2, N_FEATURES)))


def _bounds(distribution):
    if hasattr(distribution, "choices"):
        return list(distribution.choices)
    return [distribution.low, distribution.high]


def test_search_space_row_fits_at_its_bounds(data):
    """Every corner of the search space shared with LGBMRegressor fits."""
    X, y = data
    space = get_search_space(GBDTRegressor(), n_features=N_FEATURES)
    assert space is not None
    names = list(space)
    for values in product(*(_bounds(space[name]) for name in names)):
        params = dict(zip(names, values))
        model = GBDTRegressor(random_state=SEED, **params).fit(X, y)
        assert np.all(np.isfinite(model.predict(X))), params


def test_objective_searches_gbdt_regressor(data):
    X, y = data
    objective = Objective(
        GBDTRegressor(n_jobs=1, random_state=SEED),
        X,
        y,
        cv=KFold(n_splits=3, shuffle=True, random_state=SEED),
        random_state=SEED,
        scoring="neg_mean_squared_error",
    )
    study = optuna.create_study(
        direction="maximize", sampler=objective.sampler
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=3)

    assert all(
        trial.state == optuna.trial.TrialState.COMPLETE
        for trial in study.trials
    )
    best = objective.get_best_estimator(study)
    assert best.n_jobs == 1
    assert best.random_state == SEED
    assert best.fit(X, y).predict(X).shape == (N_SAMPLES,)
