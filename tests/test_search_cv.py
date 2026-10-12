"""Tests of ``yikit.models._search_cv.OptunaSearchRegressor``."""

from __future__ import annotations

import inspect
import logging
import math
import pickle
import threading
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest
import sklearn
from numpy.testing import assert_allclose
from sklearn.base import BaseEstimator, clone, is_regressor
from sklearn.datasets import make_regression
from sklearn.ensemble import StackingRegressor, VotingRegressor
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GroupKFold

optuna = pytest.importorskip("optuna")

try:
    from optuna_integration import OptunaSearchCV
except ImportError:  # old optuna that still bundles the integration
    try:
        from optuna.integration import OptunaSearchCV
    except ImportError:
        pytest.skip(
            "OptunaSearchCV (optuna-integration) is not installed",
            allow_module_level=True,
        )

from yikit.models import ParamDistributions  # noqa: E402
from yikit.models._search_cv import OptunaSearchRegressor  # noqa: E402

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

SEED = 334
N_SAMPLES = 40
N_FEATURES = 3
N_TRIALS = 2
CV = 3

#: Seconds to wait for the other thread in the tests of concurrent fits.
TIMEOUT = 60.0

#: ``feature_names_in_`` exists from scikit-learn 1.0.
HAS_FEATURE_NAMES = tuple(
    int(part) for part in sklearn.__version__.split(".")[:2]
) >= (1, 0)

#: Small data for the fast searches.
X_SMALL, y_SMALL = make_regression(
    n_samples=N_SAMPLES,
    n_features=N_FEATURES,
    noise=1.0,
    random_state=SEED,
)
COLUMNS = [f"x{i}" for i in range(N_FEATURES)]

#: Build a scikit-learn ensemble from ``(name, estimator)`` pairs.
ENSEMBLE_FACTORIES = pytest.mark.parametrize(
    "make_ensemble",
    [
        lambda estimators: VotingRegressor(estimators),
        lambda estimators: StackingRegressor(estimators, cv=CV),
    ],
    ids=["voting", "stacking"],
)


class _PropertyPredictSearchCV(OptunaSearchCV):
    """``OptunaSearchCV`` whose ``predict`` is a property.

    This is how optuna 3.0 to 3.6 (``optuna.integration``) and
    optuna-integration 4.0.0 or older define ``predict``: reading it before
    ``fit`` raises ``NotFittedError``, an ``AttributeError``.
    """

    @property
    def predict(self) -> Any:
        self._check_is_fitted()
        return self.best_estimator_.predict


class _OldParentSearchRegressor(
    OptunaSearchRegressor, _PropertyPredictSearchCV
):
    """``OptunaSearchRegressor`` whose parent defines ``predict`` as above.

    In the method resolution order, ``super()`` in the methods of
    ``OptunaSearchRegressor`` reaches ``_PropertyPredictSearchCV``.
    """


def _make_search(
    estimator: Any = None,
    *,
    search_class: type[Any] = OptunaSearchRegressor,
    **kwargs: Any,
) -> Any:
    """Return a small, seeded search of ``estimator`` (Ridge by default).

    ``search_class`` is the class of the search, and ``kwargs`` override
    the default arguments of the search.
    """
    if estimator is None:
        estimator = Ridge()
    params: dict[str, Any] = dict(n_trials=N_TRIALS, cv=CV, random_state=SEED)
    params.update(kwargs)
    return search_class(estimator, ParamDistributions(estimator), **params)


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

    optuna does not propagate its records to the root logger (so
    ``caplog`` does not see them); its loggers (``optuna.*``) all pass
    their records to the handlers of the ``optuna`` logger.
    """
    handler = _RecordsHandler()
    logger = logging.getLogger("optuna")
    logger.addHandler(handler)
    try:
        yield handler.records
    finally:
        logger.removeHandler(handler)


@pytest.fixture
def keep_optuna_verbosity() -> Iterator[None]:
    """Restore the verbosity of optuna after the test changes it."""
    previous = optuna.logging.get_verbosity()
    try:
        yield
    finally:
        optuna.logging.set_verbosity(previous)


def _verbosity_recorder(levels: list[int]) -> Callable[[Any, Any], None]:
    """Return an optuna callback that records the verbosity of optuna."""

    def callback(study: Any, trial: Any) -> None:
        levels.append(optuna.logging.get_verbosity())

    return callback


def _same_value(a: Any, b: Any) -> bool:
    """Return whether two parameter values are equal (NaN equals NaN)."""
    if isinstance(a, float) and isinstance(b, float):
        return a == b or (math.isnan(a) and math.isnan(b))
    return bool(a == b)


def test_is_a_regressor() -> None:
    search = _make_search()

    assert is_regressor(search)
    assert search._estimator_type == "regressor"


@pytest.mark.skipif(
    not hasattr(BaseEstimator, "__sklearn_tags__"),
    reason="estimator tags exist from scikit-learn 1.6",
)
def test_sklearn_tags_only_change_the_estimator_type() -> None:
    search = _make_search()

    expected = OptunaSearchCV.__sklearn_tags__(search)
    expected.estimator_type = "regressor"

    assert search.__sklearn_tags__() == expected


def test_keeps_the_parameters_of_optuna_search_cv() -> None:
    search = _make_search()
    parent = OptunaSearchCV(Ridge(), ParamDistributions(Ridge()))

    assert "__init__" not in vars(OptunaSearchRegressor)
    assert set(search.get_params(deep=False)) == set(
        parent.get_params(deep=False)
    )


def test_predict_is_present_before_fit() -> None:
    search = _make_search()

    assert hasattr(search, "predict")
    with pytest.raises(NotFittedError):
        search.predict(X_SMALL)

    search.fit(X_SMALL, y_SMALL)

    assert_allclose(
        search.predict(X_SMALL), search.best_estimator_.predict(X_SMALL)
    )


def test_predict_works_when_the_parent_defines_it_as_a_property() -> None:
    # The simulated old parent hides predict before fit.
    assert not hasattr(
        _make_search(search_class=_PropertyPredictSearchCV), "predict"
    )
    search = _make_search(search_class=_OldParentSearchRegressor)
    assert hasattr(search, "predict")

    ensemble = StackingRegressor(
        [
            ("ridge", search),
            (
                "lasso",
                _make_search(Lasso(), search_class=_OldParentSearchRegressor),
            ),
        ],
        cv=CV,
    )
    ensemble.fit(X_SMALL, y_SMALL)

    assert ensemble.predict(X_SMALL).shape == (N_SAMPLES,)
    for fitted in ensemble.estimators_:
        assert isinstance(fitted, _OldParentSearchRegressor)
        assert_allclose(
            fitted.predict(X_SMALL), fitted.best_estimator_.predict(X_SMALL)
        )


@ENSEMBLE_FACTORIES
def test_fits_inside_sklearn_ensembles(
    make_ensemble: Callable[[list[tuple[str, Any]]], Any],
) -> None:
    ensemble = make_ensemble(
        [
            ("ridge", _make_search(Ridge())),
            ("lasso", _make_search(Lasso())),
        ]
    )

    ensemble.fit(X_SMALL, y_SMALL)
    predicted = ensemble.predict(X_SMALL)

    assert predicted.shape == (N_SAMPLES,)
    assert np.all(np.isfinite(predicted))
    assert len(ensemble.estimators_) == 2
    for fitted in ensemble.estimators_:
        assert isinstance(fitted, OptunaSearchCV)
        assert isinstance(fitted, OptunaSearchRegressor)
        assert len(fitted.study_.trials) == N_TRIALS

    loaded = pickle.loads(pickle.dumps(ensemble))
    assert_allclose(loaded.predict(X_SMALL), predicted)


def test_fit_takes_groups_like_optuna_search_cv() -> None:
    assert list(inspect.signature(OptunaSearchRegressor.fit).parameters) == (
        list(inspect.signature(OptunaSearchCV.fit).parameters)
    )
    groups = np.arange(N_SAMPLES) % 5

    positional = _make_search(cv=GroupKFold(n_splits=CV))
    positional.fit(X_SMALL, y_SMALL, groups)
    keyword = _make_search(cv=GroupKFold(n_splits=CV))
    keyword.fit(X_SMALL, y_SMALL, groups=groups)

    assert positional.n_splits_ == keyword.n_splits_ == CV
    assert positional.best_params_ == keyword.best_params_


def test_n_features_in_comes_from_the_best_estimator() -> None:
    search = _make_search()

    assert not hasattr(search, "n_features_in_")
    assert not hasattr(search, "feature_names_in_")
    with pytest.raises(AttributeError, match="n_features_in_"):
        search.n_features_in_

    search.fit(X_SMALL, y_SMALL)

    assert search.n_features_in_ == N_FEATURES
    assert search.n_features_in_ == search.best_estimator_.n_features_in_
    # Fitted on an array: the best estimator has no feature names.
    assert not hasattr(search, "feature_names_in_")


def test_no_n_features_in_without_refit() -> None:
    search = _make_search(refit=False).fit(X_SMALL, y_SMALL)

    assert not hasattr(search, "n_features_in_")
    assert not hasattr(search, "feature_names_in_")


@pytest.mark.skipif(
    not HAS_FEATURE_NAMES,
    reason="feature_names_in_ exists from scikit-learn 1.0",
)
def test_feature_names_in_comes_from_the_best_estimator() -> None:
    X_frame = pd.DataFrame(X_SMALL, columns=COLUMNS)

    search = _make_search().fit(X_frame, y_SMALL)

    assert search.n_features_in_ == N_FEATURES
    assert list(search.feature_names_in_) == COLUMNS


@ENSEMBLE_FACTORIES
@pytest.mark.parametrize("as_frame", [False, True], ids=["array", "frame"])
def test_sklearn_ensembles_expose_the_features(
    make_ensemble: Callable[[list[tuple[str, Any]]], Any], as_frame: bool
) -> None:
    if as_frame and not HAS_FEATURE_NAMES:
        pytest.skip("feature_names_in_ exists from scikit-learn 1.0")
    X = pd.DataFrame(X_SMALL, columns=COLUMNS) if as_frame else X_SMALL
    ensemble = make_ensemble(
        [
            ("ridge", _make_search(Ridge())),
            ("lasso", _make_search(Lasso())),
        ]
    )

    ensemble.fit(X, y_SMALL)

    assert ensemble.n_features_in_ == N_FEATURES
    if as_frame:
        assert list(ensemble.feature_names_in_) == COLUMNS
    else:
        assert not hasattr(ensemble, "feature_names_in_")


def test_clone_keeps_the_parameters() -> None:
    search = _make_search(
        n_trials=3, cv=4, scoring="neg_mean_absolute_error", verbose=1
    )

    cloned = clone(search)

    assert type(cloned) is OptunaSearchRegressor
    params = search.get_params(deep=False)
    cloned_params = cloned.get_params(deep=False)
    assert set(cloned_params) == set(params)
    for name, value in params.items():
        if name == "estimator":
            assert cloned_params[name] is not value
            assert cloned_params[name].get_params() == value.get_params()
        else:
            assert _same_value(cloned_params[name], value), name
    assert type(cloned_params["param_distributions"]) is ParamDistributions


def test_pickle_round_trip_of_a_fitted_search() -> None:
    search = _make_search().fit(X_SMALL, y_SMALL)

    loaded = pickle.loads(pickle.dumps(search))

    assert type(loaded) is OptunaSearchRegressor
    assert loaded.best_params_ == search.best_params_
    assert_allclose(loaded.predict(X_SMALL), search.predict(X_SMALL))


@pytest.mark.usefixtures("keep_optuna_verbosity")
@pytest.mark.parametrize(
    "verbosity",
    [
        optuna.logging.DEBUG,
        optuna.logging.INFO,
        optuna.logging.WARNING,
        optuna.logging.ERROR,
    ],
)
def test_verbose_0_hides_the_info_logs_of_optuna(
    verbosity: int, optuna_records: list[logging.LogRecord]
) -> None:
    optuna.logging.set_verbosity(verbosity)
    levels: list[int] = []
    search = _make_search(verbose=0, callbacks=[_verbosity_recorder(levels)])

    search.fit(X_SMALL, y_SMALL)

    # Lowered to WARNING during fit, but never made more verbose.
    assert levels == [max(verbosity, optuna.logging.WARNING)] * N_TRIALS
    assert [
        record.getMessage()
        for record in optuna_records
        if record.levelno < logging.WARNING
    ] == []
    assert optuna.logging.get_verbosity() == verbosity


@pytest.mark.usefixtures("keep_optuna_verbosity")
def test_verbose_1_keeps_the_verbosity_of_optuna(
    optuna_records: list[logging.LogRecord],
) -> None:
    optuna.logging.set_verbosity(optuna.logging.INFO)
    levels: list[int] = []
    search = _make_search(verbose=1, callbacks=[_verbosity_recorder(levels)])

    search.fit(X_SMALL, y_SMALL)

    assert levels == [optuna.logging.INFO] * N_TRIALS
    # The trial logs reach the handler when they are not hidden.
    assert any(
        record.levelno == logging.INFO and "Trial" in record.getMessage()
        for record in optuna_records
    )
    assert optuna.logging.get_verbosity() == optuna.logging.INFO


@pytest.mark.usefixtures("keep_optuna_verbosity")
def test_verbosity_is_restored_when_fit_raises() -> None:
    optuna.logging.set_verbosity(optuna.logging.INFO)
    levels: list[int] = []

    def failing_callback(study: Any, trial: Any) -> None:
        levels.append(optuna.logging.get_verbosity())
        raise RuntimeError("stop the search on purpose")

    search = _make_search(verbose=0, callbacks=[failing_callback])

    with pytest.raises(RuntimeError, match="on purpose"):
        search.fit(X_SMALL, y_SMALL)

    # The error was raised during the search, while the logs were hidden.
    assert levels == [optuna.logging.WARNING]
    assert optuna.logging.get_verbosity() == optuna.logging.INFO


def _wait(event: threading.Event, what: str) -> None:
    """Wait for ``event``, raising an error after ``TIMEOUT`` seconds."""
    if not event.wait(TIMEOUT):
        raise RuntimeError(f"timed out waiting for {what}")


def _pause_after_the_first_trial(
    paused: threading.Event, release: threading.Event
) -> Callable[[Any, Any], None]:
    """Return a callback that, after trial 0, waits until ``release``."""

    def callback(study: Any, trial: Any) -> None:
        if trial.number == 0:
            paused.set()
            _wait(release, "the release of the search")

    return callback


@pytest.mark.usefixtures("keep_optuna_verbosity")
def test_concurrent_fits_restore_the_verbosity_after_the_last_one() -> None:
    optuna.logging.set_verbosity(optuna.logging.INFO)
    a_paused, release_a = threading.Event(), threading.Event()
    b_paused, release_b = threading.Event(), threading.Event()
    search_a = _make_search(
        verbose=0,
        callbacks=[_pause_after_the_first_trial(a_paused, release_a)],
    )
    search_b = _make_search(
        verbose=0,
        callbacks=[_pause_after_the_first_trial(b_paused, release_b)],
    )
    errors: list[BaseException] = []

    def fit(search: OptunaSearchRegressor) -> None:
        try:
            search.fit(X_SMALL, y_SMALL)
        except BaseException as exc:  # reported by the main thread
            errors.append(exc)

    thread_a = threading.Thread(target=fit, args=(search_a,))
    thread_b = threading.Thread(target=fit, args=(search_b,))
    try:
        # A starts first and B starts while A is searching.
        thread_a.start()
        _wait(a_paused, "search A")
        thread_b.start()
        _wait(b_paused, "search B")
        assert optuna.logging.get_verbosity() == optuna.logging.WARNING

        # A finishes first: B is still searching, so the logs stay hidden.
        release_a.set()
        thread_a.join(TIMEOUT)
        assert not thread_a.is_alive()
        assert optuna.logging.get_verbosity() == optuna.logging.WARNING

        # B finishes last and restores the verbosity from before A.
        release_b.set()
        thread_b.join(TIMEOUT)
        assert not thread_b.is_alive()
    finally:
        release_a.set()
        release_b.set()
        for thread in (thread_a, thread_b):
            if thread.ident is not None:
                thread.join(TIMEOUT)

    assert errors == []
    assert (
        len(search_a.study_.trials) == len(search_b.study_.trials) == N_TRIALS
    )
    assert optuna.logging.get_verbosity() == optuna.logging.INFO
