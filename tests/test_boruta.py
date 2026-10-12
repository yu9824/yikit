from __future__ import annotations

import logging
import numbers
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline

pytest.importorskip("boruta")

from yikit.feature_selection import BorutaPy  # noqa: E402

SEED = 334

#: Small data for the fast BorutaPy tests (3 informative of 6 features).
X_SMALL, y_SMALL = make_regression(
    n_samples=80,
    n_features=6,
    n_informative=3,
    noise=1.0,
    shuffle=False,
    random_state=SEED,
)


def _make_selector(**kwargs) -> BorutaPy:
    """Return a fast, deterministic BorutaPy; ``kwargs`` override defaults."""
    params = dict(
        estimator=RandomForestRegressor(
            n_estimators=50, n_jobs=1, random_state=SEED
        ),
        max_iter=20,
        max_shuf=50,
        verbose=0,
        n_jobs=1,
        random_state=SEED,
    )
    params.update(kwargs)
    return BorutaPy(**params)


class _FailingRegressor(BaseEstimator, RegressorMixin):
    """Regressor whose ``fit`` always raises, to break ``boruta._fit``."""

    def __init__(self, n_estimators=10, random_state=None):
        self.n_estimators = n_estimators
        self.random_state = random_state

    def fit(self, X, y):
        raise RuntimeError("fit failed on purpose")


def test_boruta(X_regression, y_regression):
    X = X_regression
    y = y_regression

    selector = BorutaPy(
        RandomForestRegressor(n_jobs=1, random_state=334),
        random_state=334,
        n_jobs=1,
    )
    selector.fit(X, y)
    assert np.allclose(
        selector.support_,
        np.array(
            [
                True,
                True,
                False,
                True,
                True,
                True,
                True,
                True,
                True,
                True,
            ]
        ),
    )


def test_fit_transform_with_auto_perc_selects_the_same_features_as_fit():
    fitted = _make_selector(perc="auto").fit(X_SMALL, y_SMALL)

    selector = _make_selector(perc="auto")
    X_selected = selector.fit_transform(X_SMALL, y_SMALL)

    assert selector.perc_ == fitted.perc_
    np.testing.assert_array_equal(selector.support_, fitted.support_)
    np.testing.assert_array_equal(selector.support_weak_, fitted.support_weak_)
    np.testing.assert_array_equal(selector.ranking_, fitted.ranking_)
    assert selector.support_.any()
    np.testing.assert_array_equal(X_selected, X_SMALL[:, fitted.support_])


def test_fit_keeps_auto_perc_and_stores_the_resolved_value_in_perc_():
    selector = _make_selector(perc="auto")

    returned = selector.fit(X_SMALL, y_SMALL)

    assert returned is selector
    assert selector.get_params()["perc"] == "auto"
    assert selector.perc == "auto"
    assert isinstance(selector.perc_, numbers.Real)
    assert 0 < selector.perc_ <= 100
    assert selector.perc_ == pytest.approx(100 * (1 - selector.r_ccmax_))


def test_numeric_perc_is_used_as_perc_():
    selector = _make_selector(perc=80)

    selector.fit(X_SMALL, y_SMALL)

    assert selector.perc_ == 80
    assert selector.get_params()["perc"] == 80
    assert not hasattr(selector, "r_ccmax_")


def test_fit_restores_perc_when_boruta_fails():
    selector = _make_selector(
        perc="auto", estimator=_FailingRegressor(), n_estimators=10
    )

    # boruta 0.4.3 re-raises the estimator's error as a ValueError that
    # contains the original message.
    with pytest.raises(
        (RuntimeError, ValueError), match="fit failed on purpose"
    ):
        selector.fit(X_SMALL, y_SMALL)

    assert selector.perc == "auto"


def test_fit_transform_sets_up_the_progress_bar():
    # With tqdm installed and verbose=1, boruta's loop updates ``_pbar``,
    # so ``fit_transform`` must create it as ``fit`` does.
    selector = _make_selector(perc=80, verbose=1)

    X_selected = selector.fit_transform(X_SMALL, y_SMALL)

    assert X_selected.shape == (X_SMALL.shape[0], selector.n_features_)


def test_boruta_with_auto_perc_fits_and_predicts_in_a_pipeline():
    pipeline = Pipeline(
        [
            ("boruta", _make_selector(perc="auto")),
            ("ridge", Ridge()),
        ]
    )

    pipeline.fit(X_SMALL, y_SMALL)
    y_pred = pipeline.predict(X_SMALL)

    boruta = pipeline.named_steps["boruta"]
    assert y_pred.shape == y_SMALL.shape
    assert boruta.get_params()["perc"] == "auto"
    assert 0 < boruta.perc_ <= 100
    assert pipeline.named_steps["ridge"].n_features_in_ == boruta.n_features_


def _small_dataframe() -> pd.DataFrame:
    """Return ``X_SMALL`` as a DataFrame with string column names."""
    return pd.DataFrame(
        X_SMALL, columns=[f"x{i}" for i in range(X_SMALL.shape[1])]
    )


@pytest.mark.parametrize("weak", [False, True])
def test_transform_of_a_dataframe_returns_the_selected_columns_as_array(
    weak,
):
    # boruta 0.4.3 indexes X with ``X[:, mask]`` when ``return_df=False``,
    # which a DataFrame does not support.
    X_df = _small_dataframe()
    expected = _make_selector().fit(X_SMALL, y_SMALL).transform(X_SMALL, weak)

    selector = _make_selector()
    X_fit_transformed = selector.fit_transform(X_df, y_SMALL, weak=weak)
    X_transformed = selector.transform(X_df, weak=weak)

    for X_selected in (X_fit_transformed, X_transformed):
        assert type(X_selected) is np.ndarray
        np.testing.assert_array_equal(X_selected, expected)


def test_return_df_keeps_the_selected_columns_of_a_dataframe():
    X_df = _small_dataframe()
    selector = _make_selector()

    X_fit_transformed = selector.fit_transform(X_df, y_SMALL, return_df=True)
    X_transformed = selector.transform(X_df, return_df=True)

    expected = X_df.loc[:, selector.support_]
    assert selector.support_.any()
    for X_selected in (X_fit_transformed, X_transformed):
        assert isinstance(X_selected, pd.DataFrame)
        pd.testing.assert_frame_equal(X_selected, expected)


def test_boruta_fits_and_predicts_in_a_pipeline_on_a_dataframe():
    X_df = _small_dataframe()
    expected = (
        Pipeline([("boruta", _make_selector()), ("ridge", Ridge())])
        .fit(X_SMALL, y_SMALL)
        .predict(X_SMALL)
    )
    pipeline = Pipeline([("boruta", _make_selector()), ("ridge", Ridge())])

    y_pred = pipeline.fit(X_df, y_SMALL).predict(X_df)

    boruta = pipeline.named_steps["boruta"]
    assert boruta.get_params()["perc"] == "auto"
    assert pipeline.named_steps["ridge"].n_features_in_ == boruta.n_features_
    np.testing.assert_allclose(y_pred, expected)


def test_clone_of_a_fitted_selector_keeps_auto_perc():
    selector = _make_selector(perc="auto").fit(X_SMALL, y_SMALL)

    cloned = clone(selector)

    assert cloned.get_params()["perc"] == "auto"
    assert not hasattr(cloned, "perc_")


# --- scikit-learn conventions, random state and display ----------------------


class _RecordingProgressBar:
    """Stand-in for ``tqdm`` that records how BorutaPy uses it."""

    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs
        self.n = 0
        self.closed = False

    def update(self, n=1):
        self.n += n

    def close(self):
        self.closed = True


@pytest.fixture
def progress_bars(monkeypatch):
    """Record the progress bars that BorutaPy makes for its iterations.

    BorutaPy shows them only when tqdm is installed, so the tests that use
    this fixture are skipped without tqdm.
    """
    pytest.importorskip("tqdm")
    from yikit.feature_selection import _wrapper

    bars: list[_RecordingProgressBar] = []

    def make_progress_bar(*args, **kwargs):
        bar = _RecordingProgressBar(*args, **kwargs)
        bars.append(bar)
        return bar

    monkeypatch.setattr(_wrapper, "tqdm", make_progress_bar)
    return bars


class _MessageList(logging.Handler):
    """Logging handler that keeps the messages it handles."""

    def __init__(self):
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(record.getMessage())


@pytest.fixture
def boruta_messages():
    """Yield the messages logged by the BorutaPy module during the test."""
    from yikit.feature_selection import _wrapper

    handler = _MessageList()
    level = _wrapper.logger.level
    _wrapper.logger.setLevel(logging.INFO)
    _wrapper.logger.addHandler(handler)
    try:
        yield handler.messages
    finally:
        _wrapper.logger.removeHandler(handler)
        _wrapper.logger.setLevel(level)


def _fitted_result(selector: BorutaPy) -> dict:
    """Return copies of the fitted attributes that a fit decides."""
    return {
        "perc_": selector.perc_,
        "support_": selector.support_.copy(),
        "support_weak_": selector.support_weak_.copy(),
        "ranking_": selector.ranking_.copy(),
    }


def _assert_same_result(actual: dict, expected: dict) -> None:
    """Check that two results of ``_fitted_result`` are the same."""
    assert actual["perc_"] == expected["perc_"]
    for name in ("support_", "support_weak_", "ranking_"):
        np.testing.assert_array_equal(actual[name], expected[name])


@pytest.mark.parametrize(
    "random_state",
    [SEED, None, pytest.param(np.random.RandomState(SEED), id="RandomState")],
)
def test_random_state_is_kept_as_passed(random_state):
    selector = _make_selector(perc="auto", random_state=random_state)
    assert selector.get_params()["random_state"] is random_state

    selector.fit(X_SMALL, y_SMALL)

    assert selector.get_params()["random_state"] is random_state
    assert selector.random_state is random_state


def test_fitting_twice_with_an_int_random_state_gives_the_same_result():
    selector = _make_selector(perc="auto")

    first = _fitted_result(selector.fit(X_SMALL, y_SMALL))
    second = _fitted_result(selector.fit(X_SMALL, y_SMALL))
    cloned = _fitted_result(clone(selector).fit(X_SMALL, y_SMALL))

    assert selector.support_.any()
    _assert_same_result(second, first)
    _assert_same_result(cloned, first)


def test_auto_perc_does_not_depend_on_n_jobs():
    max_shuf = 50
    selectors = {
        n_jobs: _make_selector(
            perc="auto", n_jobs=n_jobs, max_shuf=max_shuf
        ).fit(X_SMALL, y_SMALL)
        for n_jobs in (1, 2)
    }

    assert selectors[2].perc_ == selectors[1].perc_
    pears = np.asarray(selectors[2].pears_)
    np.testing.assert_array_equal(pears, np.asarray(selectors[1].pears_))
    # Each shuffle has its own seed, so the workers do not repeat one.
    assert pears.shape == (max_shuf, X_SMALL.shape[1])
    assert len(np.unique(pears, axis=0)) == max_shuf


def test_set_params_verbose_takes_effect_on_the_next_fit(
    progress_bars, boruta_messages
):
    max_iter = 20
    selector = _make_selector(perc=80, verbose=0, max_iter=max_iter)

    selector.fit(X_SMALL, y_SMALL)
    assert progress_bars == []
    assert boruta_messages == []

    # verbose=1 with tqdm: a progress bar, and only the summary is logged.
    selector.set_params(verbose=1).fit(X_SMALL, y_SMALL)
    assert len(progress_bars) == 1
    assert progress_bars[0].kwargs["total"] == max_iter
    assert progress_bars[0].n == max_iter
    assert progress_bars[0].closed
    assert len(boruta_messages) == 1
    assert "BorutaPy finished running" in boruta_messages[0]

    # verbose=2: a progress bar, and every iteration is logged too.
    del boruta_messages[:]
    selector.set_params(verbose=2).fit(X_SMALL, y_SMALL)
    assert len(progress_bars) == 2
    assert progress_bars[1].closed
    assert len(boruta_messages) > 1
    assert all("Confirmed" in message for message in boruta_messages)

    # verbose=0 again: no progress bar and nothing logged.
    del boruta_messages[:]
    selector.set_params(verbose=0).fit(X_SMALL, y_SMALL)
    assert len(progress_bars) == 2
    assert boruta_messages == []


@pytest.mark.parametrize("random_state", [SEED, None])
def test_an_error_in_fit_closes_the_progress_bar_and_restores_parameters(
    progress_bars, random_state
):
    selector = _make_selector(
        perc="auto",
        verbose=1,
        estimator=_FailingRegressor(),
        n_estimators=10,
        random_state=random_state,
    )

    # boruta 0.4.3 re-raises the estimator's error as a ValueError that
    # contains the original message.
    with pytest.raises(
        (RuntimeError, ValueError), match="fit failed on purpose"
    ):
        selector.fit(X_SMALL, y_SMALL)

    assert [bar.closed for bar in progress_bars] == [True]
    assert selector.perc == "auto"
    assert selector.get_params()["random_state"] is random_state


def test_a_fit_with_a_numeric_perc_removes_the_attributes_of_auto_perc():
    selector = _make_selector(perc="auto").fit(X_SMALL, y_SMALL)
    assert hasattr(selector, "r_ccmax_")
    assert hasattr(selector, "pears_")

    selector.set_params(perc=80).fit(X_SMALL, y_SMALL)

    assert selector.perc_ == 80
    assert not hasattr(selector, "r_ccmax_")
    assert not hasattr(selector, "pears_")


def test_a_fitted_selector_with_a_progress_bar_can_be_pickled():
    pytest.importorskip("tqdm")
    selector = _make_selector(perc=80, verbose=1).fit(X_SMALL, y_SMALL)

    restored = pickle.loads(pickle.dumps(selector))

    np.testing.assert_array_equal(restored.support_, selector.support_)
    np.testing.assert_array_equal(
        restored.transform(X_SMALL), selector.transform(X_SMALL)
    )
