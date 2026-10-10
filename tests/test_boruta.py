from __future__ import annotations

import numbers

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
