"""Tests of ``yikit.metrics`` and the RMSE used by ``yikit.visualize``."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

import yikit.metrics
from yikit.helpers import is_installed
from yikit.visualize import _yyplot

SEED = 334

#: Small number of boosting iterations to keep NGBoost fast.
N_ESTIMATORS = 20


def _regression_data() -> tuple[np.ndarray, np.ndarray]:
    """Return a small regression data set with a positive target.

    The target is positive so that the same data fits every distribution
    in the tests (``LogNormal`` and ``Exponential`` need ``y > 0``).
    """
    X, y = make_regression(
        n_samples=40, n_features=3, noise=5.0, random_state=SEED
    )
    return X, np.exp(y / np.std(y))


def _classification_data() -> tuple[np.ndarray, np.ndarray]:
    """Return a small binary classification data set with noisy labels."""
    X, _ = _regression_data()
    rng = np.random.RandomState(SEED)
    y = (X[:, 0] + rng.normal(size=len(X)) > 0).astype(int)
    return X, y


def _base() -> DecisionTreeRegressor:
    """Return a fresh base learner (NGBoost's default is a shared object)."""
    return DecisionTreeRegressor(max_depth=3, random_state=SEED)


# --- Public names ------------------------------------------------------------


def test_public_names():
    expected = ("log_likelihood",) if is_installed("ngboost") else ()
    assert tuple(yikit.metrics.__all__) == expected


def test_removed_rmse_raises_import_error():
    assert "root_mean_squared_error" not in yikit.metrics.__all__
    with pytest.raises(ImportError, match="cannot import name"):
        from yikit.metrics import root_mean_squared_error  # noqa: F401


# --- log_likelihood ----------------------------------------------------------


@pytest.mark.parametrize(
    ("dist_name", "score_name"),
    [
        ("Normal", "LogScore"),
        ("Normal", "CRPScore"),
        # LogNormal and Exponential have censored scores only, so
        # NGBRegressor wraps them. LogNormal with CRPScore is left out
        # because ngboost 0.3.6 cannot fit it.
        ("LogNormal", "LogScore"),
        ("Exponential", "LogScore"),
        ("Exponential", "CRPScore"),
    ],
)
def test_log_likelihood_of_ngb_regressor(dist_name, score_name):
    ngboost = pytest.importorskip("ngboost")
    from ngboost import distns, scores

    from yikit.metrics import log_likelihood

    X, y = _regression_data()
    ngb = ngboost.NGBRegressor(
        Dist=getattr(distns, dist_name),
        Score=getattr(scores, score_name),
        Base=_base(),
        n_estimators=N_ESTIMATORS,
        verbose=False,
        random_state=SEED,
    ).fit(X, y)

    value = log_likelihood(ngb, X, y)

    # The log-density of the predicted (scipy) distribution, whatever
    # scoring rule the model was trained with.
    assert value == pytest.approx(np.mean(ngb.pred_dist(X).logpdf(y)))
    if score_name == "LogScore":
        # ``score`` is the mean negative log-likelihood for LogScore.
        assert value == pytest.approx(-ngb.score(X, y))
    else:
        # ``score`` is the mean CRPS, which is not a log-likelihood.
        assert value != pytest.approx(-ngb.score(X, y))


def test_log_likelihood_of_ngb_classifier():
    # ngboost's categorical distributions implement only LogScore for
    # training, so ``-score`` is the log-likelihood here.
    ngboost = pytest.importorskip("ngboost")
    from ngboost.distns import Bernoulli

    from yikit.metrics import log_likelihood

    X, y = _classification_data()
    ngb = ngboost.NGBClassifier(
        Dist=Bernoulli,
        Base=_base(),
        n_estimators=N_ESTIMATORS,
        verbose=False,
        random_state=SEED,
    ).fit(X, y)

    value = log_likelihood(ngb, X, y)

    proba_of_true_class = ngb.predict_proba(X)[np.arange(len(y)), y]
    assert value == pytest.approx(np.mean(np.log(proba_of_true_class)))
    assert value == pytest.approx(-ngb.score(X, y))
    assert value < 0


def test_log_likelihood_needs_log_score_of_distribution():
    ngboost = pytest.importorskip("ngboost")
    from ngboost.distns import Normal
    from ngboost.distns.normal import NormalCRPScore
    from ngboost.scores import CRPScore

    from yikit.metrics import log_likelihood

    class NormalWithoutLogScore(Normal):
        scores = [NormalCRPScore]

    X, y = _regression_data()
    ngb = ngboost.NGBRegressor(
        Dist=NormalWithoutLogScore,
        Score=CRPScore,
        Base=_base(),
        n_estimators=N_ESTIMATORS,
        verbose=False,
        random_state=SEED,
    ).fit(X, y)

    with pytest.raises(ValueError, match="NormalWithoutLogScore"):
        log_likelihood(ngb, X, y)


def test_log_likelihood_rejects_other_models():
    pytest.importorskip("ngboost")
    from yikit.metrics import log_likelihood

    X, y = _regression_data()
    model = LinearRegression().fit(X, y)

    with pytest.raises(TypeError, match="NGBRegressor or NGBClassifier"):
        log_likelihood(model, X, y)


# --- RMSE of yyplot ----------------------------------------------------------


def test_yyplot_rmse_is_root_of_mean_squared_error():
    rng = np.random.RandomState(SEED)
    y_true = rng.normal(size=20)
    y_pred = y_true + rng.normal(scale=0.5, size=20)

    value = _yyplot._rmse(y_true, y_pred)

    assert value == pytest.approx(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    assert _yyplot.METRIC_INFO_MAP["rmse"]["func"] is _yyplot._rmse


def test_yyplot_rmse_averages_rmse_of_each_output():
    # Multi-output: the mean of the per-output RMSE (scikit-learn's default
    # ``multioutput="uniform_average"``), not the root of the overall MSE.
    rng = np.random.RandomState(SEED)
    Y = rng.normal(size=(50, 2)) * np.array([1.0, 3.0])
    P = Y + rng.normal(size=(50, 2)) * np.array([0.5, 2.0])

    value = _yyplot._rmse(Y, P)

    expected = np.mean(np.sqrt(np.mean((Y - P) ** 2, axis=0)))
    assert value == pytest.approx(expected)
    assert value != pytest.approx(np.sqrt(np.mean((Y - P) ** 2)))
