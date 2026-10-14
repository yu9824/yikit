"""Log-likelihood of NGBoost models.

This module imports ngboost at the top, so :mod:`yikit.metrics` imports it
and exports :func:`log_likelihood` only when ngboost is installed.
"""

from __future__ import annotations

import numpy as np
from ngboost import (  # type: ignore[reportMissingImports]
    NGBClassifier,
    NGBRegressor,
)
from ngboost.manifold import (  # type: ignore[reportMissingImports]
    manifold,
)
from ngboost.scores import LogScore  # type: ignore[reportMissingImports]

#: Name of the class that ``Distn.uncensor`` creates. ngboost detects the
#: wrapper by this name too (``NGBRegressor.__getstate__``).
_UNCENSORED_DIST_NAME = "DistWithUncensoredScore"


def _log_score_manifold(dist_class):
    """Return ``dist_class`` combined with its LogScore implementation.

    Parameters
    ----------
    dist_class : type
        The distribution class of a fitted model (``estimator.Dist``).

    Returns
    -------
    type
        A class that is built from predicted parameters like
        ``dist_class`` and whose ``total_score(Y)`` is the mean negative
        log-likelihood.

    Raises
    ------
    ValueError
        If ngboost has no LogScore implementation for the distribution.
    """
    name = dist_class.__name__
    try:
        if name == _UNCENSORED_DIST_NAME:
            # NGBRegressor wraps a distribution that has only censored
            # scores (LogNormal, Exponential) with ``Dist.uncensor(Score)``,
            # which keeps only the trained score. Rebuild it for LogScore.
            name = dist_class.__base__.__name__
            dist_class = dist_class.__base__.uncensor(LogScore)
        return manifold(LogScore, dist_class)
    except ValueError as err:
        raise ValueError(
            f"Cannot compute the log-likelihood: ngboost has no LogScore "
            f"implementation for the {name} distribution."
        ) from err


def log_likelihood(estimator, X, y):
    """Return the mean log-likelihood of ``y`` under an NGBoost model.

    The value is the average, over the samples, of the log of the predicted
    probability density (or mass) of each target for a regressor, or of the
    predicted probability of the true class for a classifier. It does not
    depend on the scoring rule (``Score``) the model was trained with.
    Higher is better.

    Parameters
    ----------
    estimator : ngboost.NGBRegressor or ngboost.NGBClassifier
        A fitted NGBoost regressor or classifier.
    X : array-like of shape (n_samples, n_features)
        Input samples.
    y : array-like of shape (n_samples,)
        True target values or class labels.

    Returns
    -------
    float
        The mean log-likelihood. For a model trained with the default
        ``Score=LogScore``, it equals ``-estimator.score(X, y)``.

    Raises
    ------
    TypeError
        If ``estimator`` is not an ``NGBRegressor`` or an ``NGBClassifier``.
    ValueError
        If ngboost has no LogScore implementation for the model's
        distribution.

    Notes
    -----
    ``estimator.score`` returns the mean of the scoring rule the model was
    trained with, so ``-estimator.score(X, y)`` is the log-likelihood only
    for ``LogScore`` (for ``CRPScore`` it is the negated mean CRPS). This
    function instead evaluates ngboost's LogScore implementation for the
    model's distribution at the predicted parameters, with equal weights
    for all samples. For the distributions that ngboost implements with
    censored scores only (such as ``LogNormal`` and ``Exponential``), every
    target is treated as observed (not censored).

    Examples
    --------
    >>> from ngboost import NGBRegressor
    >>> from ngboost.scores import CRPScore
    >>> from sklearn.datasets import make_regression
    >>> from yikit.metrics import log_likelihood
    >>> X, y = make_regression(n_samples=50, random_state=334)
    >>> ngb = NGBRegressor(Score=CRPScore, verbose=False)
    >>> ngb = ngb.fit(X, y)  # doctest: +SKIP
    >>> log_likelihood(ngb, X, y)  # doctest: +SKIP
    """
    if not isinstance(estimator, (NGBRegressor, NGBClassifier)):
        raise TypeError(
            "'estimator' must be an NGBRegressor or NGBClassifier, "
            f"got {type(estimator).__name__}."
        )
    log_score_dist = _log_score_manifold(estimator.Dist)
    # ``NGBoost.score`` builds its scoring object from the same parameters
    # (``pred_dist(X)._params``, present in ngboost 0.3.6 through 0.5.8).
    params = estimator.pred_dist(X)._params
    # ``total_score`` without weights is the mean negative log-likelihood.
    return -log_score_dist(params).total_score(np.asarray(y))
