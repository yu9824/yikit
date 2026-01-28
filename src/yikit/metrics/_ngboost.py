from ngboost import (  # type: ignore[reportMissingImports]
    NGBClassifier,
    NGBRegressor,
)


def log_likelihood(estimator, X, y):
    """
    Compute the negative log-likelihood for an NGBoost estimator.

    Parameters
    ----------
    estimator : NGBRegressor or NGBClassifier
        A fitted NGBoost regressor or classifier.
    X : array-like of shape (n_samples, n_features)
        Input samples for which to compute the log-likelihood.
    y : array-like of shape (n_samples,)
        True target values or class labels.

    Returns
    -------
    float
        The negative log-likelihood computed by the estimator.
        Returns `-estimator.score(X, y)`.

    Raises
    ------
    TypeError
        If the estimator is not an instance of NGBRegressor or NGBClassifier.

    Notes
    -----
    This function is only supported for NGBoost estimators.
    """

    if isinstance(estimator, (NGBRegressor, NGBClassifier)):
        return -estimator.score(X, y)  # LL (- NLL)
    else:
        raise TypeError(
            "'estimator' is not {0} or {1}".format(NGBRegressor, NGBClassifier)
        )
