from ngboost import NGBClassifier, NGBRegressor


def log_likelihood(estimator, X, y):
    if isinstance(estimator, (NGBRegressor, NGBClassifier)):
        return -estimator.score(X, y)  # LL (- NLL)
    else:
        raise TypeError(
            "'estimator' is not {0} or {1}".format(NGBRegressor, NGBClassifier)
        )
