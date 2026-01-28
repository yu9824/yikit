import numpy as np
from sklearn.ensemble import RandomForestRegressor

from yikit.feature_selection import BorutaPy


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
