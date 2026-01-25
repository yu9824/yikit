import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from yikit.feature_selection import BorutaPy


def test_boruta(X_regression, y_regression):
    X = X_regression
    y = y_regression

    (
        BorutaPy(
            RandomForestRegressor(n_jobs=-1, random_state=334),
            random_state=334,
        )
        .fit(X, y)
        .support_
    )
