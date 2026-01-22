import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor

from yikit.feature_selection import BorutaPy


def test_boruta():
    diabetes = load_diabetes()
    X = pd.DataFrame(diabetes["data"], columns=diabetes["feature_names"])
    y = pd.Series(diabetes["target"])

    (
        BorutaPy(
            RandomForestRegressor(n_jobs=-1, random_state=334),
            random_state=334,
        )
        .fit(X, y)
        .support_
    )
