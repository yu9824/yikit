# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3.9.13 ('yikit')
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import optuna
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from tqdm.auto import tqdm

from yikit.models import GBDTRegressor, Objective

optuna.logging.set_verbosity(optuna.logging.WARNING)

# %%
SEED = 334

# %%
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = pd.Series(diabetes.target, name="target")
display(X.head(), y.head())

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)


# %% [markdown]
# 標準化は```Pipeline```と```TransformedTargetRegressor```で行う．
# 入れ子のモデルも，予測するモデルの探索範囲（```regressor__svr__C```など）で探索される．
# 削除した```SupportVectorRegressor```・```LinearModelRegressor```の代わりに，```SVR```・```Ridge```・```Lasso```・```ElasticNet```を使う．

# %%
def scaled(regressor):
    return TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), regressor),
        transformer=StandardScaler(),
    )


estimators = {
    "MLP": scaled(MLPRegressor(max_iter=1000)),
    "GBDT": GBDTRegressor(),
    "RandomForest": RandomForestRegressor(),
    "SVR": scaled(SVR()),
    "Ridge": scaled(Ridge()),
    "Lasso": scaled(Lasso()),
    "ElasticNet": scaled(ElasticNet()),
}


def get_best_estimator(estimator):
    objective = Objective(
        estimator,
        X_train,
        y_train,
        scoring="neg_root_mean_squared_error",
        random_state=SEED,
        cv=kf,
    )
    study = optuna.create_study(
        sampler=objective.sampler, direction="maximize"
    )
    study.optimize(objective, n_trials=10)
    return objective.get_best_estimator(study)


best_estimators = {
    name: get_best_estimator(estimator)
    for name, estimator in tqdm(estimators.items())
}

# %%
pd.Series(
    {
        name: np.sqrt(
            mean_squared_error(
                y_test, estimator.fit(X_train, y_train).predict(X_test)
            )
        )
        for name, estimator in best_estimators.items()
    },
    name="RMSE",
)
