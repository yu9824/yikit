# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.20.0
#   kernelspec:
#     display_name: Python 3.9.13 ('yikit')
#     language: python
#     name: python3
# ---

# %%
from yikit.models import Objective
from yikit.models import SupportVectorRegressor, GBDTRegressor, LinearModelRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_diabetes
from sklearn.neural_network import MLPRegressor
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold, train_test_split
import optuna
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm

# %%
SEED = 334

# %%
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = pd.DataFrame(diabetes.target, columns=['PRICE'])
display(X.head(), y.head())

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
kf = KFold(n_splits=5)

# %%
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)
estimators = [MLPRegressor(max_iter=1000), GBDTRegressor(), RandomForestRegressor(), SupportVectorRegressor(), GBDTRegressor(), LinearModelRegressor()]
def get_best_estimator(estimator):
    objective = Objective(estimator=estimator, X=X_train, y=y_train.ravel(), scoring='neg_mean_squared_error', random_state=SEED, cv=kf)
    study = optuna.create_study(sampler=objective.sampler, direction='maximize')
    study.optimize(objective, n_trials=10, n_jobs=1)
    best_estimator_ = clone(estimator)
    best_estimator_.set_params(**objective.fixed_params_, **study.best_params)
    return best_estimator_

optuna.logging.disable_default_handler()
[get_best_estimator(estimator) for estimator in tqdm(estimators)]

# %%
optuna.logging.enable_default_handler()
