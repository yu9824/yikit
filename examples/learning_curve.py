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
import optuna
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.base import clone
from sklearn.model_selection import KFold
import pandas as pd
from yikit.models import Objective
from yikit.tools.visualization import get_learning_curve_optuna

# %%
SEED = 334

# %%
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns = diabetes.feature_names)
y = pd.Series(diabetes.target)

# %%
rf = RandomForestRegressor(n_jobs = -1, random_state = SEED)

# %%
objective = Objective(rf, X, y, scoring = 'neg_root_mean_squared_error', n_jobs = -1, random_state = SEED, cv = KFold(n_splits=5, shuffle=True, random_state=SEED))

# %% tags=["outputPrepend"]
study = optuna.create_study(direction = 'maximize', sampler = objective.sampler)
study.optimize(objective, n_trials = 100)

# %%
get_learning_curve_optuna(study)

# %%
rf_opt = clone(rf).set_params(**study.best_params)

# %%
kf = KFold(n_splits = 5, shuffle = True, random_state = SEED)
for i_train, i_test in kf.split(X, y):
    rf_opt = clone(rf_opt)
    X_train, y_train = X.loc[i_train], y[i_train]
    X_test, y_test = X.loc[i_test], y[i_test]

    rf_opt.fit(X_train, y_train)
    print(mean_squared_error(y_test, rf_opt.predict(X_test), squared = False))
