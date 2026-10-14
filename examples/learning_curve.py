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
from sklearn.base import clone
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from yikit.models import Objective
from yikit.visualize import get_learning_curve_optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

# %%
SEED = 334

# %%
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = pd.Series(diabetes.target)

# %%
rf = RandomForestRegressor(n_jobs=-1, random_state=SEED)

# %%
objective = Objective(
    rf,
    X,
    y,
    scoring="neg_root_mean_squared_error",
    random_state=SEED,
    cv=KFold(n_splits=5, shuffle=True, random_state=SEED),
)

# %%
study = optuna.create_study(direction="maximize", sampler=objective.sampler)
study.optimize(objective, n_trials=100)

# %%
get_learning_curve_optuna(study)

# %%
rf_opt = objective.get_best_estimator(study)

# %%
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
for i_train, i_test in kf.split(X, y):
    rf_opt = clone(rf_opt)
    X_train, y_train = X.iloc[i_train], y.iloc[i_train]
    X_test, y_test = X.iloc[i_test], y.iloc[i_test]

    rf_opt.fit(X_train, y_train)
    print(np.sqrt(mean_squared_error(y_test, rf_opt.predict(X_test))))
