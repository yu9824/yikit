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
from yikit.models import EnsembleRegressor, Objective

import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes
import optuna
from lightgbm import LGBMRegressor

# %%
SEED = 334
kf = KFold(n_splits = 5, shuffle=True, random_state=SEED)

# %%
data = load_diabetes()
X = pd.DataFrame(data.data, columns = data.feature_names)
y = pd.Series(data.target, name = 'PRICE')

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=SEED)

# %%
# rf = RandomForestRegressor(random_state = SEED, n_jobs = -1)
lgbt = LGBMRegressor(random_state = SEED, n_jobs = -1)

# %% tags=["outputPrepend"]
objective = Objective(lgbt, X_train, y_train, scoring = 'neg_root_mean_squared_error', cv = kf)
study = optuna.create_study(sampler = objective.sampler, direction = 'maximize')
study.optimize(objective, n_trials = 100)

# %%
best_estimator = objective.model(**objective.fixed_params_, **study.best_params).fit(X_train, y_train)

# %%
mean_squared_error(best_estimator.predict(X_test), y_test, squared = False)

# %% [markdown]
# 同じ条件にするために```boruta```を```False```に．

# %%
er = EnsembleRegressor([lgbt], random_state = SEED, n_jobs = -1, boruta = False, scoring = 'neg_root_mean_squared_error', verbose = 0, cv = kf)

# %%
er.fit(X_train, y_train)

# %%
mean_squared_error(er.predict(X_test), y_test, squared = False)

# %%
for estimators in er.results_.estimators:
    print(mean_squared_error(estimators[0].predict(X_test), y_test, squared = False))
