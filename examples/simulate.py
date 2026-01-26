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
from yikit.models import GBDTRegressor, EnsembleRegressor
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes
import pickle
from joblib import Parallel, delayed

# %%
SEED = 334

# %%
diabetes = load_diabetes()
X = pd.DataFrame(diabetes['data'], columns = diabetes['feature_names'])
y = pd.Series(diabetes['target'])

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = SEED)

# %%
methods = ('average', 'blending', 'stacking')

def f(method):
    model = EnsembleRegressor(estimators = [RandomForestRegressor(n_estimators = 100, random_state = SEED), GBDTRegressor(random_state = SEED)], random_state = SEED, method = method, boruta = False, n_jobs=-1)
    model.fit(X_train, y_train)
    print('{0}: {1:.4f}'.format(method, mean_squared_error(model.predict(X_test), y_test, squared = False)))

Parallel(n_jobs=-1)(delayed(f)(method) for method in methods)
