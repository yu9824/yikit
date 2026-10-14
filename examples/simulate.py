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
from joblib import Parallel, delayed
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR

from yikit.helpers import tqdm_joblib
from yikit.models import EnsembleRegressor, GBDTRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)

# %%
SEED = 334

# %%
diabetes = load_diabetes()
X = pd.DataFrame(diabetes["data"], columns=diabetes["feature_names"])
y = pd.Series(diabetes["target"])

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

# %%
methods = ("average", "blending", "stacking")


def f(method):
    model = EnsembleRegressor(
        estimators=(
            make_pipeline(StandardScaler(), LinearSVR(max_iter=10000)),
            GBDTRegressor(n_jobs=1),
        ),
        method=method,
        random_state=SEED,
        n_trials=20,
    )
    model.fit(X_train, y_train)
    rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
    return method, rmse


with tqdm_joblib(total=len(methods)):
    results = Parallel(n_jobs=-1)(delayed(f)(method) for method in methods)
pd.Series(dict(results), name="RMSE")
