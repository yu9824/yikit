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
from lightgbm import LGBMRegressor
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split

from yikit.models import EnsembleRegressor, Objective

optuna.logging.set_verbosity(optuna.logging.WARNING)

# %%
SEED = 334
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

# %%
data = load_diabetes()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

# %%
lgbm = LGBMRegressor(random_state=SEED, n_jobs=-1, verbosity=-1)

# %% [markdown]
# ```Objective```でハイパーパラメータを探索する．

# %%
objective = Objective(
    lgbm,
    X_train,
    y_train,
    scoring="neg_root_mean_squared_error",
    cv=kf,
    random_state=SEED,
)
study = optuna.create_study(sampler=objective.sampler, direction="maximize")
study.optimize(objective, n_trials=100)

# %%
best_estimator = objective.get_best_estimator(study).fit(X_train, y_train)
np.sqrt(mean_squared_error(y_test, best_estimator.predict(X_test)))

# %% [markdown]
# ```EnsembleRegressor```は各モデルを```OptunaSearchCV```で調整してからまとめる．
# モデルが1つで```method="average"```なら，調整したモデルの予測をそのまま使う．
# 特徴量を選ぶときは，```Pipeline```の最初に```BorutaPy```を入れたモデルを渡す．

# %%
er = EnsembleRegressor(
    [lgbm],
    method="average",
    cv=kf,
    n_jobs=-1,
    random_state=SEED,
    scoring="neg_root_mean_squared_error",
    n_trials=100,
)

# %%
er.fit(X_train, y_train)

# %%
np.sqrt(mean_squared_error(y_test, er.predict(X_test)))

# %%
er.estimators_[0].best_params_
