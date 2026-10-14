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
import optuna
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from yikit.models import Objective
from yikit.visualize import get_learning_curve_optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

# %%
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)
display(X, y)

# %%
SEED = 334

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=SEED, test_size=0.2, stratify=y
)

# %% [markdown]
# ベンチマーク

# %%
estimator = RidgeClassifier(random_state=SEED)
accuracy_score(y_test, estimator.fit(X_train, y_train).predict(X_test))


# %% [markdown]
# ハイパーパラメータ探索（探索範囲を自分で決める）

# %%
def custom_params(trial):
    return {
        "alpha": trial.suggest_float("alpha", 1e-2, 1e2, log=True),
    }


# %%
kf = StratifiedKFold(n_splits=4, shuffle=False)
objective = Objective(
    estimator,
    X_train,
    y_train,
    custom_params,
    cv=kf,
    random_state=SEED,
    scoring="accuracy",
    n_jobs=-1,
)

# %%
study = optuna.create_study(sampler=objective.sampler, direction="maximize")
study.optimize(objective, n_trials=50, n_jobs=1)

# %%
get_learning_curve_optuna(study=study)

# %%
estimator_opt = objective.get_best_estimator(study)
accuracy_score(y_test, estimator_opt.fit(X_train, y_train).predict(X_test))

# %%
estimator_opt, estimator
