import math

import optuna
from optuna.integration import OptunaSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

from yikit.models import Objective, ParamDistributions


def test_create_study(X_regression, y_regression):
    x = X_regression
    y = y_regression

    estimator = RandomForestRegressor(random_state=334)
    objective = Objective(
        estimator,
        x,
        y,
        scoring="neg_mean_absolute_error",
        cv=KFold(n_splits=5, shuffle=True, random_state=334),
        random_state=334,
    )
    study = optuna.create_study(
        direction="maximize", sampler=objective.sampler
    )
    study.optimize(objective, n_trials=10)

    assert math.isclose(study.best_value, -65.6, abs_tol=0.1)
    assert study.best_trial.number == 3


def test_optuna_search_cv(X_regression, y_regression):
    x = X_regression
    y = y_regression

    param_distributions = ParamDistributions(
        RandomForestRegressor(random_state=334), random_state=334
    )

    estimator = RandomForestRegressor(random_state=334)
    ocv = OptunaSearchCV(
        estimator,
        param_distributions=param_distributions,
        scoring="neg_mean_absolute_error",
        cv=KFold(n_splits=5, shuffle=True, random_state=334),
        random_state=334,
    )
    ocv.fit(x, y)
    assert math.isclose(ocv.best_score_, -65.6, abs_tol=0.1)
    assert ocv.study_.best_trial.number == 3
