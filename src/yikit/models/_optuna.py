import numpy as np
import optuna
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import check_scoring
from sklearn.model_selection import check_cv
from sklearn.model_selection._validation import _fit_and_score
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import check_random_state, check_X_y

from yikit.helpers import is_installed
from yikit.models._linear import LinearModelRegressor
from yikit.models._svm import SupportVectorRegressor

if is_installed("lightgbm"):
    from lightgbm import LGBMRegressor

    from yikit.models._gbdt import GBDTRegressor

else:
    LGBMRegressor = None  # type: ignore[assignment,misc]
    GBDTRegressor = None  # type: ignore[assignment,misc]


if is_installed("keras"):
    from yikit.models._mlp import NNRegressor
else:
    NNRegressor = None  # type: ignore[assignment,misc]

if is_installed("ngboost"):
    from ngboost import NGBRegressor
else:
    NGBRegressor = None  # type: ignore[assignment,misc]


class Objective:
    def __init__(
        self,
        estimator,
        X,
        y,
        custom_params=lambda trial: {},
        fixed_params={},
        cv=5,
        random_state=None,
        scoring=None,
        n_jobs=None,
    ):
        """objective function of optuna.

        Parameters
        ----------
        estimator : sklearn-based estimator instance
            e.g. sklearn.ensemble.RandomForestRegressor()
        X : 2-d array
            features
        y : 1-d array
            target
        custom_params : func, optional
            If you want to do your own custom range of optimization, you can define it here with a function that returns a dictionary., by default lambda trial:{}
        fixed_params : dict, optional
            If you have a fixed variable, you can specify it in the dictionary., by default {}
        cv : int, KFold object, optional
            How to cross-validate, by default 5
        random_state : int or RandomState object, optional
            seed, by default None
        scoring : scorer object, str, etc., optional
            If you don't specify it, it will use the default evaluation function of sklearn., by default None
        n_jobs : int, optional
            parallel processing, by default None
        """
        self.estimator = estimator
        self.X, self.y = check_X_y(X, y)
        self.custom_params = custom_params
        self._fixed_params = fixed_params
        self.cv = check_cv(cv)
        self.rng = check_random_state(random_state)
        self.scoring = check_scoring(estimator, scoring)
        self.n_jobs = n_jobs

        # sampler
        self.sampler = optuna.samplers.TPESampler(
            seed=self.rng.randint(2**31 - 1)
        )

    def __call__(self, trial: optuna.trial.Trial):
        if isinstance(self.estimator, NNRegressor):
            params_ = {
                "input_dropout": trial.suggest_float(
                    "input_dropout", 0.0, 0.3
                ),
                "hidden_layers": trial.suggest_int("hidden_layers", 2, 4),
                "hidden_units": trial.suggest_int(
                    "hidden_units", 32, 1024, 32
                ),
                "hidden_activation": trial.suggest_categorical(
                    "hidden_activation", ["prelu", "relu"]
                ),
                "hidden_dropout": trial.suggest_float(
                    "hidden_dropout", 0.2, 0.5
                ),
                "batch_norm": trial.suggest_categorical(
                    "batch_norm", ["before_act", "no"]
                ),
                "optimizer_type": trial.suggest_categorical(
                    "optimizer_type", ["adam", "sgd"]
                ),
                "lr": trial.suggest_loguniform("lr", 0.00001, 0.01),
                "batch_size": trial.suggest_int("hidden_units", 32, 1024, 32),
                "l": trial.suggest_loguniform("l", 1e-7, 0.1),
            }
            self.fixed_params_ = {
                "progress_bar": False,
                "random_state": self.rng,
            }
        elif isinstance(self.estimator, (GBDTRegressor, LGBMRegressor)):
            params_ = {
                "n_estimators": trial.suggest_int(
                    "n_estimators", 10, 1000, log=True
                ),
                # 'max_depth' : trial.suggest_int('n_estimators', 3, 9),    # num_leaves変えた方が良さそう．制約条件的に．
                "min_child_weight": trial.suggest_loguniform(
                    "min_child_weight", 0.001, 10
                ),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", 0.6, 0.95
                ),
                "subsample": trial.suggest_float("subsample", 0.6, 0.95),
                "num_leaves": trial.suggest_int(
                    "num_leaves", 2**3, 2**9, log=True
                ),
            }
            self.fixed_params_ = {
                "random_state": self.rng,
                "n_jobs": -1,
                "objective": "regression",
            }
        elif isinstance(self.estimator, RandomForestRegressor):
            # 最適化するべきパラメータ
            params_ = {
                "min_samples_split": trial.suggest_int(
                    "min_samples_split", 2, 16
                ),
                "max_depth": trial.suggest_int("max_depth", 10, 100),
                "n_estimators": trial.suggest_int(
                    "n_estimators", 10, 1000, log=True
                ),
            }
            # 固定するパラメータ (外でも取り出せるようにインスタンス変数としてる．)
            self.fixed_params_ = {
                "random_state": self.rng,
                "n_jobs": -1,
            }
        elif isinstance(self.estimator, (SupportVectorRegressor, SVR)):
            # 最適化するべきパラメータ
            params_ = {
                "C": trial.suggest_loguniform("C", 2**-5, 2**10),
                "epsilon": trial.suggest_loguniform("epsilon", 2**-10, 2**0),
            }
            # 固定するパラメータ (外でも取り出せるようにインスタンス変数としてる．)
            self.fixed_params_ = {"gamma": "auto", "kernel": "rbf"}
        elif isinstance(self.estimator, LinearModelRegressor):
            # 最適化するべきパラメータ
            params_ = {
                "linear_model": trial.suggest_categorical(
                    "linear_model", ["ridge", "lasso"]
                ),
                "alpha": trial.suggest_loguniform("alpha", 0.1, 10),
                "fit_intercept": trial.suggest_categorical(
                    "fit_intercept", [True, False]
                ),
                "max_iter": trial.suggest_loguniform("max_iter", 100, 10000),
                "tol": trial.suggest_loguniform("tol", 0.0001, 0.01),
            }
            # 固定するパラメータ (外でも取り出せるようにインスタンス変数としてる．)
            self.fixed_params_ = {
                "random_state": self.rng,
            }
        elif isinstance(self.estimator, MLPRegressor):
            # 最適化するべきパラメータ
            params_ = {
                "hidden_layer_sizes": trial.suggest_int(
                    "hidden_layer_sizes", 50, 300
                ),
                "alpha": trial.suggest_loguniform("alpha", 1e-5, 1e-3),
                "learning_rate_init": trial.suggest_loguniform(
                    "learning_rate_init", 1e-5, 1e-3
                ),
            }
            # 固定するパラメータ (外でも取り出せるようにインスタンス変数としてる．)
            self.fixed_params_ = {
                "random_state": self.rng,
            }
        elif is_installed("ngboost") and isinstance(
            self.estimator, NGBRegressor
        ):
            # 最適化するべきパラメータ
            params_ = {
                "Base": DecisionTreeRegressor(
                    max_depth=trial.suggest_int("Base__max_depth", 2, 100),
                    criterion=trial.suggest_categorical(
                        "Base__criterion", ["squared_error", "friedman_mse"]
                    ),  # FutureWarning: Criterion 'mse' was deprecated in v1.0 and will be removed in version 1.2. Use `criterion='squared_error'` which is equivalent.
                    random_state=self.rng,
                ),
                "n_estimators": trial.suggest_int(
                    "n_estimators", 10, 1000, log=True
                ),
                "minibatch_frac": trial.suggest_float(
                    "minibatch_frac", 0.5, 1.0
                ),
            }
            # 固定するパラメータ (外でも取り出せるようにインスタンス変数としてる．)
            self.fixed_params_ = {
                "random_state": self.rng,
            }
        elif self.custom_params(trial):
            params_ = self.custom_params(trial)
            self.fixed_params_ = {}  # あとで加えるので空でOK．
        else:
            raise NotImplementedError("{0}".format(self.estimator))

        # もしfixed_paramsを追加で指定されたらそれを取り入れる
        self.fixed_params_.update(self._fixed_params)

        self.model = type(self.estimator)
        # self.estimator_ = self.model(**params_, **self.fixed_params_)
        self.estimator_ = clone(self.estimator)
        self.estimator_.set_params(
            **params_,
            **self.fixed_params_,
        )

        parallel = Parallel(n_jobs=self.n_jobs)
        # support old version of scikit-learn (<1.4)
        results = parallel(
            delayed(_fit_and_score)(
                clone(self.estimator_),
                self.X,
                self.y,
                self.scoring,
                train,
                test,
                0,
                dict(**self.fixed_params_, **params_),
                None,
            )
            for train, test in self.cv.split(self.X, self.y)
        )
        return np.mean(
            [d["test_scores"] for d in results]
        )  # scikit-learn>=0.24.1

    def get_best_estimator(self, study):
        best_params_ = self.get_best_params(study)
        return self.model(**best_params_)

    def get_best_params(self, study):
        if isinstance(self.estimator_, NGBRegressor):
            dt_best_params_ = {}
            best_params_ = {}
            key_base = "Base__"
            for k, v in study.best_params.items():
                if key_base in k:
                    dt_best_params_[k[len(key_base) :]] = v
                else:
                    best_params_[k] = v
            else:
                if "random_state" in self.fixed_params_:
                    dt_best_params_["random_state"] = self.fixed_params_[
                        "random_state"
                    ]
                best_params_["Base"] = DecisionTreeRegressor(**dt_best_params_)
        else:
            best_params_ = study.best_params
        best_params_.update(**self.fixed_params_)
        return best_params_
