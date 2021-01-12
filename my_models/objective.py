from my_models import NNRegressor, GBDTRegressor, SupportVectorRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.utils import check_X_y
import optuna


class _Metric:
    def __init__(self, metric = 'rmse'):
        if metric == 'r2':
            self.metric_ = r2_score
            self.direction_ = 'maximize'
            self.name_ = 'R^2'
        else:
            self.direction_ = 'minimize'
            if metric == 'rmse':
                self.metric_ = lambda x1, x2: mean_squared_error(x1, x2, squared=False)
                self.name_ = 'RMSE'
            elif metric == 'mse':
                self.metric_ = mean_squared_error
                self.name_ = 'MSE'
            elif metric == 'mae':
                self.metric_ = mean_absolute_error
                self.name_ = 'MAE'
            else:
                raise NotImplementedError('{0}'.format(metric))


class Objective:
    def __init__(self, estimator, X, y, fixed_params = {}, cv = 5, random_state = None, metric = 'rmse'):
        self.estimator = estimator
        self.X, self.y = check_X_y(X, y)
        self.fixed_params = fixed_params
        self.cv = cv
        self.random_state = random_state

        # metric関係
        self._metric = _Metric(metric = metric)
        self.direction = self._metric.direction_

        # sampler
        self.sampler = optuna.samplers.TPESampler(seed = self.random_state)

        # このままcreate_studyに渡せる．
        self.params_create_study = {
            'direction': self.direction,
            'sampler': self.sampler
        }

    def __call__(self, trial):
        if isinstance(self.estimator, NNRegressor):
            params_ = {
                'input_dropout': trial.suggest_uniform('input_dropout', 0.0, 0.3),
                'hidden_layers': trial.suggest_int('hidden_layers', 2, 4),
                'hidden_units' : trial.suggest_int('hidden_units', 32, 1024, 32),
                'hidden_activation' : trial.suggest_categorical('hidden_activation', ['prelu', 'relu']),
                'hidden_dropout' : trial.suggest_uniform('hidden_dropout', 0.2, 0.5),
                'batch_norm' : trial.suggest_categorical('batch_norm', ['before_act', 'no']),
                'optimizer_type' : trial.suggest_categorical('optimizer_type', ['adam', 'sgd']),
                'lr' : trial.suggest_loguniform('lr', 0.00001, 0.01),
                'batch_size' : trial.suggest_int('hidden_units', 32, 1024, 32),
                'l' : trial.suggest_loguniform('l', 1E-7, 0.1),
            }
            self.fixed_params_ = {
                'progress_bar': False,
                'random_state': self.random_state,
            }
        elif isinstance(self.estimator, (GBDTRegressor, LGBMRegressor)):
            params_ = {
                'n_estimators' : trial.suggest_int('n_estimators', 10, 500),
                # 'max_depth' : trial.suggest_int('n_estimators', 3, 9),    # num_leaves変えた方が良さそう．制約条件的に．
                'min_child_weight' : trial.suggest_loguniform('min_child_weight', 0.001, 10),
                'colsample_bytree' : trial.suggest_uniform('colsample_bytree', 0.6, 0.95),
                'subsample': trial.suggest_uniform('subsample', 0.6, 0.95),
                'num_leaves' : trial.suggest_int('num_leaves', 2 ** 3, 2 ** 9, log = True)
            }
            self.fixed_params_ = {
                'random_state' : self.random_state,
                'n_jobs' : -1,
                'objective' : 'regression',
            }
        elif isinstance(self.estimator, RandomForestRegressor):
            # 最適化するべきパラメータ
            params_ = {
                'min_samples_split' : trial.suggest_int('min_samples_split', 2, 16),
                'max_depth' : trial.suggest_int('max_depth', 10, 100),
                'n_estimators' : trial.suggest_int('n_estimators', 10, 85)  # 500にしてたが，pickleファイルが90以上だと10MB超えちゃうため．
            }
            # 固定するパラメータ (外でも取り出せるようにインスタンス変数としてる．)
            self.fixed_params_ = {
                'random_state' : self.random_state,
                'n_jobs' : -1,
            }
        elif isinstance(self.estimator, (SupportVectorRegressor, SVR)):
            params_ = {
                'C' : trial.suggest_loguniform('C', 2 ** -5, 2 ** 10),
                'epsilon' : trial.suggest_loguniform('epsilon', 2 ** -10, 2 ** 0),
            }
            # 固定するパラメータ (外でも取り出せるようにインスタンス変数としてる．)
            self.fixed_params_ = {
                'gamma' : 'auto',
                'kernel' : 'rbf'
            }
        else:
            raise NotImplementedError('{0}'.format(self.estimator))

        # もしfixed_paramsを追加で指定されたらそれを取り入れる
        self.fixed_params_.update(self.fixed_params)

        self.model = type(self.estimator)
        self.estimator_ = self.model(**params_, **self.fixed_params_)

        if self.cv < 2:
            raise ValueError('"cv" must be an integer greater than 2.')
        y_pred_oof_ = cross_val_predict(self.estimator_, self.X, self.y)

        return self._metric.metric_(y_pred_oof_, self.y)


        
