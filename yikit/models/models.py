
'''
Copyright (c) 2021 yu9824

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

from yikit.tools.tools import is_notebook
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.base import is_classifier, is_regressor
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array, check_X_y, check_random_state
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import Bunch
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.model_selection import check_cv
from sklearn.model_selection._validation import _fit_and_score, _aggregate_score_dicts, _score
from sklearn.metrics import check_scoring
from sklearn.metrics._scorer import _check_multimetric_scoring
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.base import clone
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
from sklearn.inspection import permutation_importance
from lightgbm import LGBMRegressor
import sys

from yikit.tools import is_notebook
if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

from ..feature_selection.wrapper_method import WrapperSelector

import optuna
from joblib import Parallel, delayed

import numpy as np
import pandas as pd

__all__ = [
    'NNRegressor',
    'GBDTRegressor',
    'LinearModelRegressor',
    'SupportVectorRegressor',
    'EnsembleRegressor',
    'Objective'
]


class NNRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, input_dropout = 0.0, hidden_layers = 3, hidden_units = 96, hidden_activation = 'relu', hidden_dropout = 0.2, batch_norm = 'before_act', optimizer_type = 'adam', lr = 0.001, batch_size = 64, l = 0.01, random_state = None, epochs = 200, patience = 20, progress_bar = True, scale = True):
        try:
            import keras
        except ModuleNotFoundError as e:
            sys.stdout.write(e)
            raise ModuleNotFoundError('If you want to use this module, please install keras.')

        self.input_dropout = input_dropout
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.hidden_activation = hidden_activation
        self.hidden_dropout = hidden_dropout
        self.batch_norm = batch_norm
        self.optimizer_type = optimizer_type
        self.lr = lr
        self.batch_size = batch_size
        self.l = l
        self.random_state = random_state
        self.epochs = epochs
        self.patience = patience
        self.progress_bar = progress_bar
        self.scale = scale

    def fit(self, X, y):
        from keras.callbacks import EarlyStopping
        from keras.layers.advanced_activations import ReLU, PReLU
        from keras.layers import Dense, Dropout
        from keras.layers.normalization import BatchNormalization
        from keras.models import Sequential
        from keras.optimizers import SGD, Adam
        from keras.regularizers import l2
        from keras.backend import clear_session
        
        # 入力されたXとyが良い感じか判定（サイズが適切かetc)
        X, y = check_X_y(X, y)

        '''
        sklearn/utils/estimator_checks.py:3063:
        FutureWarning: As of scikit-learn 0.23, estimators should expose a n_features_in_ attribute, 
        unless the 'no_validation' tag is True.
        This attribute should be equal to the number of features passed to the fit method.
        An error will be raised from version 1.0 (renaming of 0.25) when calling check_estimator().
        See SLEP010: https://scikit-learn-enhancement-proposals.readthedocs.io/en/latest/slep010/proposal.html
        '''
        self.n_features_in_ = X.shape[1]    # check_X_yのあとでないとエラーになりうる．

        # check_random_state
        self.rng_ = check_random_state(self.random_state)

        # 標準化
        if self.scale:
            self.scaler_X_ = StandardScaler()
            X_ = self.scaler_X_.fit_transform(X)

            self.scaler_y_ = StandardScaler()
            y_ = self.scaler_y_.fit_transform(np.array(y).reshape(-1, 1)).flatten()
        else:
            X_ = X
            y_ = y
        

        # なぜか本当によくわかんないけど名前が一緒だと怒られるのでそれをずらすためのやつをつくる
        j = int(10E5)

        # プログレスバー
        if self.progress_bar:
            pbar = tqdm(total = 1 + self.hidden_layers + 6)
        self.estimator_ = Sequential()

        # 入力層
        self.estimator_.add(Dropout(self.input_dropout, input_shape = (self.n_features_in_,), seed = self.rng_.randint(2 ** 32), name = 'Dropout_' + str(j)))
        if self.progress_bar:
            pbar.update(1)


        # 中間層
        for i in range(self.hidden_layers):
            self.estimator_.add(Dense(units = self.hidden_units, kernel_regularizer = l2(l = self.l), name = 'Dense_' + str(j+i+1)))  # kernel_regularizer: 過学習対策
            if self.batch_norm == 'before_act':
                self.estimator_.add(BatchNormalization(name = 'BatchNormalization' + str(j+i+1)))
            
            # 活性化関数
            if self.hidden_activation == 'relu':
                self.estimator_.add(ReLU(name = 'Re_Lu_' + str(j+i+1)))
            elif self.hidden_activation == 'prelu':
                self.estimator_.add(PReLU(name = 'PRe_Lu_' + str(j+i+1)))
            else:
                raise NotImplementedError

            self.estimator_.add(Dropout(self.hidden_dropout, seed = self.rng_.randint(2 ** 32), name = 'Dropout_' + str(j+i+1)))

            # プログレスバー
            if self.progress_bar:
                pbar.update(1)
        
        # 出力層
        self.estimator_.add(Dense(1, activation = 'linear', name = 'Dense_' + str(j+self.hidden_layers+1)))

        # optimizer
        if self.optimizer_type == 'adam':
            optimizer_ = Adam(lr = self.lr, beta_1 = 0.9, beta_2 = 0.999, decay = 0.0)
        elif self.optimizer_type == 'sgd':
            optimizer_ = SGD(lr = self.lr, decay = 1E-6, momentum = 0.9, nesterov = True)
        else:
            raise NotImplementedError
        
        # 目的関数，評価指標などの設定
        self.estimator_.compile(loss = 'mean_squared_error', optimizer = optimizer_, metrics=['mse', 'mae'])

        # プログレスバー
        if self.progress_bar:
            pbar.update(1)

        # 変数の定義
        self.early_stopping_ = EarlyStopping(patience = self.patience, restore_best_weights = True)
        self.validation_split_ = 0.2

        # fit
        self.estimator_.fit(X_, y_, validation_split = self.validation_split_, epochs = self.epochs, batch_size = self.batch_size, callbacks = [self.early_stopping_], verbose = 0)

        # プログレスバー
        if self.progress_bar:
            pbar.update(5)
            pbar.close()

        return self

    def predict(self, X):
        # fitが行われたかどうかをインスタンス変数が定義されているかで判定（第二引数を文字列ではなくてリストで与えることでより厳密に判定可能）
        check_is_fitted(self, 'estimator_')

        # 入力されたXが妥当か判定
        X = check_array(X)

        if self.scale:
            X_ = self.scaler_X_.transform(X)
            y_pred_ = self.estimator_.predict(X_)
            y_pred_ = self.scaler_y_.inverse_transform(y_pred_).flatten()
        else:
            X_ = X
            y_pred_ = self.estimator_.predict(X_).flatten()
        return y_pred_
    


class GBDTRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, boosting_type='gbdt', num_leaves=31, max_depth=-1, learning_rate=0.1, n_estimators=100, subsample_for_bin=200000, objective=None, class_weight=None, min_split_gain=0.0, min_child_weight=0.001, min_child_samples=20, subsample=1.0, subsample_freq=0, colsample_bytree=1.0, reg_alpha=0.0, reg_lambda=0.0, random_state=None, n_jobs=-1, silent=True, importance_type='split', **kwargs):
        # self.hoge = hogeとしなければいけない．つまりself.fuga = hogeだと怒られる
        self.boosting_type = boosting_type
        self.num_leaves = num_leaves
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.n_estimators = n_estimators
        self.subsample_for_bin = subsample_for_bin
        self.objective = objective
        self.class_weight = class_weight
        self.min_split_gain = min_split_gain
        self.min_child_weight = min_child_weight
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.subsample_freq = subsample_freq
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.silent = silent
        self.importance_type = importance_type
        if len(kwargs):
            self.kwargs = kwargs

    def fit(self, X, y):
        try:
            kwargs = self.kwargs
        except:
            kwargs = {}

        # check_random_state
        self.rng_ = check_random_state(self.random_state)

        # fitしたあとに確定する値は変数名 + '_' としなければならない．
        self.estimator_ = LGBMRegressor(boosting_type = self.boosting_type,
            num_leaves = self.num_leaves,
            max_depth = self.max_depth,
            learning_rate = self.learning_rate,
            n_estimators = self.n_estimators,
            subsample_for_bin = self.subsample_for_bin,
            objective = self.objective,
            class_weight = self.class_weight,
            min_split_gain = self.min_split_gain,
            min_child_weight = self.min_child_weight,
            min_child_samples = self.min_child_samples,
            subsample = self.subsample,
            subsample_freq = self.subsample_freq,
            colsample_bytree = self.colsample_bytree,
            reg_alpha = self.reg_alpha,
            reg_lambda = self.reg_lambda,
            random_state = self.rng_,
            n_jobs = self.n_jobs,
            silent = self.silent,
            importance_type = self.importance_type,
            **kwargs
        )

        # 入力されたXとyが良い感じか判定（サイズが適切かetc)
        X, y = check_X_y(X, y)

        '''
        sklearn/utils/estimator_checks.py:3063:
        FutureWarning: As of scikit-learn 0.23, estimators should expose a n_features_in_ attribute, 
        unless the 'no_validation' tag is True.
        This attribute should be equal to the number of features passed to the fit method.
        An error will be raised from version 1.0 (renaming of 0.25) when calling check_estimator().
        See SLEP010: https://scikit-learn-enhancement-proposals.readthedocs.io/en/latest/slep010/proposal.html
        '''
        self.n_features_in_ = X.shape[1]    # check_X_yのあとでないとエラーになりうる．

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = self.rng_, test_size = 0.2)

        self.estimator_.fit(X, y, eval_set = [(X_test, y_test)], eval_metric = ['mse', 'mae'], early_stopping_rounds = 20, verbose = False)
        self.feature_importances_ = self.estimator_.feature_importances_

        # 慣例と聞いたはずなのにこれをreturnしないと怒られる．審査が厳しい．
        return self

    def predict(self, X):
        # fitが行われたかどうかをインスタンス変数が定義されているかで判定（第二引数を文字列ではなくてリストで与えることでより厳密に判定可能）
        check_is_fitted(self, 'estimator_')

        # 入力されたXが妥当か判定
        X = check_array(X)

        # 予測結果を返す
        return self.estimator_.predict(X)


# スケールの概念が入っていないので，それらを内包したscikit-learn準拠モデルを自分で定義する必要がある．
class EnsembleRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, estimators = (RandomForestRegressor(),), method = 'blending', cv = 5, n_jobs = -1, random_state = None, scoring = 'neg_mean_squared_error', verbose = 0, boruta = True, opt = True):
        '''
        Parameters
        ----------
        estimators: 1-d list, default = (RandomForestRegressor(), )
            List of estimators to ensemble.

        method: {'blending', 'average', 'stacking'}, default = 'blending'
            How to ensemble.

        cv: int or callable, default = 5

        n_jobs: int, default = -1

        random_state: None, int or callable, default = None

        scoring: str, callable or list, default = 'neg_mean_squared_error'
            https://scikit-learn.org/stable/modules/model_evaluation.html

        verbose: int, default = 0

        boruta: bool, default = True
            Do boruta or not.

        opt: bool, default = True
            Do hyperparameter optimization or not.
        '''
        self.estimators = estimators
        self.method = method
        self.cv = cv
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.scoring = scoring
        self.verbose = verbose
        self.boruta = boruta
        self.opt = opt

    def fit(self, X, y):
        # よく使うので変数化
        self.n_estimators_ = len(self.estimators)

        # check_X_y
        X, y = check_X_y(X, y)

        '''
        sklearn/utils/estimator_checks.py:3063:
        FutureWarning: As of scikit-learn 0.23, estimators should expose a n_features_in_ attribute, 
        unless the 'no_validation' tag is True.
        This attribute should be equal to the number of features passed to the fit method.
        An error will be raised from version 1.0 (renaming of 0.25) when calling check_estimator().
        See SLEP010: https://scikit-learn-enhancement-proposals.readthedocs.io/en/latest/slep010/proposal.html
        '''
        self.n_features_in_ = X.shape[1]    # check_X_yのあとでないとエラーになりうる．
        
        # check_random_state
        rng_ = check_random_state(self.random_state)
        
        # isRegressor
        if sum([is_regressor(estimator) for estimator in self.estimators]) != self.n_estimators_:
            raise ValueError

        # check_cv
        cv_ = check_cv(self.cv, y = y, classifier = False)

        # check_scoring
        estimator = self.estimators[0]
        if callable(self.scoring):
            scorers = self.scoring
        elif self.scoring is None or isinstance(self.scoring, str):
            scorers = check_scoring(estimator = estimator)
        else:
            # 0.24.1のコードだと辞書を返すことになっているが，0.23.2ではtupleが返ってきてしまう？
            scorers = _check_multimetric_scoring(estimator, self.scoring)
            if isinstance(scorers, tuple):
                scorers = scorers[0]

        # 並列処理する部分を関数化
        def _f(i_train, i_test):
            X_train, X_test, y_train, y_test = X[i_train], X[i_test], y[i_train], y[i_test]

            if self.boruta:
                # 特徴量削減
                feature_selector_ = WrapperSelector(estimator = RandomForestRegressor(n_jobs = -1), random_state = rng_, max_iter = 300, verbose = self.verbose)
                feature_selector_.fit(X_train, y_train)
                
                # 抽出
                support_ = feature_selector_.get_support()
                X_train_selected = feature_selector_.transform(X_train)
                X_test_selected = feature_selector_.transform(X_test)
            else:   # borutaしない場合でもresultsに組み込まれるので変数を定義しておく．
                support_ = np.ones(X_train.shape[1], dtype = np.bool)
                X_train_selected = X_train
                X_test_selected = X_test

            # verbose
            if self.verbose == 0:
                optuna.logging.disable_default_handler()

            results_estimators = []
            for estimator in self.estimators:
                if self.opt:
                    # ハイパーパラメータ（scoringで最初にしていしたやつで最適化）
                    objective = Objective(estimator, X_train_selected, y_train, cv = cv_, random_state = rng_, scoring = scorers.values()[0] if isinstance(scorers, dict) else scorers)
                    sampler = optuna.samplers.TPESampler(seed = rng_.randint(2 ** 32))

                    study = optuna.create_study(sampler = sampler, direction = 'maximize')
                    study.optimize(objective, n_trials = 100, n_jobs = 1)

                    # 最適化後のモデル
                    _best_estimator_ = objective.model(**objective.fixed_params_, **study.best_params)
                else:   # optunaしない場合でもresultsに組み込まれるので変数を定義しておく．
                    study = None
                    _best_estimator_ = clone(estimator)

                # fit
                _best_estimator_.fit(X_train_selected, y_train)

                # predict
                _y_pred_on_train = _best_estimator_.predict(X_train_selected)
                _y_pred_on_test = _best_estimator_.predict(X_test_selected)

                # score
                _train_scores = _score(_best_estimator_, X_train_selected, y_train, scorers)
                _test_scores = _score(_best_estimator_, X_test_selected, y_test, scorers)

                # importances
                _gi = _best_estimator_.feature_importances_ if 'feature_importances_' in dir(_best_estimator_) else None
                _pi = permutation_importance(_best_estimator_, X_test_selected, y_test, scoring = 'neg_mean_squared_error', n_repeats = 10, n_jobs = -1, random_state = rng_).importances

                # 予測結果をDataFrameにまとめる．
                _y_train = pd.DataFrame(np.hstack([y_train.reshape(-1, 1), _y_pred_on_train.reshape(-1, 1)]), columns = ['true', 'pred'], index = i_train)
                _y_test = pd.DataFrame(np.hstack([y_test.reshape(-1, 1), _y_pred_on_test.reshape(-1, 1)]), columns = ['true', 'pred'], index = i_test)

                results_estimators.append({
                        'estimators': _best_estimator_,
                        'params': _best_estimator_.get_params(),
                        'y_train': _y_train,
                        'y_test': _y_test,
                        'train_scores': _train_scores,
                        'test_scores': _test_scores,
                        'gini_importances': _gi,
                        'permutation_importances': _pi,
                        'studies': study,
                    })
            # verbose
            if self.verbose == 0:
                optuna.logging.disable_default_handler()

            # 出力結果をいい感じにする．←ここから
            ret = {}
            temp = {}
            for result_estimator in results_estimators: # それぞれのestimatorについて
                for k, v in result_estimator.items():   # その中の各々の値について
                    # スコア系かつそれらが複数指定されている場合だけ特別処理
                    if '_score' in k and isinstance(v, dict):
                        if k not in temp:
                            temp[k] = {}
                        for score_name, score in v.items():
                            if score_name in temp[k]:
                                temp[k][score_name].append(score)
                            else:
                                temp[k][score_name] = [score]
                    else:   # スコア系以外
                        if k in ret:
                            ret[k].append(v)
                        else:
                            ret[k] = [v]
            # scoreをためてるやつをBunch型に変換
            for k in temp:
                temp[k] = Bunch(**temp[k])
            
            # 返すように最終整形
            ret['support_'] = support_
            ret.update(temp)
            return ret
        
        # 上記で定義した_f関数どうしは互いに独立なので並列で処理する．
        parallel = Parallel(n_jobs = self.n_jobs, verbose = self.verbose)
        results = parallel(delayed(_f)(i_train, i_test) for i_train, i_test in cv_.split(X, y))
        # results = [_f(i_train, i_test) for i_train, i_test in cv_.split(X, y)]    # デバッグ用．並列しないでやる方法

        # データを整形
        self.results_ = {}
        for result in results:
            for k, v in result.items():
                if k in self.results_:
                    self.results_[k].append(v)
                else:
                    self.results_[k] = [v]

        # 扱いやすいようにBunch型に変換
        self.results_ = Bunch(**self.results_)

        # OOFの予測結果を取得
        dfs_y_oof_ = [pd.concat([lst[n] for lst in self.results_['y_test']], axis = 0).sort_index() for n in range(self.n_estimators_)]
        y_oof_ = pd.concat([df.loc[:, 'pred'] for df in dfs_y_oof_], axis = 1)
        y_oof_.columns = ['estimator{}'.format(n) for n in range(self.n_estimators_)]

        # *** ensemble ***
        # モデルがひとつのとき．
        if self.method == 'average' or self.n_estimators_ == 1:
            self.weights_ = None
        elif self.method == 'blending':
            # rmseで最適化（今後指定できるようにしてもいいかも．）
            def objective(trial):
                params = {'weight{0}'.format(i): trial.suggest_uniform('weight{0}'.format(i), 0, 1) for i in range(self.n_estimators_)}
                weights = np.array(list(params.values()))
                y_oof_ave = np.average(y_oof_, weights = weights, axis = 1)
                return mean_squared_error(y_oof_ave, y, squared = False)

            # optunaのログを非表示
            if self.verbose == 0:
                optuna.logging.disable_default_handler()

            # 重みの最適化
            sampler_ = optuna.samplers.TPESampler(seed = rng_.randint(2 ** 32))
            study = optuna.create_study(sampler = sampler_, direction = 'minimize') # 普通のRMSEなので．
            study.optimize(objective, n_trials = 100, n_jobs = 1)   # -1にするとなぜかバグるので．（そもそもそんなに重くないので1で．）

            # optunaのログを再表示
            if self.verbose == 0:
                optuna.logging.enable_default_handler()

            self.weights_ = np.array(list(study.best_params.values()), dtype = np.float64)
            self.weights_ /= np.sum(self.weights_)
        elif self.method == 'stacking':
            # 線形モデルの定義
            self.stacking_model_ = LinearRegression(n_jobs = self.n_jobs)
            self.stacking_model_.fit(y_oof_.values, y)
            # resultsに保存するために定義だけする．
            self.weights_ = None
        else:
            raise NotImplementedError

        # 重みを結果に保存
        self.results_['weights'] = self.weights_

        return self

    def predict(self, X):
        # fitが行われたかどうかをインスタンス変数が定義されているかで判定（第二引数を文字列ではなくてリストで与えることでより厳密に判定可能）
        check_is_fitted(self, 'results_')

        # 入力されたXが妥当か判定
        X = check_array(X)

        # 各予測モデルの予測結果をまとめる．(内包リストで得られる配列は3-Dベクトル (n_estimators_, cv, n_samples))
        y_preds_ = np.average(np.array([[estimators[n].predict(X[:, self.results_.support_[m]]) for m, estimators in enumerate(self.results_.estimators)] for n in range(self.n_estimators_)]), axis = 1).transpose()   # 同じ種類のやつは単純に平均を取る．
        
        if self.method in ('blending', 'average') or self.n_estimators_ == 1:
            y_pred_ = np.average(y_preds_, weights = self.weights_, axis = 1)
        elif self.method == 'stacking':
            y_pred_ = self.stacking_model_.predict(y_preds_)
        return y_pred_


class SupportVectorRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, kernel = 'rbf', gamma = 'auto', tol = 0.01, C = 1.0, epsilon = 0.1, scale = True):
        self.kernel = kernel
        self.gamma = gamma
        self.tol = tol
        self.C = C
        self.epsilon = epsilon
        self.scale = scale

    def fit(self, X, y):
        # 入力されたXとyが良い感じか判定（サイズが適切かetc)
        X, y = check_X_y(X, y)

        '''
        sklearn/utils/estimator_checks.py:3063:
        FutureWarning: As of scikit-learn 0.23, estimators should expose a n_features_in_ attribute, 
        unless the 'no_validation' tag is True.
        This attribute should be equal to the number of features passed to the fit method.
        An error will be raised from version 1.0 (renaming of 0.25) when calling check_estimator().
        See SLEP010: https://scikit-learn-enhancement-proposals.readthedocs.io/en/latest/slep010/proposal.html
        '''
        self.n_features_in_ = X.shape[1]    # check_X_yのあとでないとエラーになりうる．

        if self.scale:
            self.scaler_X_ = StandardScaler()
            X_ = self.scaler_X_.fit_transform(X)

            self.scaler_y_ = StandardScaler()
            y_ = self.scaler_y_.fit_transform(np.array(y).reshape(-1, 1)).flatten()
        else:
            X_ = X
            y_ = y

        self.estimator_ = SVR(kernel=self.kernel, gamma=self.gamma, tol=self.tol, C=self.C, epsilon=self.epsilon)
        self.estimator_.fit(X_, y_)

        return self

    def predict(self, X):
        # fitが行われたかどうかをインスタンス変数が定義されているかで判定（第二引数を文字列ではなくてリストで与えることでより厳密に判定可能）
        check_is_fitted(self, 'estimator_')

        # 入力されたXが妥当か判定
        X = check_array(X)

        if self.scale:
            X_ = self.scaler_X_.transform(X)
        else:
            X_ = X

        y_pred_ = self.estimator_.predict(X_)
        if self.scale:
            y_pred_ = self.scaler_y_.inverse_transform(np.array(y_pred_).reshape(-1, 1)).flatten()
        
        return y_pred_

class LinearModelRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, linear_model = 'ridge', alpha = 1.0, fit_intercept = True, max_iter = 1000, tol = 0.001, random_state = None):
        self.linear_model = linear_model
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        '''
        sklearn/utils/estimator_checks.py:3063:
        FutureWarning: As of scikit-learn 0.23, estimators should expose a n_features_in_ attribute, 
        unless the 'no_validation' tag is True.
        This attribute should be equal to the number of features passed to the fit method.
        An error will be raised from version 1.0 (renaming of 0.25) when calling check_estimator().
        See SLEP010: https://scikit-learn-enhancement-proposals.readthedocs.io/en/latest/slep010/proposal.html
        '''
        self.n_features_in_ = X.shape[1]    # check_X_yのあとでないとエラーになりうる．

        self.rng_ = check_random_state(self.random_state)

        # max_iterを引数に入れてるとこの変数ないとダメ！って怒られるから．
        self.n_iter_ = 1

        if self.linear_model == 'ridge':
            model_ = Ridge
        elif self.linear_model == 'lasso':
            model_ = Lasso
        else:
            raise NotImplementedError

        self.estimator_ = model_(alpha = self.alpha, fit_intercept = self.fit_intercept, max_iter = self.max_iter, tol = self.tol, random_state = self.rng_)
        self.estimator_.fit(X, y)
        return self
    
    def predict(self, X):
        check_is_fitted(self, 'estimator_')

        X = check_array(X)

        return self.estimator_.predict(X)

class Objective:
    def __init__(self, estimator, X, y, custom_params = lambda trial: {}, fixed_params = {}, cv = 5, random_state = None, scoring = None, n_jobs = None):
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
        self.sampler = optuna.samplers.TPESampler(seed = self.rng.randint(2 ** 32))

    def __call__(self, trial):
        if self.custom_params(trial):
            params_ = self.custom_params(trial)
            self.fixed_params_ = {} # あとで加えるので空でOK．
        elif isinstance(self.estimator, NNRegressor):
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
                'random_state': self.rng,
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
                'random_state' : self.rng,
                'n_jobs' : -1,
                'objective' : 'regression',
            }
        elif isinstance(self.estimator, RandomForestRegressor):
            # 最適化するべきパラメータ
            params_ = {
                'min_samples_split' : trial.suggest_int('min_samples_split', 2, 16),
                'max_depth' : trial.suggest_int('max_depth', 10, 100),
                'n_estimators' : trial.suggest_int('n_estimators', 10, 500)
            }
            # 固定するパラメータ (外でも取り出せるようにインスタンス変数としてる．)
            self.fixed_params_ = {
                'random_state' : self.rng,
                'n_jobs' : -1,
            }
        elif isinstance(self.estimator, (SupportVectorRegressor, SVR)):
            # 最適化するべきパラメータ
            params_ = {
                'C' : trial.suggest_loguniform('C', 2 ** -5, 2 ** 10),
                'epsilon' : trial.suggest_loguniform('epsilon', 2 ** -10, 2 ** 0),
            }
            # 固定するパラメータ (外でも取り出せるようにインスタンス変数としてる．)
            self.fixed_params_ = {
                'gamma' : 'auto',
                'kernel' : 'rbf'
            }
        elif isinstance(self.estimator, LinearModelRegressor):
            # 最適化するべきパラメータ
            params_ = {
                'linear_model' : trial.suggest_categorical('linear_model', ['ridge', 'lasso']),
                'alpha' : trial.suggest_loguniform('alpha', 0.1, 10),
                'fit_intercept' : trial.suggest_categorical('fit_intercept', [True, False]),
                'max_iter' : trial.suggest_loguniform('max_iter', 100, 10000),
                'tol' : trial.suggest_loguniform('tol', 0.0001, 0.01),
            }
            # 固定するパラメータ (外でも取り出せるようにインスタンス変数としてる．)
            self.fixed_params_ = {
                'random_state' : self.rng,
            }
        elif isinstance(self.estimator, MLPRegressor):
            # 最適化するべきパラメータ
            params_ = {
                'hidden_layer_sizes': trial.suggest_int('hidden_layer_sizes', 50, 300),
                'alpha': trial.suggest_loguniform('alpha', 1e-5, 1e-3),
                'learning_rate_init': trial.suggest_loguniform('learning_rate_init', 1e-5, 1e-3),
            }
            # 固定するパラメータ (外でも取り出せるようにインスタンス変数としてる．)
            self.fixed_params_ = {
                'random_state' : self.rng,
            }
        else:
            raise NotImplementedError('{0}'.format(self.estimator))

        # もしfixed_paramsを追加で指定されたらそれを取り入れる
        self.fixed_params_.update(self._fixed_params)

        self.model = type(self.estimator)
        # self.estimator_ = self.model(**params_, **self.fixed_params_)
        self.estimator_ = clone(self.estimator)
        self.estimator_.set_params(
            **params_,
            **self.fixed_params_,
        )

        parallel = Parallel(n_jobs = self.n_jobs)
        results = parallel(
            delayed(_fit_and_score)(
                clone(self.estimator_), self.X, self.y, self.scoring, train, test, 0, dict(**self.fixed_params_, **params_), None
            )
        for train, test in self.cv.split(self.X, self.y))
        return np.mean([d['test_scores'] for d in results])


if __name__ == '__main__':
    pass