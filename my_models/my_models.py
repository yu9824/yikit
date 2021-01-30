from keras.callbacks import EarlyStopping
from keras.layers.advanced_activations import ReLU, PReLU
from keras.layers import Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras.regularizers import l2
from keras.backend import clear_session

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils import check_array, check_X_y, check_random_state
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.base import clone
from sklearn.svm import SVR

from lightgbm import LGBMRegressor
import optuna
from tqdm import tqdm

import numpy as np
import pandas as pd


class NNRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, input_dropout = 0.0, hidden_layers = 3, hidden_units = 96, hidden_activation = 'relu', hidden_dropout = 0.2, batch_norm = 'before_act', optimizer_type = 'adam', lr = 0.001, batch_size = 64, l = 0.01, random_state = None, epochs = 200, patience = 20, progress_bar = True, scale = True):
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
        # 入力されたXとyが良い感じか判定（サイズが適切かetc)
        X, y = check_X_y(X, y)

        # check_random_state
        self.rng_ = self.random_state
        
        if isinstance(X, list):
            X = np.array(X)
        n_features_ = X.shape[1]
            

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
        self.estimator_.add(Dropout(self.input_dropout, input_shape = (n_features_,), seed = self.rng_.randint(2 ** 32), name = 'Dropout_' + str(j)))
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

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = self.rng_, test_size = 0.2)

        self.estimator_.fit(X, y, eval_set = [(X_test, y_test)], eval_metric = ['mse', 'mae'], early_stopping_rounds = 20, verbose = False)

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
    def __init__(self, estimators = [RandomForestRegressor()], method = 'blending', cv = 5, n_jobs = -1, random_state = None, metric = 'mse', silent = True):
        self.estimators = estimators
        self.method = method
        self.cv = cv
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.metric = metric
        self.silent = silent

    def fit(self, X, y):
        '''
        self.estimatorsはfitしてないモデル．self.estimators_はfitしたモデル．
        '''
        # 入力されたXとyが良い感じか判定（サイズが適切かetc)
        X, y = check_X_y(X, y)

        # check_random_state
        self.rng_ = check_random_state(self.random_state)

        # よく使うので変数化
        n_estimators_ = len(self.estimators)

        # metricの処理
        if self.metric == 'r2':
            metric_ = r2_score
            direction_ = 'maximize'
        else:
            direction_ = 'minimize'
            if self.metric == 'mse':
                metric_ = mean_squared_error
            elif self.metric == 'mae':
                metric_ = mean_absolute_error
            elif self.metric == 'rmse':
                metric_ = lambda x, y:mean_squared_error(x, y, squared = False)
            else:
                raise NotImplementedError('{}'.format(self.metric))
        
        # 各モデルのOOFの予測値を求める．
        self.y_oof_s_ = [cross_val_predict(estimator, X, y, cv = self.cv, n_jobs = self.n_jobs) for estimator in self.estimators]

        if self.method == 'blending':
            def objective(trial):
                params = {'weight{0}'.format(i): trial.suggest_uniform('weight{0}'.format(i), 0, 1) for i in range(n_estimators_)}
                weights = np.array(list(params.values()))
                y_oof_ave = np.average(self.y_oof_s_, weights = weights, axis = 0)
                return metric_(y_oof_ave, y)
            # optunaのログを非表示
            if self.silent:
                optuna.logging.disable_default_handler()

            # 重みの最適化
            sampler_ = optuna.samplers.TPESampler(seed = self.rng_.randint(2 ** 32))
            study = optuna.create_study(sampler = sampler_, direction = direction_)
            study.optimize(objective, n_trials = 100, n_jobs = 1)   # -1にするとバグる

            # optunaのログを再表示
            if self.silent:
                optuna.logging.enable_default_handler()

            self.weights_ = np.array(list(study.best_params.values()))
            self.weights_ /= np.sum(self.weights_)
        elif self.method == 'average':
            self.weights_ = np.ones(n_estimators_) / n_estimators_
        elif self.method == 'stacking':
            # 線形モデルの定義
            self.stacking_model_ = LinearRegression(n_jobs = self.n_jobs)
            self.stacking_model_.fit(np.array(self.y_oof_s_).transpose(), y)
        else:
            raise NotImplementedError
        
        # すべてのモデルをfitして保存（copyしないと元のオブジェクトまで変わってしまい，エラーが生じる．）
        self.estimators_ = []
        for i in range(n_estimators_):
            estimator_ = clone(self.estimators[i])
            estimator_.fit(X, y)
            self.estimators_.append(estimator_)
        return self


    def predict(self, X):
        # fitが行われたかどうかをインスタンス変数が定義されているかで判定（第二引数を文字列ではなくてリストで与えることでより厳密に判定可能）
        check_is_fitted(self, 'estimators_')

        # 入力されたXが妥当か判定
        X = check_array(X)

        # 各予測モデルの予測結果をまとめる
        y_pred_s_ = [estimator_.predict(X) for estimator_ in self.estimators_]
        
        if self.method in ('blending', 'average'):
            y_pred_ = np.average(y_pred_s_, weights = self.weights_, axis = 0)
        elif self.method == 'stacking':
            y_pred_ = self.stacking_model_.predict(np.array(y_pred_s_).transpose())
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

        if self.scale:
            self.scaler_X_ = StandardScaler()
            X_ = self.scaler_X_.fit_transform(X)

            self.scaler_y_ = StandardScaler()
            y_ = self.scaler_y_.fit_transform(np.array(y).reshape(-1, 1))
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



if __name__ == '__main__':
    import warnings
    warnings.simplefilter('ignore', FutureWarning)

    from pdb import set_trace
    
    # check_estimator(GBDTRegressor)
    # check_estimator(NNRegressor)
    # check_estimator(EnsembleRegressor(random_state = 334, n_jobs = -1, estimators = [RandomForestRegressor(random_state=334)]))
    # check_estimator(SupportVectorRegressor)
    check_estimator(LinearModelRegressor())

    # from sklearn.datasets import load_boston
    # boston = load_boston()
    # X = pd.DataFrame(boston['data'], columns = boston['feature_names'])
    # y = pd.Series(boston['target'], name = 'PRICE')

    # for method in ('blending', 'average', 'stacking'):
    #     print(method)
    #     er = EnsembleRegressor(random_state = 334, n_jobs = -1, estimators = [RandomForestRegressor(random_state=334)], cv = 5)
    #     {print('MSE{0}: {1:.3f}'.format(i, -score)) for i, score in enumerate(cross_val_score(er, X, y, scoring = 'neg_mean_squared_error', cv = 20))}

