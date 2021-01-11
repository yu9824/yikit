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
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import train_test_split

from lightgbm import LGBMRegressor

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

        if isinstance(X, pd.DataFrame) or isinstance(X, np.ndarray):
            n_features_ = X.shape[1]
        elif isinstance(X, list):
            X = np.array(X)
            n_features_ = np.array(X)

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
        self.estimator_.add(Dropout(self.input_dropout, input_shape = (n_features_,), seed = self.random_state, name = 'Dropout_' + str(j)))
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

            self.estimator_.add(Dropout(self.hidden_dropout, seed = self.random_state, name = 'Dropout_' + str(j+i+1)))

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
            X_ = self.scaler_X_.fit_transform(X)
            y_pred_ = self.estimator_.predict(X)
            y_pred_ = self.scaler_y_.inverse_transform(y_pred_).flatten()
        else:
            y_pred_ = self.estimator_.predict(X).flatten()
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
            random_state = self.random_state,
            n_jobs = self.n_jobs,
            silent = self.silent,
            importance_type = self.importance_type,
            **kwargs
        )

        # 入力されたXとyが良い感じか判定（サイズが適切かetc)
        X, y = check_X_y(X, y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = self.random_state, test_size = 0.2)

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


if __name__ == '__main__':
    import warnings
    warnings.simplefilter('ignore', FutureWarning)
    
    check_estimator(GBDTRegressor)
    # check_estimator(NNRegressor)
