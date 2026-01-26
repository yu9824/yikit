"""Neural Network regressor using Keras.

This module provides a scikit-learn compatible wrapper for neural network
regression using Keras with configurable architecture and training options.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array, check_random_state, check_X_y
from sklearn.utils.validation import check_is_fitted

from yikit.helpers import is_installed

if is_installed("tqdm"):
    from tqdm.auto import tqdm
else:
    from yikit.helpers import dummy_tqdm as tqdm


class NNRegressor(BaseEstimator, RegressorMixin):
    """Neural Network regressor using Keras.

    This class provides a scikit-learn compatible wrapper for neural network
    regression using Keras. It supports configurable network architecture
    with dropout, batch normalization, and early stopping.

    Parameters
    ----------
    input_dropout : float, default=0.0
        Dropout rate for the input layer.
    hidden_layers : int, default=3
        Number of hidden layers.
    hidden_units : int, default=96
        Number of units in each hidden layer.
    hidden_activation : {'relu', 'prelu'}, default='relu'
        Activation function for hidden layers.
    hidden_dropout : float, default=0.2
        Dropout rate for hidden layers.
    batch_norm : {'before_act'}, default='before_act'
        Batch normalization placement. Currently only 'before_act' is supported.
    optimizer_type : {'adam', 'sgd'}, default='adam'
        Optimizer to use for training.
    lr : float, default=0.001
        Learning rate for the optimizer.
    batch_size : int, default=64
        Batch size for training.
    l : float, default=0.01
        L2 regularization coefficient.
    random_state : int, RandomState instance or None, default=None
        Random state for reproducibility.
    epochs : int, default=200
        Maximum number of training epochs.
    patience : int, default=20
        Number of epochs with no improvement after which training will be stopped.
    progress_bar : bool, default=True
        Whether to show a progress bar during model construction.
    scale : bool, default=True
        Whether to scale features and target using StandardScaler.

    Attributes
    ----------
    estimator_ : keras.models.Sequential
        The fitted Keras neural network model.
    scaler_X_ : StandardScaler or None
        Feature scaler if scale=True, None otherwise.
    scaler_y_ : StandardScaler or None
        Target scaler if scale=True, None otherwise.
    n_features_in_ : int
        Number of features seen during fit.
    rng_ : RandomState
        Random state instance used for reproducibility.
    early_stopping_ : EarlyStopping
        Early stopping callback used during training.
    validation_split_ : float
        Validation split ratio used during training.

    Examples
    --------
    >>> from yikit.models import NNRegressor
    >>> import numpy as np
    >>> X = np.random.randn(100, 10)
    >>> y = np.random.randn(100)
    >>> model = NNRegressor(hidden_layers=3, hidden_units=96, epochs=100)
    >>> model.fit(X, y)
    >>> predictions = model.predict(X)

    Notes
    -----
    This class requires the 'keras' package to be installed.
    """
    def __init__(
        self,
        input_dropout=0.0,
        hidden_layers=3,
        hidden_units=96,
        hidden_activation="relu",
        hidden_dropout=0.2,
        batch_norm="before_act",
        optimizer_type="adam",
        lr=0.001,
        batch_size=64,
        l=0.01,  # noqa: E741   # FIXME
        random_state=None,
        epochs=200,
        patience=20,
        progress_bar=True,
        scale=True,
    ):
        if not is_installed("keras"):
            raise ModuleNotFoundError(
                "If you want to use this module, please install keras."
            )

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
        from keras.callbacks import (  # type: ignore[reportMissingImports]
            EarlyStopping,
        )
        from keras.layers import (  # type: ignore[reportMissingImports]
            Dense,
            Dropout,
        )
        from keras.layers.advanced_activations import (  # type: ignore[reportMissingImports]
            PReLU,
            ReLU,
        )
        from keras.layers.normalization import (  # type: ignore[reportMissingImports]
            BatchNormalization,
        )
        from keras.models import (  # type: ignore[reportMissingImports]
            Sequential,
        )
        from keras.optimizers import (  # type: ignore[reportMissingImports]
            SGD,
            Adam,
        )
        from keras.regularizers import l2  # type: ignore[reportMissingImports]

        # 入力されたXとyが良い感じか判定（サイズが適切かetc)
        X, y = check_X_y(X, y)

        """
        sklearn/utils/estimator_checks.py:3063:
        FutureWarning: As of scikit-learn 0.23, estimators should expose a n_features_in_ attribute,
        unless the 'no_validation' tag is True.
        This attribute should be equal to the number of features passed to the fit method.
        An error will be raised from version 1.0 (renaming of 0.25) when calling check_estimator().
        See SLEP010: https://scikit-learn-enhancement-proposals.readthedocs.io/en/latest/slep010/proposal.html
        """
        self.n_features_in_ = X.shape[
            1
        ]  # check_X_yのあとでないとエラーになりうる．

        # check_random_state
        self.rng_ = check_random_state(self.random_state)

        # 標準化
        if self.scale:
            self.scaler_X_ = StandardScaler()
            X_ = self.scaler_X_.fit_transform(X)

            self.scaler_y_ = StandardScaler()
            y_ = self.scaler_y_.fit_transform(
                np.array(y).reshape(-1, 1)
            ).flatten()
        else:
            X_ = X
            y_ = y

        # なぜか本当によくわかんないけど名前が一緒だと怒られるのでそれをずらすためのやつをつくる
        j = int(10e5)

        # プログレスバー
        if self.progress_bar:
            pbar = tqdm(total=1 + self.hidden_layers + 6)
        self.estimator_ = Sequential()

        # 入力層
        self.estimator_.add(
            Dropout(
                self.input_dropout,
                input_shape=(self.n_features_in_,),
                seed=self.rng_.randint(2**31 - 1),
                name="Dropout_" + str(j),
            )
        )
        if self.progress_bar:
            pbar.update(1)

        # 中間層
        for i in range(self.hidden_layers):
            self.estimator_.add(
                Dense(
                    units=self.hidden_units,
                    kernel_regularizer=l2(l=self.l),
                    name="Dense_" + str(j + i + 1),
                )
            )  # kernel_regularizer: 過学習対策
            if self.batch_norm == "before_act":
                self.estimator_.add(
                    BatchNormalization(
                        name="BatchNormalization" + str(j + i + 1)
                    )
                )

            # 活性化関数
            if self.hidden_activation == "relu":
                self.estimator_.add(ReLU(name="Re_Lu_" + str(j + i + 1)))
            elif self.hidden_activation == "prelu":
                self.estimator_.add(PReLU(name="PRe_Lu_" + str(j + i + 1)))
            else:
                raise NotImplementedError

            self.estimator_.add(
                Dropout(
                    self.hidden_dropout,
                    seed=self.rng_.randint(2**32),
                    name="Dropout_" + str(j + i + 1),
                )
            )

            # プログレスバー
            if self.progress_bar:
                pbar.update(1)

        # 出力層
        self.estimator_.add(
            Dense(
                1,
                activation="linear",
                name="Dense_" + str(j + self.hidden_layers + 1),
            )
        )

        # optimizer
        if self.optimizer_type == "adam":
            optimizer_ = Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, decay=0.0)
        elif self.optimizer_type == "sgd":
            optimizer_ = SGD(
                lr=self.lr, decay=1e-6, momentum=0.9, nesterov=True
            )
        else:
            raise NotImplementedError

        # 目的関数，評価指標などの設定
        self.estimator_.compile(
            loss="mean_squared_error",
            optimizer=optimizer_,
            metrics=["mse", "mae"],
        )

        # プログレスバー
        if self.progress_bar:
            pbar.update(1)

        # 変数の定義
        self.early_stopping_ = EarlyStopping(
            patience=self.patience, restore_best_weights=True
        )
        self.validation_split_ = 0.2

        # fit
        self.estimator_.fit(
            X_,
            y_,
            validation_split=self.validation_split_,
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=[self.early_stopping_],
            verbose=0,
        )

        # プログレスバー
        if self.progress_bar:
            pbar.update(5)
            pbar.close()

        return self

    def predict(self, X):
        # fitが行われたかどうかをインスタンス変数が定義されているかで判定（第二引数を文字列ではなくてリストで与えることでより厳密に判定可能）
        check_is_fitted(self, "estimator_")

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
