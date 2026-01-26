"""Support Vector Machine regressor for regression tasks.

This module provides a scikit-learn compatible wrapper for Support Vector
Regression (SVR) with optional feature scaling.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.utils import check_array, check_X_y
from sklearn.utils.validation import check_is_fitted


class SupportVectorRegressor(BaseEstimator, RegressorMixin):
    """Support Vector Machine regressor with optional scaling.

    This class provides a scikit-learn compatible wrapper for Support Vector
    Regression (SVR) with built-in feature and target scaling capabilities.
    Scaling is recommended for SVR as it is sensitive to feature scales.

    Parameters
    ----------
    kernel : {'linear', 'poly', 'rbf', 'sigmoid'}, default='rbf'
        Kernel type to be used in the algorithm.
    gamma : {'scale', 'auto'} or float, default='auto'
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
    tol : float, default=0.01
        Tolerance for stopping criterion.
    C : float, default=1.0
        Regularization parameter. The strength of the regularization is
        inversely proportional to C.
    epsilon : float, default=0.1
        Epsilon in the epsilon-SVR model. It specifies the epsilon-tube
        within which no penalty is associated in the training loss function.
    scale : bool, default=True
        Whether to scale features and target using StandardScaler.
        Recommended to keep True for better performance.

    Attributes
    ----------
    estimator_ : SVR
        The fitted SVR estimator.
    scaler_X_ : StandardScaler or None
        Feature scaler if scale=True, None otherwise.
    scaler_y_ : StandardScaler or None
        Target scaler if scale=True, None otherwise.
    n_features_in_ : int
        Number of features seen during fit.

    Examples
    --------
    >>> from yikit.models import SupportVectorRegressor
    >>> import numpy as np
    >>> X = np.random.randn(100, 10)
    >>> y = np.random.randn(100)
    >>> model = SupportVectorRegressor(kernel='rbf', C=1.0, scale=True)
    >>> model.fit(X, y)
    >>> predictions = model.predict(X)
    """
    def __init__(
        self,
        kernel="rbf",
        gamma="auto",
        tol=0.01,
        C=1.0,
        epsilon=0.1,
        scale=True,
    ):
        self.kernel = kernel
        self.gamma = gamma
        self.tol = tol
        self.C = C
        self.epsilon = epsilon
        self.scale = scale

    def fit(self, X, y):
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

        self.estimator_ = SVR(
            kernel=self.kernel,
            gamma=self.gamma,
            tol=self.tol,
            C=self.C,
            epsilon=self.epsilon,
        )
        self.estimator_.fit(X_, y_)

        return self

    def predict(self, X):
        # fitが行われたかどうかをインスタンス変数が定義されているかで判定（第二引数を文字列ではなくてリストで与えることでより厳密に判定可能）
        check_is_fitted(self, "estimator_")

        # 入力されたXが妥当か判定
        X = check_array(X)

        if self.scale:
            X_ = self.scaler_X_.transform(X)
        else:
            X_ = X

        y_pred_ = self.estimator_.predict(X_)
        if self.scale:
            y_pred_ = self.scaler_y_.inverse_transform(
                np.array(y_pred_).reshape(-1, 1)
            ).flatten()

        return y_pred_
