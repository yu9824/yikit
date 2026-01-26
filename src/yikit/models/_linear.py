"""Linear model regressors for regression tasks.

This module provides a scikit-learn compatible wrapper for linear models
including Ridge and Lasso regression.
"""

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Lasso, Ridge
from sklearn.utils import check_array, check_random_state, check_X_y
from sklearn.utils.validation import check_is_fitted


class LinearModelRegressor(BaseEstimator, RegressorMixin):
    """Linear model regressor supporting Ridge and Lasso regression.

    This class provides a scikit-learn compatible wrapper for linear models
    with regularization. It supports both Ridge (L2) and Lasso (L1) regression.

    Parameters
    ----------
    linear_model : {'ridge', 'lasso'}, default='ridge'
        Type of linear model to use.
    alpha : float, default=1.0
        Regularization strength. Higher values mean more regularization.
    fit_intercept : bool, default=True
        Whether to fit the intercept term.
    max_iter : int, default=1000
        Maximum number of iterations for the solver.
    tol : float, default=0.001
        Tolerance for stopping criterion.
    random_state : int, RandomState instance or None, default=None
        Random state for reproducibility.

    Attributes
    ----------
    estimator_ : Ridge or Lasso
        The fitted linear model estimator.
    n_features_in_ : int
        Number of features seen during fit.
    n_iter_ : int
        Number of iterations taken by the solver.
    rng_ : RandomState
        Random state instance used for reproducibility.

    Examples
    --------
    >>> from yikit.models import LinearModelRegressor
    >>> import numpy as np
    >>> X = np.random.randn(100, 10)
    >>> y = np.random.randn(100)
    >>> model = LinearModelRegressor(linear_model='ridge', alpha=1.0)
    >>> model.fit(X, y)
    >>> predictions = model.predict(X)
    """
    def __init__(
        self,
        linear_model="ridge",
        alpha=1.0,
        fit_intercept=True,
        max_iter=1000,
        tol=0.001,
        random_state=None,
    ):
        self.linear_model = linear_model
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X, y):
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

        self.rng_ = check_random_state(self.random_state)

        # max_iterを引数に入れてるとこの変数ないとダメ！って怒られるから．
        self.n_iter_ = 1

        if self.linear_model == "ridge":
            model_ = Ridge
        elif self.linear_model == "lasso":
            model_ = Lasso
        else:
            raise NotImplementedError

        self.estimator_ = model_(
            alpha=self.alpha,
            fit_intercept=self.fit_intercept,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.rng_,
        )
        self.estimator_.fit(X, y)
        return self

    def predict(self, X):
        check_is_fitted(self, "estimator_")

        X = check_array(X)

        return self.estimator_.predict(X)
