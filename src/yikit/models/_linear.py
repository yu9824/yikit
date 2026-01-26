from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Lasso, Ridge
from sklearn.utils import check_array, check_random_state, check_X_y
from sklearn.utils.validation import check_is_fitted


class LinearModelRegressor(BaseEstimator, RegressorMixin):
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
