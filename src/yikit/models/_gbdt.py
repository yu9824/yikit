"""Gradient Boosting Decision Tree regressor using LightGBM.

This module provides a scikit-learn compatible wrapper for LightGBM's
gradient boosting decision tree regressor with early stopping.
"""

from lightgbm import LGBMRegressor  # type: ignore[reportMissingImports]
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.utils import check_array, check_random_state, check_X_y
from sklearn.utils.validation import check_is_fitted


class GBDTRegressor(RegressorMixin, BaseEstimator):
    """Gradient Boosting Decision Tree regressor using LightGBM.

    This class provides a scikit-learn compatible wrapper for LightGBM's
    gradient boosting decision tree regressor. It includes automatic early
    stopping using a validation set and supports all LightGBM parameters.

    Parameters
    ----------
    boosting_type : str, default='gbdt'
        Type of boosting algorithm to use.
    num_leaves : int, default=31
        Maximum tree leaves for base learners.
    max_depth : int, default=-1
        Maximum tree depth for base learners, <=0 means no limit.
    learning_rate : float, default=0.1
        Boosting learning rate.
    n_estimators : int, default=100
        Number of boosted trees to fit.
    subsample_for_bin : int, default=200000
        Number of samples for constructing bins.
    objective : str or None, default=None
        Specify the learning task and the corresponding learning objective.
    class_weight : dict, 'balanced' or None, default=None
        Weights associated with classes.
    min_split_gain : float, default=0.0
        Minimum loss reduction required to make a further partition.
    min_child_weight : float, default=0.001
        Minimum sum of instance weight (hessian) needed in a child.
    min_child_samples : int, default=20
        Minimum number of data needed in a child (leaf).
    subsample : float, default=1.0
        Subsample ratio of the training instance.
    subsample_freq : int, default=0
        Frequency of subsample, <=0 means no enable.
    colsample_bytree : float, default=1.0
        Subsample ratio of columns when constructing each tree.
    reg_alpha : float, default=0.0
        L1 regularization term on weights.
    reg_lambda : float, default=0.0
        L2 regularization term on weights.
    random_state : int, RandomState instance or None, default=None
        Random state for reproducibility.
    n_jobs : int, default=-1
        Number of parallel threads.
    silent : bool, default=True
        Whether to print messages while running boosting.
    importance_type : str, default='split'
        The type of feature importance to be filled in feature_importances_.
    **kwargs
        Additional keyword arguments passed to LGBMRegressor.

    Attributes
    ----------
    estimator_ : LGBMRegressor
        The fitted LightGBM regressor.
    feature_importances_ : array-like of shape (n_features,)
        The feature importances.
    n_features_in_ : int
        Number of features seen during fit.
    rng_ : RandomState
        Random state instance used for reproducibility.

    Examples
    --------
    >>> from yikit.models import GBDTRegressor
    >>> import numpy as np
    >>> X = np.random.randn(100, 10)
    >>> y = np.random.randn(100)
    >>> model = GBDTRegressor(n_estimators=100, learning_rate=0.1)
    >>> model.fit(X, y)
    >>> predictions = model.predict(X)

    Notes
    -----
    This class requires the 'lightgbm' package to be installed.
    """

    def __init__(
        self,
        boosting_type="gbdt",
        num_leaves=31,
        max_depth=-1,
        learning_rate=0.1,
        n_estimators=100,
        subsample_for_bin=200000,
        objective=None,
        class_weight=None,
        min_split_gain=0.0,
        min_child_weight=0.001,
        min_child_samples=20,
        subsample=1.0,
        subsample_freq=0,
        colsample_bytree=1.0,
        reg_alpha=0.0,
        reg_lambda=0.0,
        random_state=None,
        n_jobs=-1,
        silent=True,
        importance_type="split",
        **kwargs,
    ):
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
        except Exception:
            kwargs = {}

        # check_random_state
        self.rng_ = check_random_state(self.random_state)

        # fitしたあとに確定する値は変数名 + '_' としなければならない．
        self.estimator_ = LGBMRegressor(
            boosting_type=self.boosting_type,
            num_leaves=self.num_leaves,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            subsample_for_bin=self.subsample_for_bin,
            objective=self.objective,
            class_weight=self.class_weight,
            min_split_gain=self.min_split_gain,
            min_child_weight=self.min_child_weight,
            min_child_samples=self.min_child_samples,
            subsample=self.subsample,
            subsample_freq=self.subsample_freq,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.rng_,
            n_jobs=self.n_jobs,
            silent=self.silent,
            importance_type=self.importance_type,
            **kwargs,
        )

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

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=self.rng_, test_size=0.2
        )

        self.estimator_.fit(
            X,
            y,
            eval_set=[(X_test, y_test)],
            eval_metric=["mse", "mae"],
            early_stopping_rounds=20,
            verbose=False,
        )
        self.feature_importances_ = self.estimator_.feature_importances_

        # 慣例と聞いたはずなのにこれをreturnしないと怒られる．審査が厳しい．
        return self

    def predict(self, X):
        # fitが行われたかどうかをインスタンス変数が定義されているかで判定（第二引数を文字列ではなくてリストで与えることでより厳密に判定可能）
        check_is_fitted(self, "estimator_")

        # 入力されたXが妥当か判定
        X = check_array(X)

        # 予測結果を返す
        return self.estimator_.predict(X)
