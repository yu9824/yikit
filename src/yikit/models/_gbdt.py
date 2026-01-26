from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.utils import check_array, check_random_state, check_X_y
from sklearn.utils.validation import check_is_fitted

from yikit.helpers import is_installed

if is_installed("lightgbm"):
    from lightgbm import LGBMRegressor
else:
    LGBMRegressor = None  # type: ignore[assignment,misc]


class GBDTRegressor(RegressorMixin, BaseEstimator):
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
