"""Ensemble methods for regression.

This module provides ensemble regression methods including blending, averaging,
and stacking of multiple base estimators.
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, RegressorMixin, clone, is_regressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import check_scoring
from sklearn.metrics._scorer import _check_multimetric_scoring
from sklearn.model_selection import check_cv
from sklearn.model_selection._validation import _score
from sklearn.utils import Bunch, check_array, check_random_state, check_X_y
from sklearn.utils.validation import check_is_fitted

from yikit.helpers import is_installed
from yikit.metrics import root_mean_squared_error


class EnsembleRegressor(BaseEstimator, RegressorMixin):
    """Ensemble regressor combining multiple base estimators.

    This class provides ensemble methods for regression including blending,
    averaging, and stacking. It supports optional feature selection using
    Boruta and hyperparameter optimization using Optuna.

    Parameters
    ----------
    estimators : list of estimator objects, default=(RandomForestRegressor(),)
        List of base estimators to ensemble. Each estimator must be a
        scikit-learn compatible regressor.
    method : {'blending', 'average', 'stacking'}, default='blending'
        Ensemble method to use:
        - 'blending': Train each estimator on a different fold and combine predictions
        - 'average': Simple average of predictions from all estimators
        - 'stacking': Use a meta-learner to combine predictions
    cv : int, cross-validation generator or iterable, default=5
        Determines the cross-validation splitting strategy.
    n_jobs : int, default=-1
        Number of jobs to run in parallel.
    random_state : int, RandomState instance or None, default=None
        Random state for reproducibility.
    scoring : str, callable or list, default='neg_mean_squared_error'
        Scoring metric(s) to use. See scikit-learn documentation for options.
    verbose : int, default=0
        Verbosity level.
    boruta : bool, default=True
        Whether to perform Boruta feature selection before training.
    opt : bool, default=True
        Whether to perform hyperparameter optimization using Optuna.

    Attributes
    ----------
    estimators_ : list of fitted estimators
        The fitted base estimators.
    meta_estimator_ : estimator or None
        The meta-learner used in stacking (None for other methods).
    n_features_in_ : int
        Number of features seen during fit.
    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during fit.

    Examples
    --------
    >>> from yikit.models import EnsembleRegressor
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sklearn.linear_model import Ridge
    >>> import numpy as np
    >>> X = np.random.randn(100, 10)
    >>> y = np.random.randn(100)
    >>> ensemble = EnsembleRegressor(
    ...     estimators=[RandomForestRegressor(), Ridge()],
    ...     method='blending',
    ...     cv=5
    ... )
    >>> ensemble.fit(X, y)
    >>> predictions = ensemble.predict(X)
    """
    def __init__(
        self,
        estimators=(RandomForestRegressor(),),
        method="blending",
        cv=5,
        n_jobs=-1,
        random_state=None,
        scoring="neg_mean_squared_error",
        verbose=0,
        boruta=True,
        opt=True,
    ):
        """
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
        """
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
        rng_ = check_random_state(self.random_state)

        # isRegressor
        if (
            sum([is_regressor(estimator) for estimator in self.estimators])
            != self.n_estimators_
        ):
            raise ValueError

        # check_cv
        cv_ = check_cv(self.cv, y=y, classifier=False)

        # check_scoring
        estimator = self.estimators[0]
        if callable(self.scoring):
            scorers = self.scoring
        elif self.scoring is None or isinstance(self.scoring, str):
            scorers = check_scoring(estimator=estimator)
        else:
            # 0.24.1のコードだと辞書を返すことになっているが，0.23.2ではtupleが返ってきてしまう？
            scorers = _check_multimetric_scoring(estimator, self.scoring)
            if isinstance(scorers, tuple):
                scorers = scorers[0]

        # 並列処理する部分を関数化
        def _f(i_train, i_test):
            X_train, X_test, y_train, y_test = (
                X[i_train],
                X[i_test],
                y[i_train],
                y[i_test],
            )

            if self.boruta:
                if is_installed("boruta"):
                    from yikit.feature_selection import BorutaPy
                else:
                    raise ModuleNotFoundError(
                        "If you want to use boruta, please install boruta."
                    )

                # 特徴量削減
                feature_selector_ = BorutaPy(
                    estimator=RandomForestRegressor(
                        n_jobs=-1, random_state=rng_
                    ),
                    random_state=rng_,
                    max_iter=300,
                    verbose=self.verbose,
                )
                feature_selector_.fit(X_train, y_train)

                # 抽出
                support_ = feature_selector_.get_support()
                X_train_selected = feature_selector_.transform(X_train)
                X_test_selected = feature_selector_.transform(X_test)
            else:  # borutaしない場合でもresultsに組み込まれるので変数を定義しておく．
                support_ = np.ones(X_train.shape[1], dtype=np.bool)
                X_train_selected = X_train
                X_test_selected = X_test

            results_estimators = []
            for estimator in self.estimators:
                if self.opt:
                    if is_installed("optuna"):
                        import optuna

                        from yikit.models._optuna import Objective
                    else:
                        raise ModuleNotFoundError(
                            "If you want to use optuna optimization, please install optuna."
                        )

                    # verbose
                    if self.verbose == 0:
                        optuna.logging.disable_default_handler()

                    # ハイパーパラメータ（scoringで最初にしていしたやつで最適化）
                    objective = Objective(
                        estimator,
                        X_train_selected,
                        y_train,
                        cv=cv_,
                        random_state=rng_,
                        scoring=(
                            scorers.values()[0]
                            if isinstance(scorers, dict)
                            else scorers
                        ),
                    )
                    sampler = optuna.samplers.TPESampler(
                        seed=rng_.randint(2**31 - 1)
                    )

                    study = optuna.create_study(
                        sampler=sampler, direction="maximize"
                    )
                    study.optimize(objective, n_trials=100, n_jobs=1)

                    # 最適化後のモデル
                    _best_estimator_ = objective.model(
                        **objective.fixed_params_, **study.best_params
                    )
                else:  # optunaしない場合でもresultsに組み込まれるので変数を定義しておく．
                    study = None
                    _best_estimator_ = clone(estimator)

                # fit
                _best_estimator_.fit(X_train_selected, y_train)

                # predict
                _y_pred_on_train = _best_estimator_.predict(X_train_selected)
                _y_pred_on_test = _best_estimator_.predict(X_test_selected)

                # score
                _train_scores = _score(
                    _best_estimator_, X_train_selected, y_train, scorers
                )
                _test_scores = _score(
                    _best_estimator_, X_test_selected, y_test, scorers
                )

                # importances
                _gi = (
                    _best_estimator_.feature_importances_
                    if "feature_importances_" in dir(_best_estimator_)
                    else None
                )
                _pi = permutation_importance(
                    _best_estimator_,
                    X_test_selected,
                    y_test,
                    scoring="neg_mean_squared_error",
                    n_repeats=10,
                    n_jobs=-1,
                    random_state=rng_,
                ).importances

                # 予測結果をDataFrameにまとめる．
                _y_train = pd.DataFrame(
                    np.hstack(
                        [
                            y_train.reshape(-1, 1),
                            _y_pred_on_train.reshape(-1, 1),
                        ]
                    ),
                    columns=["true", "pred"],
                    index=i_train,
                )
                _y_test = pd.DataFrame(
                    np.hstack(
                        [y_test.reshape(-1, 1), _y_pred_on_test.reshape(-1, 1)]
                    ),
                    columns=["true", "pred"],
                    index=i_test,
                )

                results_estimators.append(
                    {
                        "estimators": _best_estimator_,
                        "params": _best_estimator_.get_params(),
                        "y_train": _y_train,
                        "y_test": _y_test,
                        "train_scores": _train_scores,
                        "test_scores": _test_scores,
                        "gini_importances": _gi,
                        "permutation_importances": _pi,
                        "studies": study,
                    }
                )
            # verbose
            if self.verbose == 0:
                optuna.logging.disable_default_handler()

            # 出力結果をいい感じにする．←ここから
            ret = {}
            temp = {}
            for (
                result_estimator
            ) in results_estimators:  # それぞれのestimatorについて
                for (
                    k,
                    v,
                ) in result_estimator.items():  # その中の各々の値について
                    # スコア系かつそれらが複数指定されている場合だけ特別処理
                    if "_score" in k and isinstance(v, dict):
                        if k not in temp:
                            temp[k] = {}
                        for score_name, score in v.items():
                            if score_name in temp[k]:
                                temp[k][score_name].append(score)
                            else:
                                temp[k][score_name] = [score]
                    else:  # スコア系以外
                        if k in ret:
                            ret[k].append(v)
                        else:
                            ret[k] = [v]
            # scoreをためてるやつをBunch型に変換
            for k in temp:
                temp[k] = Bunch(**temp[k])

            # 返すように最終整形
            ret["support_"] = support_
            ret.update(temp)
            return ret

        # 上記で定義した_f関数どうしは互いに独立なので並列で処理する．
        parallel = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)
        results = parallel(
            delayed(_f)(i_train, i_test) for i_train, i_test in cv_.split(X, y)
        )
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
        dfs_y_oof_ = [
            pd.concat(
                [lst[n] for lst in self.results_["y_test"]], axis=0
            ).sort_index()
            for n in range(self.n_estimators_)
        ]
        y_oof_ = pd.concat([df.loc[:, "pred"] for df in dfs_y_oof_], axis=1)
        y_oof_.columns = [
            "estimator{}".format(n) for n in range(self.n_estimators_)
        ]

        # *** ensemble ***
        # モデルがひとつのとき．
        if self.method == "average" or self.n_estimators_ == 1:
            self.weights_ = None
        elif self.method == "blending":
            # blendingのweightをoptunaで最適化
            if is_installed("optuna"):
                import optuna
            else:
                raise ModuleNotFoundError(
                    "When 'method=blending', please install optuna."
                )

            # rmseで最適化（今後指定できるようにしてもいいかも．）
            def objective(trial: optuna.trial.Trial) -> float:
                params = {
                    "weight{0}".format(i): trial.suggest_float(
                        "weight{0}".format(i), 0, 1
                    )
                    for i in range(self.n_estimators_)
                }
                weights = np.array(list(params.values()))
                y_oof_ave = np.average(y_oof_, weights=weights, axis=1)
                return root_mean_squared_error(y, y_oof_ave)

            # optunaのログを非表示
            if self.verbose == 0:
                optuna.logging.disable_default_handler()

            # 重みの最適化
            sampler_ = optuna.samplers.TPESampler(seed=rng_.randint(2**31 - 1))
            study = optuna.create_study(
                sampler=sampler_, direction="minimize"
            )  # 普通のRMSEなので．
            study.optimize(
                objective, n_trials=100, n_jobs=1
            )  # -1にするとなぜかバグるので．（そもそもそんなに重くないので1で．）

            # optunaのログを再表示
            if self.verbose == 0:
                optuna.logging.enable_default_handler()

            self.weights_ = np.array(
                list(study.best_params.values()), dtype=np.float64
            )
            self.weights_ /= np.sum(self.weights_)
        elif self.method == "stacking":
            # 線形モデルの定義
            self.stacking_model_ = LinearRegression(n_jobs=self.n_jobs)
            self.stacking_model_.fit(y_oof_.values, y)
            # resultsに保存するために定義だけする．
            self.weights_ = None
        else:
            raise NotImplementedError("Method not implemented")

        # 重みを結果に保存
        self.results_["weights"] = self.weights_

        return self

    def predict(self, X):
        # fitが行われたかどうかをインスタンス変数が定義されているかで判定（第二引数を文字列ではなくてリストで与えることでより厳密に判定可能）
        check_is_fitted(self, "results_")

        # 入力されたXが妥当か判定
        X = check_array(X)

        # 各予測モデルの予測結果をまとめる．(内包リストで得られる配列は3-Dベクトル (n_estimators_, cv, n_samples))
        y_preds_ = np.average(
            np.array(
                [
                    [
                        estimators[n].predict(X[:, self.results_.support_[m]])
                        for m, estimators in enumerate(
                            self.results_.estimators
                        )
                    ]
                    for n in range(self.n_estimators_)
                ]
            ),
            axis=1,
        ).transpose()  # 同じ種類のやつは単純に平均を取る．

        if self.method in ("blending", "average") or self.n_estimators_ == 1:
            y_pred_ = np.average(y_preds_, weights=self.weights_, axis=1)
        elif self.method == "stacking":
            y_pred_ = self.stacking_model_.predict(y_preds_)
        return y_pred_
