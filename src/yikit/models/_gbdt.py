"""Gradient Boosting Decision Tree regressor using LightGBM.

This module provides a scikit-learn compatible wrapper for LightGBM's
gradient boosting decision tree regressor with early stopping.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any, cast

import lightgbm  # type: ignore[reportMissingImports]
from lightgbm import LGBMRegressor  # type: ignore[reportMissingImports]
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.utils import check_array, check_random_state, check_X_y
from sklearn.utils.validation import check_is_fitted

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

#: Fraction of the training data held out to decide early stopping.
_VALIDATION_FRACTION = 0.2

#: Rounds without improvement of any validation metric before stopping.
_EARLY_STOPPING_ROUNDS = 20

#: Validation metrics watched by early stopping.
_EVAL_METRIC = ["mse", "mae"]

#: Parameters of the installed ``LGBMRegressor.fit``, by name.
_FIT_PARAMETERS = inspect.signature(LGBMRegressor.fit).parameters


def _validation_fit_params(
    X_valid: NDArray[Any], y_valid: NDArray[Any]
) -> dict[str, Any]:
    """Return the ``LGBMRegressor.fit`` arguments for early stopping.

    The arguments are chosen from the signature of the installed
    ``LGBMRegressor.fit``; the ``early_stopping`` callback works on both
    LightGBM 3 and 4.

    - LightGBM 4.7 deprecates ``eval_set`` in favor of ``eval_X`` and
      ``eval_y``, so those are used when ``fit`` accepts them.
    - Before LightGBM 3.3, ``fit`` logs every iteration unless ``verbose`` is
      False (its default is True). LightGBM 3.3 deprecates ``verbose`` (its
      default is ``"warn"``) and stops logging by itself when callbacks are
      given, and LightGBM 4 removes it, so ``verbose`` is passed only when
      its default is True.
    """
    params: dict[str, Any] = {
        "eval_metric": _EVAL_METRIC,
        "callbacks": [
            lightgbm.early_stopping(_EARLY_STOPPING_ROUNDS, verbose=False)
        ],
    }
    if "eval_X" in _FIT_PARAMETERS:
        params.update(eval_X=(X_valid,), eval_y=(y_valid,))
    else:
        params["eval_set"] = [(X_valid, y_valid)]
    verbose = _FIT_PARAMETERS.get("verbose")
    if verbose is not None and verbose.default is True:
        params["verbose"] = False
    return params


class GBDTRegressor(RegressorMixin, BaseEstimator):
    """Gradient Boosting Decision Tree regressor using LightGBM.

    ``fit`` holds out 20% of the training data as a validation set, trains
    ``lightgbm.LGBMRegressor`` on the rest, and stops adding trees when
    neither the mean squared error nor the mean absolute error on the
    validation set improves for 20 rounds. Predictions use the best
    iteration. It works with LightGBM 3 and 4.

    Only the parameters below are accepted. To pass other LightGBM
    parameters, use ``lightgbm.LGBMRegressor`` directly; it is searched by
    ``yikit.models.Objective`` with the same search space.

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
        Maximum number of boosted trees to fit.
    subsample_for_bin : int, default=200000
        Number of samples for constructing bins.
    objective : str, callable or None, default=None
        Learning objective. None means LightGBM's default for regression.
    class_weight : dict, 'balanced' or None, default=None
        Passed to ``LGBMRegressor``; it has no effect on regression.
    min_split_gain : float, default=0.0
        Minimum loss reduction required to make a further partition.
    min_child_weight : float, default=0.001
        Minimum sum of instance weight (hessian) needed in a child.
    min_child_samples : int, default=20
        Minimum number of data needed in a child (leaf).
    subsample : float, default=1.0
        Subsample ratio of the training instances.
    subsample_freq : int, default=0
        Frequency of subsample, <=0 means no enable.
    colsample_bytree : float, default=1.0
        Subsample ratio of columns when constructing each tree.
    reg_alpha : float, default=0.0
        L1 regularization term on weights.
    reg_lambda : float, default=0.0
        L2 regularization term on weights.
    random_state : int, RandomState instance or None, default=None
        Controls the split into training and validation data and the
        randomness of LightGBM.
    n_jobs : int, default=-1
        Number of parallel threads used by LightGBM.
    silent : bool, default=True
        If True, LightGBM prints nothing (``verbosity=-1``); otherwise it
        prints information messages (``verbosity=1``).
    importance_type : str, default='split'
        The type of feature importance in ``feature_importances_``.

    Attributes
    ----------
    estimator_ : lightgbm.LGBMRegressor
        The fitted LightGBM regressor.
    best_iteration_ : int
        Number of trees chosen by early stopping and used by ``predict``.
        It is read from the booster, because ``LGBMRegressor.best_iteration_``
        stays None on LightGBM 3 when early stopping is set by a callback.
    feature_importances_ : ndarray of shape (n_features,)
        The feature importances.
    n_features_in_ : int
        Number of features seen during fit.
    rng_ : RandomState
        Random state used for the split and passed to LightGBM.

    Examples
    --------
    >>> import numpy as np
    >>> from yikit.models import GBDTRegressor
    >>> rng = np.random.RandomState(0)
    >>> X = rng.normal(size=(100, 5))
    >>> y = X @ rng.normal(size=5) + rng.normal(scale=0.1, size=100)
    >>> model = GBDTRegressor(n_estimators=500, random_state=0).fit(X, y)
    >>> model.predict(X).shape
    (100,)
    >>> model.best_iteration_ < 500
    True

    Notes
    -----
    This class requires the 'lightgbm' package to be installed.
    """

    def __init__(
        self,
        boosting_type: str = "gbdt",
        num_leaves: int = 31,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        n_estimators: int = 100,
        subsample_for_bin: int = 200000,
        objective: Any = None,
        class_weight: Any = None,
        min_split_gain: float = 0.0,
        min_child_weight: float = 0.001,
        min_child_samples: int = 20,
        subsample: float = 1.0,
        subsample_freq: int = 0,
        colsample_bytree: float = 1.0,
        reg_alpha: float = 0.0,
        reg_lambda: float = 0.0,
        random_state: Any = None,
        n_jobs: int = -1,
        silent: bool = True,
        importance_type: str = "split",
    ) -> None:
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

    def fit(self, X: ArrayLike, y: ArrayLike) -> GBDTRegressor:
        """Fit the model with early stopping on a held-out split.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        GBDTRegressor
            The fitted estimator.
        """
        X_checked, y_checked = check_X_y(X, y)
        self.n_features_in_ = X_checked.shape[1]
        self.rng_ = check_random_state(self.random_state)

        X_train, X_valid, y_train, y_valid = train_test_split(
            X_checked,
            y_checked,
            test_size=_VALIDATION_FRACTION,
            random_state=self.rng_,
        )
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
            importance_type=self.importance_type,
            verbosity=-1 if self.silent else 1,
        )
        self.estimator_.fit(
            X_train, y_train, **_validation_fit_params(X_valid, y_valid)
        )
        self.best_iteration_ = self.estimator_.booster_.best_iteration
        self.feature_importances_ = self.estimator_.feature_importances_
        return self

    def predict(self, X: ArrayLike) -> NDArray[Any]:
        """Predict with the best iteration found by early stopping.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted values.
        """
        check_is_fitted(self, "estimator_")
        return cast("NDArray[Any]", self.estimator_.predict(check_array(X)))
