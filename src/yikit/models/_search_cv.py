"""``OptunaSearchCV`` that scikit-learn ensembles accept as a regressor.

``OptunaSearchRegressor`` is ``OptunaSearchCV`` of optuna-integration with
the marks of a regressor added, so that ``VotingRegressor`` and
``StackingRegressor`` accept it, with the number and the names of the
features of its best estimator exposed, and with the trial logs of optuna
hidden while it is fitted with ``verbose=0``. It is internal to yikit and is
not exported from ``yikit.models``. This module needs optuna-integration
(or an old optuna that still bundles ``optuna.integration``), so it is
imported only when it is used.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import optuna

try:
    from optuna_integration import OptunaSearchCV
except ImportError:  # old optuna that still bundles the integration
    from optuna.integration import OptunaSearchCV

if TYPE_CHECKING:
    from collections.abc import Iterator


class _OptunaInfoLogsHider:
    """Hide the INFO logs of optuna while at least one search runs.

    The verbosity of optuna is global to the process, and the searches of an
    ensemble may run at the same time in threads (e.g. with the threading
    backend of joblib). The searches are counted under a lock: the first one
    to enter saves the verbosity and raises it to ``optuna.logging.WARNING``
    (never lowers it), and the last one to leave restores the saved value.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._depth = 0
        self._previous = optuna.logging.WARNING

    @contextmanager
    def hidden(self) -> Iterator[None]:
        """Hide the INFO logs of optuna in the ``with`` block.

        Yields
        ------
        None
            The verbosity is restored when the last block exits, also when
            it raises.
        """
        with self._lock:
            if self._depth == 0:
                self._previous = optuna.logging.get_verbosity()
                optuna.logging.set_verbosity(
                    max(self._previous, optuna.logging.WARNING)
                )
            self._depth += 1
        try:
            yield
        finally:
            with self._lock:
                self._depth -= 1
                if self._depth == 0:
                    optuna.logging.set_verbosity(self._previous)


#: Shared by all the searches of the process.
_INFO_LOGS_HIDER = _OptunaInfoLogsHider()


class OptunaSearchRegressor(OptunaSearchCV):
    """``OptunaSearchCV`` that scikit-learn ensembles accept as a regressor.

    ``sklearn.base.is_regressor`` reads the estimator tags
    (``__sklearn_tags__``) from scikit-learn 1.6, and ``OptunaSearchCV``
    does not define them, so ``VotingRegressor`` and ``StackingRegressor``
    reject it ("should be a regressor"). This class marks it as a
    regressor (``_estimator_type`` for scikit-learn < 1.6 and the
    ``estimator_type`` tag for scikit-learn >= 1.6) and defines
    ``predict`` as a plain method, which old versions of optuna define as a
    property (see ``predict``). ``RegressorMixin`` is not mixed in, so
    ``score`` keeps using ``scoring`` as in ``OptunaSearchCV``. The
    constructor is not overridden, so the parameters, ``get_params``,
    ``set_params`` and ``sklearn.base.clone`` are those of
    ``OptunaSearchCV``. It is defined at the top level of its module so
    that fitted ensembles that contain it can be pickled.

    The parameters and the fitted attributes (``study_``,
    ``best_params_``, ``best_estimator_``, ...) are those of
    ``OptunaSearchCV``. In addition, ``n_features_in_`` and
    ``feature_names_in_`` are read from ``best_estimator_`` (so that the
    ensembles that contain the search have them after ``fit``), and ``fit``
    hides the trial logs of optuna when ``verbose=0``. The class is
    internal to yikit (``EnsembleRegressor`` wraps each model with it when
    ``opt=True``) and is not exported from ``yikit.models``.

    See Also
    --------
    optuna_integration.OptunaSearchCV : The search that this class marks.
    """

    _estimator_type = "regressor"

    def __sklearn_tags__(self) -> Any:
        """Return the tags of ``OptunaSearchCV`` marked as a regressor.

        Only scikit-learn >= 1.6 calls this method.

        Returns
        -------
        sklearn.utils.Tags
            The tags of the parent class with ``estimator_type`` set to
            ``"regressor"``.
        """
        tags = super().__sklearn_tags__()
        tags.estimator_type = "regressor"
        return tags

    def fit(
        self,
        X: Any,
        y: Any = None,
        groups: Any = None,
        **fit_params: Any,
    ) -> OptunaSearchRegressor:
        """Run the search of ``OptunaSearchCV``, hiding the trial logs.

        With ``verbose=0`` (or smaller), the verbosity of optuna is raised
        to ``optuna.logging.WARNING`` (never lowered) during the search, so
        that optuna does not log every trial, and it is restored afterwards,
        also when the search raises. This is done in ``fit`` so that it also
        works in the parallel workers of the ensembles. The verbosity is
        global to the process, so searches that run at the same time in
        threads share it: it stays raised until the last of them finishes,
        which restores the verbosity from before the first one started (a
        change made by another thread in the meantime is overwritten, and a
        search with ``verbose > 0`` that runs at the same time also has its
        logs hidden). With ``verbose > 0`` the verbosity of optuna is not
        changed.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,) or None, default=None
            Target values.
        groups : array-like of shape (n_samples,) or None, default=None
            Group labels for the splits of the cross-validation, passed to
            ``fit`` of ``OptunaSearchCV``.
        **fit_params : dict
            Passed to ``fit`` of ``OptunaSearchCV``.

        Returns
        -------
        self : OptunaSearchRegressor
            The fitted search.
        """
        if self.verbose > 0:
            super().fit(X, y, groups=groups, **fit_params)
        else:
            with _INFO_LOGS_HIDER.hidden():
                super().fit(X, y, groups=groups, **fit_params)
        return self

    def predict(self, X: Any, **kwargs: Any) -> Any:
        """Predict with the best estimator found by the search.

        ``OptunaSearchCV`` of optuna 3.0 to 3.6 (``optuna.integration``) and
        of optuna-integration 4.0.0 or older defines ``predict`` as a
        property that raises ``NotFittedError`` (an ``AttributeError``)
        before ``fit``, so ``hasattr(unfitted_search, "predict")`` is False
        and ``StackingRegressor`` rejects the search ("does not implement
        the method predict"). This plain method is always present; it calls
        ``predict`` of ``OptunaSearchCV``, which works whether the parent
        defines it as a method or as a property that returns
        ``best_estimator_.predict``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.
        **kwargs : dict
            Passed to ``predict`` of the best estimator.

        Returns
        -------
        ndarray of shape (n_samples,) or (n_samples, n_outputs)
            Predicted values.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the search is not fitted yet.
        """
        return super().predict(X, **kwargs)

    def _from_best_estimator(self, name: str) -> Any:
        """Return the attribute ``name`` of ``best_estimator_``.

        Raises
        ------
        AttributeError
            If the search has no ``best_estimator_`` (it is not fitted, or it
            was fitted with ``refit=False``), or if the best estimator has no
            such attribute, so that ``hasattr`` is False.
        """
        try:
            best_estimator = self.best_estimator_
        except AttributeError as exc:
            raise AttributeError(
                f"{type(self).__name__} object has no attribute {name!r}: "
                "it has no best_estimator_ (it is not fitted yet, or it was "
                "fitted with refit=False)."
            ) from exc
        return getattr(best_estimator, name)

    @property
    def n_features_in_(self) -> int:
        """Number of features seen by ``best_estimator_`` during ``fit``.

        ``OptunaSearchCV`` does not have it, and ``VotingRegressor`` and
        ``StackingRegressor`` read it from their first estimator. It raises
        ``AttributeError`` (so ``hasattr`` is False) before ``fit``, with
        ``refit=False``, or when the best estimator does not have it.
        """
        return self._from_best_estimator("n_features_in_")

    @property
    def feature_names_in_(self) -> Any:
        """Names of the features seen by ``best_estimator_`` during ``fit``.

        ndarray of shape (``n_features_in_``,). Defined only when the best
        estimator has it (scikit-learn >= 1.0, fitted on data whose features
        all have string names, such as a ``pandas.DataFrame``); otherwise it
        raises ``AttributeError``, as ``n_features_in_`` does.
        ``VotingRegressor`` and ``StackingRegressor`` copy it from their
        estimators.
        """
        return self._from_best_estimator("feature_names_in_")
