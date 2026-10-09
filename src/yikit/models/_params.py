"""Apply parameters to copies of estimators without changing the originals.

``apply_params`` is how yikit sets the parameters of a trial on the
estimator passed by the user. It accepts the prefixed names returned by the
search space (``svr__C``, ``regressor__svr__C``, ``Base__max_depth``) and
always works on a copy, so the estimator passed by the user is never
modified and the copy shares no estimator or ``RandomState`` with it.

An estimator whose constructor parameters change is rebuilt from its class
with the parameters of its copy, in the same way as ``sklearn.base.clone``
builds a copy, instead of relying on its ``set_params``. This also works
for models whose own ``set_params`` differs from scikit-learn: since
ngboost 0.4.0, ``set_params`` of ``NGBRegressor`` only calls ``setattr``,
so nested names such as ``Base__max_depth`` do not reach its ``Base`` and
an integer ``random_state`` is not turned into a ``RandomState``. The state
other than the parameters that ``clone`` carries over (the ``set_output``
configuration, metadata routing requests and callbacks) is moved to the
rebuilt estimator.
"""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING, Any

from sklearn.base import clone

if TYPE_CHECKING:
    from collections.abc import Mapping

#: Attributes other than the parameters that ``sklearn.base.clone`` carries
#: over to its copy: the configuration of ``set_output`` (scikit-learn >=
#: 1.2), the metadata routing requests (>= 1.3) and the callbacks (>= 1.9).
#: They are private to scikit-learn, and are moved only when present.
_CLONED_STATE_ATTRIBUTES = (
    "_sklearn_output_config",
    "_metadata_request",
    "_skl_callbacks",
)


def _is_estimator(value: Any) -> bool:
    """Return whether ``value`` is an estimator instance (not a class)."""
    return hasattr(value, "get_params") and not isinstance(value, type)


def _constructor_param_names(estimator: Any) -> frozenset[str] | None:
    """Return the names accepted by the constructor of ``estimator``.

    Parameters
    ----------
    estimator : object
        Estimator instance whose class is inspected.

    Returns
    -------
    frozenset of str or None
        Names of the keyword parameters of ``type(estimator).__init__``,
        excluding ``self``. ``None`` when the constructor takes ``**kwargs``
        (e.g. LightGBM), which means that every name is accepted.
    """
    init = type(estimator).__init__
    if init is object.__init__:
        return frozenset()
    parameters = inspect.signature(init).parameters.values()
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters):
        return None
    return frozenset(
        p.name
        for p in parameters
        if p.name != "self"
        and p.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    )


def _split_params(
    params: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Split names into direct ones and nested ones grouped by their head.

    Parameters
    ----------
    params : Mapping of str to object
        Parameter names and values, e.g. ``{"C": 1.0, "svr__C": 2.0}``.

    Returns
    -------
    direct : dict of str to object
        Parameters whose names do not contain ``"__"``.
    nested : dict of str to dict of str to object
        The other parameters grouped by the name before the first ``"__"``,
        with that name and the separator removed, e.g.
        ``{"svr": {"C": 2.0}}``.
    """
    direct: dict[str, Any] = {}
    nested: dict[str, dict[str, Any]] = {}
    for name, value in params.items():
        head, separator, rest = name.partition("__")
        if separator:
            nested.setdefault(head, {})[rest] = value
        else:
            direct[name] = value
    return direct, nested


def _move_cloned_state(source: Any, target: Any) -> Any:
    """Move the state that ``clone`` carries over from ``source`` to ``target``.

    Only the attributes in ``_CLONED_STATE_ATTRIBUTES`` that ``source`` has
    in its instance dictionary are moved. ``source`` must be a copy made by
    ``clone`` that is discarded afterwards, so the moved objects are not
    shared with any other estimator (``clone`` itself shares only the
    callbacks).

    Parameters
    ----------
    source : object
        Discarded copy of the estimator.
    target : object
        Estimator rebuilt from ``source``. It is modified in place.

    Returns
    -------
    object
        ``target``.
    """
    state = getattr(source, "__dict__", {})
    for name in _CLONED_STATE_ATTRIBUTES:
        if name in state:
            setattr(target, name, state[name])
    return target


def _set_values(estimator: Any, values: Mapping[str, Any]) -> Any:
    """Set top-level ``values`` on ``estimator``, a discardable copy.

    Constructor parameters (every name when the constructor takes
    ``**kwargs``) are set by rebuilding the estimator from its class, and
    the state that ``clone`` carries over is moved to the rebuilt estimator.
    The other names, such as the step names of a ``Pipeline``, are set with
    ``set_params``.

    Parameters
    ----------
    estimator : object
        Copy of an estimator that nothing else refers to. It may be
        modified.
    values : Mapping of str to object
        Names without ``"__"`` that are known to ``estimator``, and their
        values, which are not shared with anything else.

    Returns
    -------
    object
        ``estimator`` or the estimator rebuilt from it, with ``values`` set.
    """
    if not values:
        return estimator
    constructor_names = _constructor_param_names(estimator)
    init_values = {
        name: value
        for name, value in values.items()
        if constructor_names is None or name in constructor_names
    }
    other_values = {
        name: value
        for name, value in values.items()
        if name not in init_values
    }
    rebuilt = (
        _move_cloned_state(
            estimator,
            type(estimator)(
                **{**estimator.get_params(deep=False), **init_values}
            ),
        )
        if init_values
        else estimator
    )
    return rebuilt.set_params(**other_values) if other_values else rebuilt


def _apply_params(
    estimator: Any, params: Mapping[str, Any], prefix: str
) -> Any:
    """Apply ``params`` to a copy of ``estimator``.

    The direct names are set first, and the nested names are then applied
    to the inner estimators of the result, in the same order as
    ``set_params`` of scikit-learn.

    Parameters
    ----------
    estimator : object
        Estimator instance to copy. It is not modified.
    params : Mapping of str to object
        Parameter names relative to ``estimator`` and their values.
    prefix : str
        Names on the way from the outermost estimator, joined with ``"__"``,
        used only in error messages.

    Returns
    -------
    object
        Unfitted copy of ``estimator`` with ``params`` applied.

    Raises
    ------
    ValueError
        If a name is not a parameter of the targeted estimator, or if a
        nested name targets a parameter that does not hold an estimator.
    """
    copied = clone(estimator)
    if not params:
        return copied
    type_name = type(copied).__name__
    direct, nested = _split_params(params)

    constructor_names = _constructor_param_names(copied)
    deep_params = copied.get_params(deep=True)
    unknown = [
        name
        for name in direct
        if constructor_names is not None
        and name not in constructor_names
        and name not in deep_params
    ]
    if unknown:
        valid_names = sorted(
            (constructor_names or frozenset())
            | {key for key in deep_params if "__" not in key}
        )
        raise ValueError(
            f"Invalid parameter {prefix + unknown[0]!r}: {type_name} has no "
            f"parameter {unknown[0]!r}. Valid parameters are: {valid_names}."
        )

    # The values are cloned (``clone(..., safe=False)`` deep-copies
    # non-estimators) so that the copy shares no estimator or RandomState
    # with the caller, nor between two names given the same object.
    with_direct = _set_values(
        copied,
        {name: clone(value, safe=False) for name, value in direct.items()},
    )

    inner_estimators = with_direct.get_params(deep=True)
    nested_values: dict[str, Any] = {}
    for head, sub_params in nested.items():
        full_name = f"{prefix}{head}__{next(iter(sub_params))}"
        if head not in inner_estimators:
            raise ValueError(
                f"Invalid parameter {full_name!r}: {type_name} has no "
                f"parameter {head!r} holding an estimator."
            )
        inner = inner_estimators[head]
        if not _is_estimator(inner):
            raise ValueError(
                f"Invalid parameter {full_name!r}: the parameter {head!r} of "
                f"{type_name} is {inner!r}, not an estimator."
            )
        nested_values[head] = _apply_params(
            inner, sub_params, prefix=f"{prefix}{head}__"
        )
    return _set_values(with_direct, nested_values)


def apply_params(estimator: Any, params: Mapping[str, Any]) -> Any:
    """Return an unfitted copy of ``estimator`` with ``params`` applied.

    ``estimator`` is copied with ``sklearn.base.clone`` and never modified.
    As with ``set_params`` of scikit-learn, the names without ``"__"`` are
    set first. Nested names such as ``svr__C`` or ``regressor__svr__C`` are
    then applied recursively to a copy of the inner estimator, which becomes
    the value of the outer parameter. Parameters of the constructor are set
    by rebuilding the estimator from its class, so that models whose own
    ``set_params`` differs from scikit-learn (ngboost's ``NGBRegressor``)
    get them too; the ``set_output`` configuration, metadata routing
    requests and callbacks that ``clone`` carries over are kept. The other
    names found in ``get_params(deep=True)``, such as the step names of a
    ``Pipeline``, are set with ``set_params``.

    Parameters
    ----------
    estimator : object
        Estimator instance to copy, e.g. ``sklearn.svm.SVR()`` or a
        ``Pipeline`` whose last step is one. It is not modified.
    params : Mapping of str to object
        Parameter names, prefixed for nested estimators as accepted by
        ``set_params`` of scikit-learn, and their values. The values are
        copied, so the returned estimator shares no estimator or
        ``RandomState`` with them.

    Returns
    -------
    object
        Unfitted copy of ``estimator``. Its parameters are the values in
        ``params`` and, for the others, equal to those of ``estimator``. It
        shares no estimator or ``RandomState`` with ``estimator``.

    Raises
    ------
    ValueError
        If a name is not a parameter of the targeted estimator, or if a
        nested name targets a parameter that does not hold an estimator.

    Examples
    --------
    >>> from sklearn.compose import TransformedTargetRegressor
    >>> from sklearn.pipeline import make_pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> from sklearn.svm import SVR
    >>> estimator = TransformedTargetRegressor(
    ...     regressor=make_pipeline(StandardScaler(), SVR())
    ... )
    >>> copied = apply_params(estimator, {"regressor__svr__C": 10.0})
    >>> copied.regressor.named_steps["svr"].C
    10.0
    >>> estimator.regressor.named_steps["svr"].C
    1.0
    """
    return _apply_params(estimator, params, prefix="")
