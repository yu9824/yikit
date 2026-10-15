"""Private helpers to validate the input of the estimators in yikit.

These helpers are not part of the public API of :mod:`yikit.helpers`; they
are imported by the estimators (such as ``FilterSelector``) from this module
directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def feature_names_of(X: Any) -> NDArray[Any] | None:
    """Return the column names of ``X`` to store as ``feature_names_in_``.

    As in scikit-learn, the names are taken from ``X.columns`` (such as the
    columns of a ``pandas.DataFrame``) only when every name is exactly of
    type ``str``; subclasses such as ``numpy.str_`` do not count.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The input data. It may have a ``columns`` attribute.

    Returns
    -------
    numpy.ndarray of shape (n_features,) or None
        The column names as an array of ``dtype=object``. None if ``X`` has
        no ``columns``, if there are no columns, or if any name is not a
        ``str``.

    Examples
    --------
    >>> import pandas as pd
    >>> from yikit.helpers._validation import feature_names_of
    >>> X = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], columns=["a", "b"])
    >>> feature_names_of(X)
    array(['a', 'b'], dtype=object)
    >>> print(feature_names_of(X.to_numpy()))
    None
    >>> print(feature_names_of(pd.DataFrame([[1.0, 2.0]], columns=[0, 1])))
    None
    """
    columns = getattr(X, "columns", None)
    if columns is None:
        return None
    names = np.array(columns, dtype=object)
    if names.ndim != 1 or len(names) == 0:
        return None
    if not all(type(name) is str for name in names):
        return None
    return names
