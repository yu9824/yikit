from __future__ import annotations

import importlib.util
import inspect
import sys
from typing import Any, Generic, TypeVar

import joblib

if sys.version_info >= (3, 9):
    from collections.abc import Callable, Iterable, Iterator
else:
    from typing import Callable, Iterable, Iterator
T = TypeVar("T")


def is_installed(package_name: str) -> bool:
    """Return whether a package can be imported.

    The package is looked up with :func:`importlib.util.find_spec`, so a
    top-level package is found without being imported.

    Parameters
    ----------
    package_name : str
        The name used to import the package, such as ``"sklearn"`` (not the
        name of the distribution, ``"scikit-learn"``).

    Returns
    -------
    bool
        True if the package is installed, False otherwise.

    Raises
    ------
    ModuleNotFoundError
        If ``package_name`` is a dotted name (such as ``"a.b"``) whose
        parent package is not installed. The parent packages of a dotted
        name are imported.

    Examples
    --------
    >>> from yikit.helpers import is_installed
    >>> is_installed("numpy")
    True
    >>> is_installed("yikit_package_that_does_not_exist")
    False
    """
    return bool(importlib.util.find_spec(package_name))


def is_argument(__callable: Callable[..., Any], arg_name: str) -> bool:
    """Return whether a callable has a parameter of the given name.

    The names of the parameters are read with :func:`inspect.signature`.
    Variable parameters count by their own names (``"kwargs"`` for
    ``**kwargs``); they do not make other names parameters.

    Parameters
    ----------
    __callable : callable
        The function, method or class to inspect (passed positionally). The
        signature of a bound method does not include ``self``, and that of a
        class is the signature of its constructor.
    arg_name : str
        The name of the parameter.

    Returns
    -------
    bool
        True if ``arg_name`` is the name of a parameter of ``__callable``,
        False otherwise.

    Raises
    ------
    TypeError
        If ``__callable`` is not callable.
    ValueError
        If no signature can be read from ``__callable`` (such as some
        built-in functions).

    Examples
    --------
    >>> from yikit.helpers import is_argument
    >>> def fit(X, y, sample_weight=None, **kwargs):
    ...     pass
    >>> is_argument(fit, "sample_weight")
    True
    >>> is_argument(fit, "eval_set")
    False
    """
    return arg_name in set(inspect.signature(__callable).parameters.keys())


# HACK: when drop python3.8, use `dummy_tqdm(Iterable[T])`
class dummy_tqdm(Iterable, Generic[T]):
    """Stand-in for ``tqdm`` that shows no progress bar.

    It replaces ``tqdm`` when tqdm is not installed or no progress bar is
    wanted. Iterating over it yields the items of the given iterable in
    order. Any other attribute, such as ``update``, ``close`` or
    ``set_postfix``, is a function that accepts any arguments and does
    nothing.

    Parameters
    ----------
    __iterable : iterable or None, default=None
        The iterable to iterate over, passed positionally (an ``iterable=``
        keyword is ignored like the other keyword arguments). None, or any
        other false value, means that nothing is iterated.
    *args
        Ignored; accepted for compatibility with ``tqdm``.
    **kwargs
        Ignored; accepted for compatibility with ``tqdm`` (such as
        ``total`` and ``desc``).

    Examples
    --------
    >>> from yikit.helpers import dummy_tqdm
    >>> list(dummy_tqdm([3, 1, 2], desc="fit"))
    [3, 1, 2]
    >>> progress_bar = dummy_tqdm(total=3)
    >>> progress_bar.update(1)
    >>> progress_bar.close()
    """

    def __init__(
        self, __iterable: Iterable[T] | None = None, *args, **kwargs
    ) -> None:
        """Store the iterable; the other arguments are ignored.

        Parameters
        ----------
        __iterable : iterable or None, default=None
            The iterable to iterate over. None, or any other false value, is
            replaced with an empty tuple.
        *args
            Ignored; accepted for compatibility with ``tqdm``.
        **kwargs
            Ignored; accepted for compatibility with ``tqdm``.
        """
        self.__iterable = __iterable if __iterable else ()

    def __iter__(self) -> Iterator[T]:
        """Return an iterator over the given iterable.

        Returns
        -------
        iterator
            An iterator over the iterable passed to the constructor.
        """
        return iter(self.__iterable)

    def __getattr__(self, name: str) -> Callable[..., None]:
        """Return a function that does nothing for any other attribute.

        Parameters
        ----------
        name : str
            The name of the attribute, such as ``"update"``.

        Returns
        -------
        callable
            A function that accepts any arguments and returns None.
        """
        return self.__no_operation

    @staticmethod
    def __no_operation(*args, **kwargs) -> None:
        """Accept any arguments and do nothing.

        Parameters
        ----------
        *args
            Ignored.
        **kwargs
            Ignored.
        """
        return


class tqdm_joblib:
    """
    Context manager for integrating tqdm progress bars with joblib.Parallel.

    This context manager ensures that a tqdm progress bar is created and safely closed
    when using joblib.Parallel for parallel processing. It temporarily overrides the
    joblib BatchCompletionCallBack to update the progress bar as batches complete.

    Parameters
    ----------
    total : int, optional
        The total number of tasks expected (passed to tqdm).
    **tqdm_kwargs
        Additional keyword arguments passed directly to tqdm.

    Example
    -------
    >>> with tqdm_joblib(total=100) as pbar:
    ...     Parallel(n_jobs=2)(
    ...         delayed(process)(i) for i in range(100)
    ...     )
    ...     # Progress is shown via tqdm
    """

    def __init__(self, total: int, silent: bool = False, **tqdm_kwargs):
        """
        Initialize the tqdm_joblib context manager.

        Parameters
        ----------
        total : int
            The total number of tasks expected (passed to tqdm).
        silent : bool, optional
            If True, suppress progress bar output even if tqdm is installed.
            Default is False.
        **tqdm_kwargs
            Additional keyword arguments passed directly to tqdm (e.g., desc, unit).
        """
        self.total = total
        self.silent = silent
        self.tqdm_kwargs = tqdm_kwargs
        self.tqdm = None
        self._old_batch_callback = None

    def __enter__(self):
        """
        Enter the context manager and set up the progress bar.

        Returns
        -------
        tqdm or dummy_tqdm
            The progress bar object (tqdm if available and not silent,
            otherwise dummy_tqdm).

        Notes
        -----
        This method temporarily overrides joblib's BatchCompletionCallBack
        to update the progress bar as batches complete.
        """
        if is_installed("tqdm") and not self.silent:
            from tqdm.auto import tqdm
        else:
            tqdm = dummy_tqdm

        # Open (create) tqdm progress bar
        self.tqdm = tqdm(total=self.total, **self.tqdm_kwargs)
        tqdm_object = self.tqdm

        class TqdmBatchCompletionCallback(
            joblib.parallel.BatchCompletionCallBack
        ):
            def __call__(self, *args, **kwargs):
                tqdm_object.update(n=self.batch_size)
                return super().__call__(*args, **kwargs)

        # Save the original BatchCompletionCallBack and override with the custom one
        self._old_batch_callback = joblib.parallel.BatchCompletionCallBack
        joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback

        return self.tqdm

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context manager and clean up resources.

        Parameters
        ----------
        exc_type : type, optional
            Exception type if an exception was raised.
        exc_val : Exception, optional
            Exception value if an exception was raised.
        exc_tb : traceback, optional
            Exception traceback if an exception was raised.

        Returns
        -------
        bool
            Always returns False to propagate exceptions.

        Notes
        -----
        This method restores the original BatchCompletionCallBack and closes
        the progress bar, regardless of whether an exception occurred.
        """
        # Restore the original BatchCompletionCallBack, regardless of exceptions
        joblib.parallel.BatchCompletionCallBack = self._old_batch_callback

        # Safely close the tqdm progress bar if it was created
        if self.tqdm is not None:
            self.tqdm.close()

        # Do not suppress exceptions; propagate them as usual
        return False
