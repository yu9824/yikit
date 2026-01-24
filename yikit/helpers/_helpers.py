import importlib.util
import inspect
import sys
from typing import Any, Generic, Optional, TypeVar

if sys.version_info >= (3, 9):
    from collections.abc import Callable, Iterable, Iterator
else:
    from typing import Callable, Iterable, Iterator
T = TypeVar("T")


def is_installed(package_name: str) -> bool:
    """
    Check whether a given Python package is installed.

    Uses `importlib.util.find_spec` to determine if the specified package
    can be imported.

    Parameters
    ----------
    package_name : str
        The name of the package (e.g., "sklearn").

    Returns
    -------
    bool
        True if the package is installed, False otherwise.
    """
    return bool(importlib.util.find_spec(package_name))


def is_argument(__callable: "Callable[..., Any]", arg_name: str) -> bool:
    """
    Check if a given argument name is present in the callable's signature.

    This function checks whether the specified argument name is part of the
    parameters in the callable's signature. It can be used to verify if a
    function or method accepts a specific argument.

    Parameters
    ----------
    __callable : Callable
        The callable (function or method) whose signature is inspected.
    arg_name : str
        The name of the argument to check for in the callable's signature.

    Returns
    -------
    bool
        True if the argument name is found in the callable's parameters, False otherwise.
    """
    return arg_name in set(inspect.signature(__callable).parameters.keys())


# HACK: when drop python3.8, use `dummy_tqdm(Iterable[T])`
class dummy_tqdm(Iterable, Generic[T]):
    """
    A dummy class that mimics the behavior of 'tqdm' for testing or placeholder purposes.

    This class allows you to use a tqdm-like interface in cases where the
    progress bar functionality is not needed or when testing code without
    depending on the actual `tqdm` library.

    Parameters
    ----------
    __iterable : Iterable[T]
        An iterable object that will be wrapped and returned by the class.

    Methods
    -------
    __iter__() -> Iterator[T]
        Returns an iterator for the provided iterable.
    __getattr__(name: str) -> Callable[..., None]
        Returns a no-operation function for unsupported attributes.
    """

    def __init__(
        self, __iterable: Optional["Iterable[T]"] = None, *args, **kwargs
    ) -> None:
        """
        Initialize the dummy tqdm wrapper.

        Parameters
        ----------
        __iterable : Iterable[T], optional
            An iterable object to wrap. If None, an empty tuple is used.
        *args
            Additional positional arguments (ignored, for tqdm compatibility).
        **kwargs
            Additional keyword arguments (ignored, for tqdm compatibility).
        """
        self.__iterable = __iterable if __iterable else ()

    def __iter__(self) -> "Iterator[T]":
        """
        Return an iterator for the given iterable.

        Returns
        -------
        Iterator[T]
            An iterator for the provided iterable object.
        """
        return iter(self.__iterable)

    def __getattr__(self, name: str) -> "Callable[..., None]":
        """
        Handle unsupported attribute access by returning a no-op function.

        This method allows the class to simulate the behavior of tqdm by
        returning a no-op function for any attribute that is not defined.

        Parameters
        ----------
        name : str
            The name of the attribute being accessed.

        Returns
        -------
        Callable[..., None]
            A no-op function that does nothing.
        """
        return self.__no_operation

    @staticmethod
    def __no_operation(*args, **kwargs) -> None:
        """
        A no-operation function used as a placeholder.

        This function does nothing and is used as a fallback for unsupported
        method calls or attributes.

        Returns
        -------
        None
        """
        return
