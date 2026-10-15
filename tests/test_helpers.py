"""Tests of ``yikit.helpers`` and ``yikit.helpers._validation``."""

from __future__ import annotations

import inspect
import io
import sys
import threading

import numpy as np
import pandas as pd
import pytest
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

import yikit.helpers
from yikit.helpers import dummy_tqdm, is_argument, is_installed, tqdm_joblib
from yikit.helpers._validation import feature_names_of

#: Small data shared by the tests of ``feature_names_of``.
X = np.arange(12, dtype=float).reshape(4, 3)
COLUMNS = ["a", "b", "c"]


def _frame(columns: list) -> pd.DataFrame:
    """Return ``X`` as a DataFrame whose column names are kept as given."""
    # ``dtype=object`` keeps the names as they are (pandas may otherwise
    # convert ``numpy.str_`` to ``str`` when it infers a string dtype).
    return pd.DataFrame(X, columns=pd.Index(columns, dtype=object))


# --- Public names ------------------------------------------------------------


def test_public_names_are_unchanged():
    assert tuple(yikit.helpers.__all__) == (
        "dummy_tqdm",
        "is_argument",
        "is_installed",
        "tqdm_joblib",
    )
    assert not hasattr(yikit.helpers, "feature_names_of")


# --- is_installed ------------------------------------------------------------


def test_is_installed_takes_one_package_name():
    # yikit.models (and the tests that replace it) rely on this signature.
    assert list(inspect.signature(is_installed).parameters) == ["package_name"]


@pytest.mark.parametrize("package_name", ["numpy", "sklearn", "yikit"])
def test_is_installed_returns_true_for_an_installed_package(package_name):
    assert is_installed(package_name) is True


def test_is_installed_returns_false_for_a_missing_package():
    assert is_installed("yikit_package_that_does_not_exist") is False


# --- is_argument -------------------------------------------------------------


def _function(alpha, beta=1, *args, gamma=None, **kwargs):
    """Take parameters of every kind (used by the tests of is_argument)."""


@pytest.mark.parametrize("arg_name", ["alpha", "beta", "gamma"])
def test_is_argument_returns_true_for_a_parameter(arg_name):
    assert is_argument(_function, arg_name) is True


@pytest.mark.parametrize("arg_name", ["delta", "Alpha", ""])
def test_is_argument_returns_false_for_a_missing_parameter(arg_name):
    # ``**kwargs`` does not make every name a parameter.
    assert is_argument(_function, arg_name) is False


def test_is_argument_inspects_classes_and_bound_methods():
    assert is_argument(LinearRegression, "fit_intercept") is True
    assert is_argument(LinearRegression().fit, "sample_weight") is True
    assert is_argument(LinearRegression().fit, "eval_set") is False
    # A bound method does not show ``self``.
    assert is_argument(LinearRegression().fit, "self") is False


# --- dummy_tqdm --------------------------------------------------------------


@pytest.mark.parametrize(
    ("make_iterable", "expected"),
    [
        (lambda: [3, 1, 2], [3, 1, 2]),
        (lambda: (3, 1, 2), [3, 1, 2]),
        (lambda: range(3), [0, 1, 2]),
        (lambda: (value for value in [3, 1, 2]), [3, 1, 2]),
        (lambda: iter([3, 1, 2]), [3, 1, 2]),
        (lambda: {"b": 1, "a": 2}, ["b", "a"]),
    ],
    ids=["list", "tuple", "range", "generator", "iterator", "dict"],
)
def test_dummy_tqdm_iterates_the_iterable_in_order(make_iterable, expected):
    assert list(dummy_tqdm(make_iterable())) == expected


def test_dummy_tqdm_ignores_the_arguments_of_tqdm():
    progress_bar = dummy_tqdm([1, 2], desc="fit", total=2, leave=False)

    assert list(progress_bar) == [1, 2]


@pytest.mark.parametrize(
    "progress_bar",
    [dummy_tqdm(), dummy_tqdm(total=3)],
    ids=["no-arguments", "total"],
)
def test_dummy_tqdm_without_an_iterable_is_empty(progress_bar):
    assert list(progress_bar) == []


@pytest.mark.parametrize(
    "method",
    ["update", "close", "refresh", "reset", "set_description", "set_postfix"],
)
def test_dummy_tqdm_accepts_calls_and_does_nothing(method):
    progress_bar = dummy_tqdm([1, 2, 3], total=3)

    assert getattr(progress_bar, method)() is None
    assert getattr(progress_bar, method)(1, n=2, desc="fit") is None
    # The calls do not change the iteration.
    assert list(progress_bar) == [1, 2, 3]


def test_dummy_tqdm_accepts_update_while_iterating():
    progress_bar = dummy_tqdm(["x", "y"])
    seen = []
    for value in progress_bar:
        progress_bar.update(1)
        seen.append(value)
    progress_bar.close()

    assert seen == ["x", "y"]


# --- tqdm_joblib -------------------------------------------------------------

#: The methods of ``Parallel`` that ``tqdm_joblib`` replaces while active.
ORIGINAL_CALL = Parallel.__call__
ORIGINAL_PRINT_PROGRESS = Parallel.print_progress

#: Seconds to wait for another thread before a test fails (never reached
#: unless the test is broken).
TIMEOUT = 60

SEQUENTIAL = {"n_jobs": 1}
THREADING = {"n_jobs": 2, "backend": "threading"}


class _Record:
    """Record the steps and the closing of a progress bar."""

    def __init__(self, progress_bar) -> None:
        self.steps: list[int] = []
        self.n_closed = 0
        update = progress_bar.update
        close = progress_bar.close

        def recording_update(n=1):
            self.steps.append(n)
            return update(n)

        def recording_close():
            self.n_closed += 1
            return close()

        progress_bar.update = recording_update
        progress_bar.close = recording_close

    def assert_advanced_to(self, total: int) -> None:
        """Check that the bar only moved forward and reached ``total``."""
        assert self.steps
        assert all(step > 0 for step in self.steps)
        assert sum(self.steps) == total


def _square(value: int) -> int:
    return value * value


def _squares(n_tasks: int, **parallel_kwargs) -> list[int]:
    """Square ``range(n_tasks)`` with a new ``Parallel``."""
    return Parallel(**parallel_kwargs)(
        delayed(_square)(value) for value in range(n_tasks)
    )


def _sum_of_inner_squares(value: int, parallel_kwargs: dict) -> int:
    """Run another ``Parallel`` inside a task."""
    return sum(_squares(value + 1, **parallel_kwargs))


def _fail_at(value: int, failing_value: int) -> int:
    if value == failing_value:
        raise ValueError(f"task {value} failed")
    return value


def _count_in_own_context(n_tasks: int) -> int:
    """Count a ``Parallel`` with a ``tqdm_joblib`` entered inside a task."""
    with tqdm_joblib(total=n_tasks, silent=True) as progress_bar:
        record = _Record(progress_bar)
        _squares(n_tasks, **SEQUENTIAL)
    return sum(record.steps)


def _parallel_is_restored() -> bool:
    return (
        Parallel.__call__ is ORIGINAL_CALL
        and Parallel.print_progress is ORIGINAL_PRINT_PROGRESS
    )


def _wait(event: threading.Event) -> None:
    if not event.wait(timeout=TIMEOUT):
        raise TimeoutError("the other thread did not reach its step")


@pytest.mark.parametrize(
    "parallel_kwargs",
    [{"n_jobs": None}, SEQUENTIAL, {"n_jobs": 2}, THREADING],
    ids=["None", "1", "2", "2-threading"],
)
def test_tqdm_joblib_advances_the_bar_to_total(parallel_kwargs):
    pytest.importorskip("tqdm")
    n_tasks = 6

    with tqdm_joblib(total=n_tasks, file=io.StringIO()) as progress_bar:
        record = _Record(progress_bar)
        results = _squares(n_tasks, **parallel_kwargs)

    assert results == [value * value for value in range(n_tasks)]
    # With n_jobs=2 the bar may move by batches.
    record.assert_advanced_to(n_tasks)
    assert progress_bar.n == n_tasks
    assert record.n_closed == 1
    assert _parallel_is_restored()


@pytest.mark.parametrize("n_jobs", [None, 1], ids=["None", "1"])
def test_tqdm_joblib_advances_one_per_task_when_sequential(n_jobs):
    # The bar is a tqdm, or a dummy_tqdm when tqdm is not installed.
    with tqdm_joblib(total=5, file=io.StringIO()) as progress_bar:
        record = _Record(progress_bar)
        _squares(5, n_jobs=n_jobs)

    assert record.steps == [1] * 5


@pytest.mark.parametrize(
    "inner_kwargs", [SEQUENTIAL, THREADING], ids=["inner-1", "inner-2"]
)
@pytest.mark.parametrize(
    "outer_kwargs", [SEQUENTIAL, THREADING], ids=["outer-1", "outer-2"]
)
def test_tqdm_joblib_does_not_count_parallel_inside_tasks(
    outer_kwargs, inner_kwargs
):
    with tqdm_joblib(total=4, silent=True) as progress_bar:
        record = _Record(progress_bar)
        results = Parallel(**outer_kwargs)(
            delayed(_sum_of_inner_squares)(value, inner_kwargs)
            for value in range(4)
        )

    assert results == [0, 1, 5, 14]
    record.assert_advanced_to(4)
    if outer_kwargs == SEQUENTIAL:
        assert record.steps == [1] * 4


@pytest.mark.parametrize(
    "parallel_kwargs", [SEQUENTIAL, THREADING], ids=["1", "2"]
)
def test_tqdm_joblib_does_not_count_generator_calls(parallel_kwargs):
    if not is_argument(Parallel, "return_as"):
        pytest.skip("joblib < 1.3 has no return_as")

    with tqdm_joblib(total=3, silent=True) as progress_bar:
        record = _Record(progress_bar)
        results = Parallel(**parallel_kwargs, return_as="generator")(
            delayed(_square)(value) for value in range(4)
        )
        assert list(results) == [0, 1, 4, 9]
        assert record.steps == []
        # A later call that returns a list is counted.
        _squares(3, **parallel_kwargs)

    record.assert_advanced_to(3)


def test_tqdm_joblib_replaces_parallel_only_while_active():
    assert _parallel_is_restored()

    with tqdm_joblib(total=1, silent=True):
        assert Parallel.__call__ is not ORIGINAL_CALL
        assert Parallel.print_progress is not ORIGINAL_PRINT_PROGRESS

    assert _parallel_is_restored()


@pytest.mark.parametrize(
    "parallel_kwargs", [SEQUENTIAL, THREADING], ids=["1", "2"]
)
def test_tqdm_joblib_restores_parallel_after_a_failed_task(parallel_kwargs):
    with pytest.raises(ValueError, match="task 2 failed"):
        with tqdm_joblib(total=5, silent=True) as progress_bar:
            record = _Record(progress_bar)
            Parallel(**parallel_kwargs)(
                delayed(_fail_at)(value, 2) for value in range(5)
            )

    assert _parallel_is_restored()
    assert record.n_closed == 1
    assert all(step > 0 for step in record.steps)
    assert sum(record.steps) <= 4


def test_tqdm_joblib_restores_parallel_after_an_exception_in_the_body():
    with pytest.raises(KeyError, match="body"):
        with tqdm_joblib(total=1, silent=True) as progress_bar:
            record = _Record(progress_bar)
            raise KeyError("body")

    assert _parallel_is_restored()
    assert record.n_closed == 1


def test_tqdm_joblib_keeps_counting_after_a_failed_call():
    with tqdm_joblib(total=5, silent=True) as progress_bar:
        record = _Record(progress_bar)
        with pytest.raises(ValueError, match="task 2 failed"):
            Parallel(n_jobs=1)(
                delayed(_fail_at)(value, 2) for value in range(5)
            )
        _squares(3, **SEQUENTIAL)

    # Tasks 0 and 1 of the failed call and the 3 tasks of the next call.
    assert record.steps == [1] * 5


def test_tqdm_joblib_counts_with_the_inner_context_when_nested():
    with tqdm_joblib(total=5, silent=True) as outer_bar:
        outer = _Record(outer_bar)
        _squares(3, **THREADING)
        with tqdm_joblib(total=4, silent=True) as inner_bar:
            inner = _Record(inner_bar)
            _squares(4, **SEQUENTIAL)
        assert inner.n_closed == 1
        assert sum(outer.steps) == 3
        assert not _parallel_is_restored()
        _squares(2, **SEQUENTIAL)

    outer.assert_advanced_to(5)
    inner.assert_advanced_to(4)
    assert outer.n_closed == 1
    assert _parallel_is_restored()


@pytest.mark.parametrize(
    "parallel_kwargs", [SEQUENTIAL, THREADING], ids=["1", "2"]
)
def test_tqdm_joblib_entered_inside_a_task_counts_its_own_calls(
    parallel_kwargs,
):
    with tqdm_joblib(total=3, silent=True) as progress_bar:
        record = _Record(progress_bar)
        counts = Parallel(**parallel_kwargs)(
            delayed(_count_in_own_context)(2) for _ in range(3)
        )

    assert counts == [2, 2, 2]
    record.assert_advanced_to(3)
    assert _parallel_is_restored()


def test_tqdm_joblib_in_two_threads_at_once():
    first_entered = threading.Event()
    second_entered = threading.Event()
    first_exited = threading.Event()
    records: dict[str, _Record] = {}
    patched_after_first_exited: list[bool] = []
    errors: list[BaseException] = []

    def first() -> None:
        with tqdm_joblib(total=4, silent=True) as progress_bar:
            records["first"] = _Record(progress_bar)
            first_entered.set()
            _wait(second_entered)
            _squares(4, **THREADING)
        first_exited.set()

    def second() -> None:
        _wait(first_entered)
        with tqdm_joblib(total=5, silent=True) as progress_bar:
            records["second"] = _Record(progress_bar)
            second_entered.set()
            _squares(3, **SEQUENTIAL)
            _wait(first_exited)
            patched_after_first_exited.append(not _parallel_is_restored())
            _squares(2, **SEQUENTIAL)

    def run(target) -> None:
        try:
            target()
        except BaseException as error:  # reported by the main thread
            errors.append(error)
            # Let the other thread go on instead of waiting until timeout.
            for event in (first_entered, second_entered, first_exited):
                event.set()

    threads = [
        threading.Thread(target=run, args=(target,))
        for target in (first, second)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=TIMEOUT)

    assert not any(thread.is_alive() for thread in threads)
    assert errors == []
    records["first"].assert_advanced_to(4)
    assert records["second"].steps == [1] * 5
    assert all(record.n_closed == 1 for record in records.values())
    # The second context still needed the replaced methods.
    assert patched_after_first_exited == [True]
    assert _parallel_is_restored()


def test_tqdm_joblib_with_silent_uses_dummy_tqdm():
    with tqdm_joblib(total=3, silent=True, desc="squares") as progress_bar:
        assert isinstance(progress_bar, dummy_tqdm)
        results = _squares(3, **SEQUENTIAL)

    assert results == [0, 1, 4]
    assert _parallel_is_restored()


def test_tqdm_joblib_without_tqdm_uses_dummy_tqdm(monkeypatch):
    # A None entry in sys.modules makes the package look missing.
    monkeypatch.setitem(sys.modules, "tqdm", None)

    with tqdm_joblib(total=3) as progress_bar:
        assert isinstance(progress_bar, dummy_tqdm)
        record = _Record(progress_bar)
        _squares(3, **SEQUENTIAL)

    assert record.steps == [1] * 3
    assert _parallel_is_restored()


def test_tqdm_joblib_can_be_reused_after_exit():
    progress = tqdm_joblib(total=2, silent=True)
    for _ in range(2):
        with progress as progress_bar:
            record = _Record(progress_bar)
            _squares(2, **SEQUENTIAL)
        assert record.steps == [1, 1]
        assert _parallel_is_restored()


def test_tqdm_joblib_cannot_be_entered_while_active():
    progress = tqdm_joblib(total=1, silent=True)

    with progress:
        with pytest.raises(RuntimeError, match="already active"):
            progress.__enter__()
        assert not _parallel_is_restored()

    assert _parallel_is_restored()


def test_tqdm_joblib_cannot_be_exited_before_entered():
    with pytest.raises(RuntimeError, match="not active"):
        tqdm_joblib(total=1).__exit__(None, None, None)

    assert _parallel_is_restored()


def test_tqdm_joblib_replaced_print_progress_works_after_exit():
    # joblib (loky) may call print_progress of a finished call late, after
    # the methods of Parallel are restored.
    with tqdm_joblib(total=2, silent=True):
        replaced_print_progress = Parallel.print_progress
    parallel = Parallel(n_jobs=1)
    parallel(delayed(_square)(value) for value in range(2))

    assert replaced_print_progress is not ORIGINAL_PRINT_PROGRESS
    assert replaced_print_progress(parallel) is None


def test_tqdm_joblib_keeps_the_progress_bar_in_tqdm():
    progress = tqdm_joblib(total=1, silent=True)
    assert progress.tqdm is None

    with progress as progress_bar:
        record = _Record(progress_bar)
        assert progress.tqdm is progress_bar

    # The closed progress bar is kept after exit.
    assert progress.tqdm is progress_bar
    assert record.n_closed == 1


class _ParallelWithoutProgress(Parallel):
    """A ``Parallel`` whose ``print_progress`` does nothing."""

    def print_progress(self):
        pass


@pytest.mark.parametrize(
    "parallel_kwargs", [SEQUENTIAL, THREADING], ids=["1", "2"]
)
def test_tqdm_joblib_moves_the_bar_when_a_call_ends(parallel_kwargs):
    with tqdm_joblib(total=4, silent=True) as progress_bar:
        record = _Record(progress_bar)
        _ParallelWithoutProgress(**parallel_kwargs)(
            delayed(_square)(value) for value in range(4)
        )
        # Without print_progress, the end of the call moves the bar.
        assert record.steps == [4]


def _run_again(parallel: Parallel) -> int:
    """Run a ``Parallel`` instance again (inside a task)."""
    return sum(parallel(delayed(_square)(value) for value in range(10)))


def test_tqdm_joblib_does_not_count_a_counted_instance_run_inside_a_task():
    parallel = Parallel(n_jobs=1)

    with tqdm_joblib(total=5, silent=True) as progress_bar:
        record = _Record(progress_bar)
        parallel(delayed(_square)(value) for value in range(2))
        # The same instance, run again inside the tasks of another call,
        # is not counted any more (it runs 10 tasks each time).
        Parallel(n_jobs=1)(delayed(_run_again)(parallel) for _ in range(3))

    assert record.steps == [1] * 5


@pytest.mark.parametrize(
    "parallel_kwargs", [SEQUENTIAL, THREADING], ids=["1", "2"]
)
def test_tqdm_joblib_leaves_generator_calls_as_they_are(parallel_kwargs):
    if not is_argument(Parallel, "return_as"):
        pytest.skip("joblib < 1.3 has no return_as")

    def run() -> Parallel:
        parallel = Parallel(**parallel_kwargs, return_as="generator")
        list(parallel(delayed(_square)(value) for value in range(3)))
        return parallel

    outside = run()
    with tqdm_joblib(total=1, silent=True):
        inside = run()

    # The call inside the context sets the same attributes as outside.
    assert sorted(vars(inside)) == sorted(vars(outside))


class _PausingParallel(Parallel):
    """A ``Parallel`` that pauses a thread reading ``n_completed_tasks``."""

    def __init__(self, *args, **kwargs) -> None:
        self.paused_thread: threading.Thread | None = None
        self.paused = threading.Event()
        self.resume = threading.Event()
        self._n_completed_tasks = 0
        super().__init__(*args, **kwargs)

    @property
    def n_completed_tasks(self) -> int:
        if threading.current_thread() is self.paused_thread:
            self.paused.set()
            _wait(self.resume)
        return self._n_completed_tasks

    @n_completed_tasks.setter
    def n_completed_tasks(self, value: int) -> None:
        self._n_completed_tasks = value


def test_tqdm_joblib_does_not_move_the_bar_after_it_is_closed():
    # joblib (loky) may run print_progress of a call from another thread
    # after the call has returned and the context has exited.
    parallel = _PausingParallel(n_jobs=1)
    late_threads: list[threading.Thread] = []
    errors: list[BaseException] = []

    def start_late_print_progress(value: int) -> int:
        if value == 0:
            # The replaced method, while the call is counted.
            print_progress = Parallel.print_progress

            def late_print_progress() -> None:
                try:
                    print_progress(parallel)
                except BaseException as error:  # reported by the test
                    errors.append(error)

            thread = threading.Thread(target=late_print_progress)
            parallel.paused_thread = thread
            thread.start()
            # Paused after it read the progress bound to the call.
            _wait(parallel.paused)
            late_threads.append(thread)
        return value

    with tqdm_joblib(total=2, silent=True) as progress_bar:
        record = _Record(progress_bar)
        parallel(
            delayed(start_late_print_progress)(value) for value in range(2)
        )
    # The late call sees more finished tasks than when the call ended.
    parallel(delayed(_square)(value) for value in range(5))
    parallel.resume.set()
    late_threads[0].join(timeout=TIMEOUT)

    assert not late_threads[0].is_alive()
    assert errors == []
    assert record.steps == [1, 1]


# --- feature_names_of --------------------------------------------------------


def test_feature_names_of_returns_str_column_names():
    names = feature_names_of(_frame(COLUMNS))

    assert isinstance(names, np.ndarray)
    assert names.dtype == object
    assert names.tolist() == COLUMNS
    assert all(type(name) is str for name in names)


@pytest.mark.parametrize(
    "columns",
    [
        [0, 1, 2],
        ["a", 1, "c"],
        # Only ``str`` itself counts (as in scikit-learn).
        [np.str_("a"), np.str_("b"), np.str_("c")],
        ["a", np.str_("b"), "c"],
    ],
    ids=["int", "mixed", "numpy-str", "str-and-numpy-str"],
)
def test_feature_names_of_returns_none_unless_all_names_are_str(columns):
    assert feature_names_of(_frame(columns)) is None


@pytest.mark.parametrize(
    "data",
    [X, X.tolist(), pd.DataFrame(X).to_numpy()],
    ids=["ndarray", "list", "ndarray-from-dataframe"],
)
def test_feature_names_of_returns_none_without_columns(data):
    assert feature_names_of(data) is None


def test_feature_names_of_returns_none_for_no_columns():
    assert feature_names_of(pd.DataFrame(index=range(3))) is None


@pytest.mark.parametrize(
    "data",
    [
        _frame(COLUMNS),
        _frame([0, 1, 2]),
        _frame([np.str_("a"), np.str_("b"), np.str_("c")]),
        X,
    ],
    ids=["str", "int", "numpy-str", "ndarray"],
)
def test_feature_names_of_agrees_with_the_installed_sklearn(data):
    # The installed scikit-learn sets the same ``feature_names_in_``, or
    # none at all.
    expected = getattr(StandardScaler().fit(data), "feature_names_in_", None)

    names = feature_names_of(data)

    if expected is None:
        assert names is None
    else:
        assert names is not None
        assert names.tolist() == expected.tolist()
