"""
Copyright © 2021 yu9824

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.testing.compare import compare_images
from matplotlib.testing.exceptions import ImageComparisonFailure
from matplotlib.text import Text
from sklearn.datasets import make_regression

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.figure import Figure

#: Directory of the committed reference images.
REFERENCE_IMAGES_DIR = Path(__file__).parent / "imgs"

#: Resolution of both the reference images and the compared figures.
FIGURE_DPI = 36

#: Largest RMS difference (0-255 color scale) accepted between a figure and
#: its reference image, after every text has been hidden.
FIGURE_RMS_TOLERANCE = 20

_SAVE_REFERENCE_OPTION = "--save-reference-figures"
_REGENERATE_COMMAND = (
    f"pytest tests/test_visualize.py {_SAVE_REFERENCE_OPTION}"
)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add ``--save-reference-figures`` to regenerate reference images."""
    parser.addoption(
        _SAVE_REFERENCE_OPTION,
        action="store_true",
        default=False,
        dest="save_reference_figures",
        help=(
            "Overwrite the reference images in tests/imgs with the figures "
            "drawn by the tests instead of comparing them."
        ),
    )


@pytest.fixture(scope="session")
def regression_data():
    """テスト用の回帰データセットを生成

    load_diabetes()と同様のサイズ（442サンプル、10特徴量）で生成
    seedは固定（334）で再現性を確保
    """
    X, y = make_regression(
        n_samples=442,
        n_features=10,
        n_informative=10,
        noise=10.0,
        random_state=334,
    )

    # DataFrameとSeriesに変換（load_diabetes()と同様の形式）
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y)

    return {
        "data": X,
        "target": y,
        "feature_names": feature_names,
        "X": X_df,
        "y": y_series,
    }


@pytest.fixture(scope="session")
def X_regression(regression_data):
    """回帰データの特徴量（DataFrame）"""
    return regression_data["X"]


@pytest.fixture(scope="session")
def y_regression(regression_data):
    """回帰データのターゲット（Series）"""
    return regression_data["y"]


@pytest.fixture(scope="session", autouse=True)
def matplotlib_settings():
    plt.rcParams["backend"] = "Agg"


def _hide_texts(fig: Figure) -> None:
    """Make every text artist of ``fig`` invisible.

    This covers titles, axis labels, tick labels, offset texts, legend
    texts and figure texts. Ticks created later while drawing copy the
    visibility of the first tick, so their labels stay hidden as well.
    """
    for text in fig.findobj(Text):
        text.set_visible(False)


def _compare_with_reference(actual_path: Path, reference_path: Path) -> None:
    """Compare ``actual_path`` with a copy of ``reference_path``.

    The copy and the difference image are written next to ``actual_path``,
    so nothing is written beside the reference image.

    Raises
    ------
    AssertionError
        If the reference image is missing, if the sizes differ, or if the
        RMS difference exceeds :data:`FIGURE_RMS_TOLERANCE`.
    """
    __tracebackhide__ = True
    regenerate = (
        "If the figure is meant to change, regenerate the reference images "
        f"with `{_REGENERATE_COMMAND}`."
    )
    if not reference_path.is_file():
        raise AssertionError(
            f"Reference image not found: {reference_path}\n"
            f"  actual: {actual_path}\n"
            f"Create it with `{_REGENERATE_COMMAND}`."
        )

    expected_path = actual_path.with_name(
        f"{actual_path.stem}-expected{actual_path.suffix}"
    )
    shutil.copyfile(reference_path, expected_path)
    try:
        result = compare_images(
            str(expected_path),
            str(actual_path),
            tol=FIGURE_RMS_TOLERANCE,
            in_decorator=True,
        )
    except ImageComparisonFailure as exc:
        raise AssertionError(
            f"Figure cannot be compared with {reference_path}: {exc}\n"
            f"  actual: {actual_path}\n"
            f"  expected (copy of the reference): {expected_path}\n"
            f"{regenerate}"
        ) from exc
    if result is not None:
        raise AssertionError(
            f"Figure differs from the reference image {reference_path}\n"
            f"  RMS: {result['rms']:.3f} "
            f"(tolerance: {FIGURE_RMS_TOLERANCE})\n"
            f"  actual: {result['actual']}\n"
            f"  expected (copy of the reference): {result['expected']}\n"
            f"  difference: {result['diff']}\n"
            f"{regenerate}"
        )


@pytest.fixture
def assert_figure_matches_reference(
    request: pytest.FixtureRequest, tmp_path: Path
) -> Callable[..., None]:
    """Return a checker that compares ``fig`` with ``tests/imgs/<name>``.

    The checker hides every text of the figure, saves it with
    :data:`FIGURE_DPI` and closes it. With ``--save-reference-figures`` it
    overwrites the reference image; otherwise it compares the figure with
    the reference inside ``tmp_path`` and fails when the RMS difference
    exceeds :data:`FIGURE_RMS_TOLERANCE`. Texts are checked as values by
    the tests themselves, before calling the checker.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Gives access to the command line options.
    tmp_path : Path
        Receives the actual image, the copy of the reference and the
        difference image.

    Returns
    -------
    Callable[..., None]
        ``check(fig, name, *, reference_dir=None)``, where ``name`` is the
        file name of the reference image and ``reference_dir`` replaces
        ``tests/imgs`` (used by the self-tests of this fixture).
    """
    work_dir = tmp_path / "figure-comparison"

    def check(
        fig: Figure, name: str, *, reference_dir: Path | None = None
    ) -> None:
        __tracebackhide__ = True
        reference_path = (
            REFERENCE_IMAGES_DIR if reference_dir is None else reference_dir
        ) / name
        save_mode = bool(request.config.getoption(_SAVE_REFERENCE_OPTION))
        output_path = reference_path if save_mode else work_dir / name
        try:
            _hide_texts(fig)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=FIGURE_DPI)
        finally:
            plt.close(fig)
        if not save_mode:
            _compare_with_reference(output_path, reference_path)

    return check
