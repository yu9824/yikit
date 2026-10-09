from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.image
import matplotlib.pyplot as plt
import optuna
import pandas as pd
import pytest
from lightgbm import LGBMRegressor
from matplotlib.testing.compare import compare_images
from matplotlib.text import Text
from matplotlib.ticker import PercentFormatter
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from yikit.models import Objective
from yikit.visualize import (
    SummarizePI,
    get_learning_curve_gb,
    get_learning_curve_optuna,
)

if sys.version_info >= (3, 10):
    from types import NoneType
else:
    NoneType = type(None)  # type: ignore[assignment,misc]

if sys.version_info < (3, 14):
    from ngboost import NGBRegressor  # type: ignore[reportMissingImports]
else:
    NGBRegressor = NoneType  # type: ignore[assignment,misc]

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from matplotlib.figure import Figure


# ===========================================================================
# Self-tests of the ``assert_figure_matches_reference`` fixture
# (tests/conftest.py).
#
# They draw small synthetic figures and keep every reference image they
# create inside ``tmp_path``; ``tests/imgs`` is only read, never written.
# ===========================================================================

_REPO_REFERENCE_DIR = Path(__file__).parent / "imgs"
_SAVE_REFERENCE_OPTION = "--save-reference-figures"
_SELF_TEST_DPI = 36  # resolution the fixture uses when saving figures

_X_VALUES = (0.0, 1.0, 2.0, 3.0, 4.0)
_BASE_LINES = (
    (1.0, 3.0, 2.0, 5.0, 4.0),
    (6.0, 5.0, 7.0, 6.0, 8.0),
)
_EXTRA_LINE = (9.0, 1.0, 9.0, 1.0, 9.0)
_CHANGED_LINES = (
    (1.0, 3.0, 2.0, 5.0, 4.0),
    (2.0, 9.0, 1.0, 8.0, 3.0),
)


@pytest.fixture
def reference_figure_mode(
    monkeypatch: pytest.MonkeyPatch, pytestconfig: pytest.Config
) -> Callable[[bool], None]:
    """Force the fixture's mode, whatever the command line says.

    Compare mode is set on entry; the returned callable switches to save
    mode (``True``) or back to compare mode (``False``).
    """

    def set_save_mode(save: bool) -> None:
        # ``raising=True`` also proves that the option is registered.
        monkeypatch.setattr(
            pytestconfig.option, "save_reference_figures", save
        )

    set_save_mode(False)
    return set_save_mode


def _make_line_figure(
    lines: Sequence[Sequence[float]] = _BASE_LINES,
    *,
    title: str = "Reference",
    xlabel: str = "x",
    ylabel: str = "y",
    fontfamily: str = "DejaVu Sans",
    fontsize: float = 10.0,
) -> Figure:
    """Draw ``lines`` on fixed axis limits so only the data moves pixels.

    The lines are thick so that changing one of them clearly exceeds the
    fixture's tolerance.
    """
    fig, ax = plt.subplots(figsize=(4, 3))
    for ys in lines:
        ax.plot(_X_VALUES, ys, linewidth=5)
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 10)
    ax.set_title(title, fontfamily=fontfamily, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontfamily=fontfamily, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontfamily=fontfamily, fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    return fig


def _make_text_variant() -> Figure:
    """Same data as the default figure, with every text changed."""
    fig = _make_line_figure(
        title="A completely different and much longer title",
        xlabel="another x label",
        ylabel="another y label",
        fontfamily="DejaVu Serif",
        fontsize=16.0,
    )
    ax = fig.axes[0]
    ax.xaxis.set_major_formatter(PercentFormatter())
    ax.tick_params(labelcolor="tab:red")
    fig.suptitle("Suptitle", fontsize=20)
    fig.text(0.5, 0.5, "WATERMARK", fontsize=40, ha="center", va="center")
    return fig


def _save_reference(
    check: Callable[..., None],
    set_save_mode: Callable[[bool], None],
    fig: Figure,
    name: str,
    reference_dir: Path,
) -> Path:
    """Write a reference image through the fixture's save mode."""
    set_save_mode(True)
    try:
        check(fig, name, reference_dir=reference_dir)
    finally:
        set_save_mode(False)
    reference_path = reference_dir / name
    assert reference_path.is_file()
    return reference_path


def _snapshot(directory: Path) -> dict[str, tuple[int, int]]:
    """Map each file name in ``directory`` to its size and mtime."""
    return {
        path.name: (path.stat().st_size, path.stat().st_mtime_ns)
        for path in directory.iterdir()
    }


def test_figure_fixture_passes_for_identical_figure(
    assert_figure_matches_reference, reference_figure_mode, tmp_path
):
    reference_dir = tmp_path / "references"
    _save_reference(
        assert_figure_matches_reference,
        reference_figure_mode,
        _make_line_figure(),
        "lines.png",
        reference_dir,
    )

    fig = _make_line_figure()
    assert_figure_matches_reference(
        fig, "lines.png", reference_dir=reference_dir
    )
    assert not plt.fignum_exists(fig.number)


@pytest.mark.parametrize(
    "lines",
    [(*_BASE_LINES, _EXTRA_LINE), _CHANGED_LINES],
    ids=["extra-line", "changed-line-values"],
)
def test_figure_fixture_fails_for_changed_lines(
    assert_figure_matches_reference, reference_figure_mode, tmp_path, lines
):
    reference_dir = tmp_path / "references"
    _save_reference(
        assert_figure_matches_reference,
        reference_figure_mode,
        _make_line_figure(),
        "lines.png",
        reference_dir,
    )

    fig = _make_line_figure(lines)
    with pytest.raises(AssertionError) as excinfo:
        assert_figure_matches_reference(
            fig, "lines.png", reference_dir=reference_dir
        )
    message = str(excinfo.value)
    assert "RMS" in message
    assert "tolerance" in message
    assert _SAVE_REFERENCE_OPTION in message
    # The difference image lands in tmp_path, next to the actual image and
    # the copy of the reference, not next to the reference itself.
    diff_images = list(tmp_path.rglob("*-failed-diff.png"))
    assert len(diff_images) == 1
    assert str(diff_images[0]) in message
    assert reference_dir not in diff_images[0].parents
    assert not plt.fignum_exists(fig.number)


def test_figure_fixture_ignores_text_only_differences(
    assert_figure_matches_reference, reference_figure_mode, tmp_path
):
    # Control: with their texts drawn, the two figures do differ.
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for fig, filename in (
        (_make_line_figure(), "base.png"),
        (_make_text_variant(), "variant.png"),
    ):
        fig.savefig(raw_dir / filename, dpi=_SELF_TEST_DPI)
        plt.close(fig)
    assert (
        compare_images(
            str(raw_dir / "base.png"), str(raw_dir / "variant.png"), tol=0
        )
        is not None
    )

    reference = _save_reference(
        assert_figure_matches_reference,
        reference_figure_mode,
        _make_line_figure(),
        "lines.png",
        tmp_path / "references",
    )
    variant = _save_reference(
        assert_figure_matches_reference,
        reference_figure_mode,
        _make_text_variant(),
        "lines.png",
        tmp_path / "variant-references",
    )
    # With the texts hidden, the renderings are pixel-identical ...
    assert compare_images(str(reference), str(variant), tol=0) is None
    # ... so the comparison passes.
    assert_figure_matches_reference(
        _make_text_variant(), "lines.png", reference_dir=reference.parent
    )


def test_figure_fixture_hides_every_text_artist(
    assert_figure_matches_reference, reference_figure_mode, tmp_path
):
    fig = _make_line_figure()
    fig.axes[0].legend(["first", "second"])
    fig.suptitle("Suptitle")
    fig.text(0.1, 0.1, "Figure text")
    assert any(
        text.get_visible() and text.get_text() for text in fig.findobj(Text)
    )

    _save_reference(
        assert_figure_matches_reference,
        reference_figure_mode,
        fig,
        "texts.png",
        tmp_path / "references",
    )
    # Includes the tick labels created while the figure was saved.
    visible_texts = [
        text.get_text() for text in fig.findobj(Text) if text.get_visible()
    ]
    assert visible_texts == []


def test_figure_fixture_writes_nothing_beside_references_on_failure(
    assert_figure_matches_reference, reference_figure_mode, tmp_path
):
    reference_dir = tmp_path / "references"
    _save_reference(
        assert_figure_matches_reference,
        reference_figure_mode,
        _make_line_figure(),
        "lines.png",
        reference_dir,
    )
    repo_before = _snapshot(_REPO_REFERENCE_DIR)
    reference_before = _snapshot(reference_dir)

    # Failure against a reference in a temporary directory.
    with pytest.raises(AssertionError, match="RMS"):
        assert_figure_matches_reference(
            _make_line_figure(_CHANGED_LINES),
            "lines.png",
            reference_dir=reference_dir,
        )
    assert _snapshot(reference_dir) == reference_before

    # Failure against a committed reference in tests/imgs.
    repo_reference = sorted(_REPO_REFERENCE_DIR.glob("*.png"))[0]
    height, width = matplotlib.image.imread(repo_reference).shape[:2]
    fig = plt.figure(
        figsize=(
            (width + 0.5) / _SELF_TEST_DPI,
            (height + 0.5) / _SELF_TEST_DPI,
        )
    )
    fig.add_subplot().fill_between([0, 1], [0, 1], color="black")
    with pytest.raises(AssertionError, match="RMS"):
        assert_figure_matches_reference(fig, repo_reference.name)
    assert _snapshot(_REPO_REFERENCE_DIR) == repo_before


def test_figure_fixture_fails_with_instructions_when_reference_is_missing(
    assert_figure_matches_reference, reference_figure_mode
):
    name = "__missing_reference_for_self_test__.png"
    repo_before = _snapshot(_REPO_REFERENCE_DIR)

    fig = _make_line_figure()
    with pytest.raises(AssertionError) as excinfo:
        assert_figure_matches_reference(fig, name)
    message = str(excinfo.value)
    assert str(_REPO_REFERENCE_DIR / name) in message
    assert _SAVE_REFERENCE_OPTION in message
    assert _snapshot(_REPO_REFERENCE_DIR) == repo_before
    assert not plt.fignum_exists(fig.number)


# ===========================================================================
# Figure tests of yikit.visualize
# ===========================================================================


SEED = 334

# 参照画像のディレクトリ
REFERENCE_IMGS_DIR = Path(__file__).parent / "imgs"

# 実行モード: pytest では compare、直実行では save に切り替える
_FIGURE_MODE = "compare"  # "compare" | "save"

# 参照画像の出力を安定させるため dpi を固定
_SAVEFIG_DPI = 36

# 画像比較の許容値
_TOL_RMS = 20


def _reference_path(filename: str) -> Path:
    reference_path = REFERENCE_IMGS_DIR / filename
    return reference_path


def save_reference_figure(fig, filename: str) -> Path:
    """参照画像（tests/imgs）を保存（上書き）する。"""
    REFERENCE_IMGS_DIR.mkdir(parents=True, exist_ok=True)
    path = _reference_path(filename)
    fig.savefig(path, dpi=_SAVEFIG_DPI)
    plt.close(fig)
    return path


def assert_figure_matches_reference(fig, filename: str) -> None:
    """図を一時ファイルに保存し、参照画像（tests/imgs）と比較する。"""
    reference_path = _reference_path(filename)
    assert reference_path.exists(), (
        f"参照画像が見つかりません: {reference_path}\n"
        f"`python3 {Path(__file__).name}` を実行して参照画像を生成してください。"
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / filename
        fig.savefig(tmp_path, dpi=_SAVEFIG_DPI)
        plt.close(fig)
        results = compare_images(
            str(tmp_path), str(reference_path), tol=_TOL_RMS
        )
        assert results is None, (
            f"生成された画像が参照画像と異なります: {tmp_path} vs {reference_path}"
            f"差分: {results}"
        )


def handle_figure(fig, filename: str) -> None:
    """モードに応じて、参照保存 or 参照比較を行う。"""
    if _FIGURE_MODE == "save":
        save_reference_figure(fig, filename)
    elif _FIGURE_MODE == "compare":
        assert_figure_matches_reference(fig, filename)
    else:
        raise ValueError(f"Unknown _FIGURE_MODE: {_FIGURE_MODE}")


def _make_regression_df(seed: int = 334):
    X, y = make_regression(
        n_samples=442,
        n_features=10,
        n_informative=10,
        noise=10.0,
        random_state=seed,
    )
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y)
    return X_df, y_series


def _make_train_test_split(seed: int = SEED):
    X, y = _make_regression_df(seed=seed)
    return train_test_split(X, y, test_size=0.2, random_state=seed)


def test_summarize_pi():
    X_train, X_test, y_train, y_test = _make_train_test_split()

    # SummarizePI test
    rf = RandomForestRegressor(random_state=SEED)
    rf.fit(X_train, y_train)

    pi = permutation_importance(rf, X_test, y_test, random_state=SEED)
    spi = SummarizePI(pd.DataFrame(pi.importances, index=X_test.columns))
    fig, _ = spi.get_figure(fontfamily="DejaVu Sans")
    handle_figure(fig, "sample_summarize_pi.png")


@pytest.mark.skipif(
    sys.version_info >= (3, 14),
    reason="ngboost is not supported on Python 3.14",
)
def test_get_dist_figure():
    from yikit.visualize import get_dist_figure

    X_train, X_test, y_train, y_test = _make_train_test_split()

    ngb = NGBRegressor(random_state=SEED, verbose=False).fit(X_train, y_train)
    # 画像サイズが大きくなりすぎる＆環境差が出やすいのでサンプル数を絞る
    X_test = X_test.iloc[:4]
    y_test = y_test.iloc[:4]
    fig = get_dist_figure(
        ngb.pred_dist(X_test),
        y_test,
        titles=["a"] * len(y_test),
        verbose=False,
        fontfamily="DejaVu Sans",
    )
    handle_figure(fig, "sample_dist_figure.png")


def test_learning_curve_optuna():
    X_train, X_test, y_train, y_test = _make_train_test_split()

    rf = RandomForestRegressor(random_state=SEED)
    objective = Objective(rf, X_train, y_train, random_state=SEED)
    study = optuna.create_study(
        sampler=objective.sampler, direction="maximize"
    )
    study.optimize(objective, n_trials=10)
    fig = get_learning_curve_optuna(study, fontfamily="DejaVu Sans")
    handle_figure(fig, "sample_learning_curve_optuna.png")


@pytest.mark.skipif(
    sys.version_info >= (3, 14),
    reason="ngboost is not supported on Python 3.14",
)
def test_learning_curve_ngboost():
    X_train, X_test, y_train, y_test = _make_train_test_split()

    ngb = NGBRegressor(
        random_state=SEED, Base=DecisionTreeRegressor(random_state=SEED)
    )
    ngb.fit(X_train, y_train, X_val=X_test, Y_val=y_test)
    fig = get_learning_curve_gb(ngb, fontfamily="DejaVu Sans")
    handle_figure(fig, "sample_learning_curve_ngboost.png")


def test_learning_curve_lightgbm():
    X_train, X_test, y_train, y_test = _make_train_test_split()

    lgbm = LGBMRegressor(random_state=SEED)
    lgbm.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        eval_names=["train", "test"],
    )
    fig = get_learning_curve_gb(lgbm, fontfamily="DejaVu Sans")
    handle_figure(fig, "sample_learning_curve_lightgbm.png")


if __name__ == "__main__":
    _FIGURE_MODE = "save"

    # `python3 tests/test_visualize.py` 直実行時は、pytest と同じ test 関数を呼ぶ
    test_summarize_pi()
    test_get_dist_figure()
    test_learning_curve_optuna()
    test_learning_curve_ngboost()
    test_learning_curve_lightgbm()
