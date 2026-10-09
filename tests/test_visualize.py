from __future__ import annotations

import sys
from itertools import accumulate
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.image
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import pytest
from lightgbm import LGBMRegressor
from matplotlib.testing.compare import compare_images
from matplotlib.text import Text
from matplotlib.ticker import PercentFormatter
from scipy.stats import norm

from yikit.visualize import (
    SummarizePI,
    get_learning_curve_gb,
    get_learning_curve_optuna,
)

if sys.version_info < (3, 14):
    from ngboost import NGBRegressor  # type: ignore[reportMissingImports]
    from ngboost.distns import Normal  # type: ignore[reportMissingImports]
else:  # ngboost does not support Python 3.14; its tests are skipped.
    NGBRegressor = Normal = None

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from matplotlib.axes import Axes
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
    # Legend frames and handles are sized from the texts, so they go too.
    assert not fig.axes[0].get_legend().get_visible()


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
#
# Every figure is drawn from fixed values; only the LightGBM learning curve
# needs a fit, on a small fixed dataset with deterministic settings. Each
# test checks the texts and the numbers of elements as values before the
# image comparison, which ignores texts.
# ===========================================================================

SEED = 334

#: Font family passed to every figure function.
FONT_FAMILY = "DejaVu Sans"

requires_ngboost = pytest.mark.skipif(
    sys.version_info >= (3, 14),
    reason="ngboost is not supported on Python 3.14",
)

# SummarizePI: permutation importances (features x repeats).
_N_FEATURES = 10
_N_REPEATS = 5

# get_dist_figure: normal distributions and the ground truth.
_DIST_LOC = (10.0, 25.0, 40.0, 55.0)
_DIST_SCALE = (4.0, 6.0, 5.0, 8.0)
_DIST_Y_TRUE = (12.0, 20.0, 43.0, 50.0)
_DIST_TITLES = ("a", "b", "c", "d")

# get_learning_curve_optuna: objective values of a maximized study.
_TRIAL_VALUES = (0.21, 0.35, 0.30, 0.52, 0.48, 0.61, 0.58, 0.66, 0.64, 0.70)

# get_learning_curve_gb (NGBoost): ``evals_result`` as NGBoost's ``fit``
# with a validation set stores it.
_NGBOOST_METRIC = "LOGSCORE"
_NGBOOST_SCORES = {
    "train": (5.0, 4.2, 3.6, 3.2, 2.9, 2.7, 2.6, 2.5),
    "val": (5.1, 4.5, 4.0, 3.8, 3.7, 3.7, 3.8, 3.9),
}

# get_learning_curve_gb (LightGBM): boosting rounds of the fit.
_LGBM_N_ESTIMATORS = 20
_LGBM_EVAL_NAMES = ("train", "test")


def _make_importances() -> pd.DataFrame:
    """Return importances drawn from a seeded random state."""
    rng = np.random.RandomState(SEED)
    return pd.DataFrame(
        rng.uniform(0.0, 1.0, size=(_N_FEATURES, _N_REPEATS)),
        index=[f"feature_{i}" for i in range(_N_FEATURES)],
    )


def _make_study() -> optuna.Study:
    """Return a study holding completed trials with fixed values."""
    study = optuna.create_study(direction="maximize")
    for value in _TRIAL_VALUES:
        study.add_trial(optuna.trial.create_trial(value=value))
    return study


def _make_ngb_regressor() -> NGBRegressor:
    """Return an NGBRegressor whose ``evals_result`` is fixed."""
    ngb = NGBRegressor()
    ngb.evals_result = {
        name: {_NGBOOST_METRIC: list(scores)}
        for name, scores in _NGBOOST_SCORES.items()
    }
    return ngb


def _fit_lgbm_regressor() -> LGBMRegressor:
    """Fit LightGBM on a small fixed dataset with deterministic settings."""
    rng = np.random.RandomState(SEED)
    X = rng.uniform(-1.0, 1.0, size=(100, 3))
    y = X @ np.array([3.0, -2.0, 1.0]) + rng.normal(0.0, 0.1, size=100)
    X_train, X_test, y_train, y_test = X[:80], X[80:], y[:80], y[80:]

    lgbm = LGBMRegressor(
        n_estimators=_LGBM_N_ESTIMATORS,
        random_state=SEED,
        deterministic=True,
        force_row_wise=True,
        n_jobs=1,
        verbose=-1,
    )
    # ``eval_set`` (not ``eval_X``) keeps LightGBM 4.6 on Python 3.8 working.
    lgbm.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        eval_names=list(_LGBM_EVAL_NAMES),
    )
    return lgbm


def _draw_summarize_pi(fontfamily: str = FONT_FAMILY) -> Figure:
    fig, _ = SummarizePI(_make_importances()).get_figure(fontfamily=fontfamily)
    return fig


def _draw_dist_figure(fontfamily: str = FONT_FAMILY) -> Figure:
    from yikit.visualize import get_dist_figure

    # ngboost's Normal takes the rows (loc, log(scale)).
    y_dist = Normal(np.array([_DIST_LOC, np.log(_DIST_SCALE)]))
    return get_dist_figure(
        y_dist,
        np.array(_DIST_Y_TRUE),
        titles=list(_DIST_TITLES),
        verbose=False,
        fontfamily=fontfamily,
    )


def _draw_learning_curve_optuna(fontfamily: str = FONT_FAMILY) -> Figure:
    return get_learning_curve_optuna(_make_study(), fontfamily=fontfamily)


def _draw_learning_curve_ngboost(fontfamily: str = FONT_FAMILY) -> Figure:
    return get_learning_curve_gb(_make_ngb_regressor(), fontfamily=fontfamily)


def _draw_learning_curve_lightgbm(fontfamily: str = FONT_FAMILY) -> Figure:
    return get_learning_curve_gb(_fit_lgbm_regressor(), fontfamily=fontfamily)


def _axis_texts(ax: Axes) -> tuple[str, str, str]:
    """Return the title, the x label and the y label of ``ax``."""
    return ax.get_title(), ax.get_xlabel(), ax.get_ylabel()


def _legend_labels(ax: Axes) -> list[str]:
    legend = ax.get_legend()
    assert legend is not None
    return [text.get_text() for text in legend.get_texts()]


def test_summarize_pi(assert_figure_matches_reference):
    mean_importances = _make_importances().mean(axis=1)
    expected = (mean_importances / mean_importances.sum()).sort_values(
        ascending=False
    )

    fig = _draw_summarize_pi()

    assert len(fig.axes) == 1
    ax = fig.axes[0]
    assert _axis_texts(ax) == ("", "", "")
    assert ax.get_legend() is None
    # One horizontal bar per feature, the most important one first.
    assert [label.get_text() for label in ax.get_yticklabels()] == list(
        expected.index
    )
    assert len(ax.patches) == _N_FEATURES
    np.testing.assert_allclose(
        [bar.get_width() for bar in ax.patches], expected.to_numpy()
    )

    assert_figure_matches_reference(fig, "sample_summarize_pi.png")


@requires_ngboost
def test_get_dist_figure(assert_figure_matches_reference):
    # Every density is drawn on one grid spanning the predicted means with
    # a 5% margin, and all subplots share the top of the y range.
    margin = np.ptp(_DIST_LOC) * 0.05
    x = np.linspace(min(_DIST_LOC) - margin, max(_DIST_LOC) + margin, 200)
    densities = [
        norm.pdf(x, loc, scale) for loc, scale in zip(_DIST_LOC, _DIST_SCALE)
    ]
    y_top = np.max(densities) * 1.05

    fig = _draw_dist_figure()

    assert len(fig.axes) == len(_DIST_TITLES)
    for ax, title, loc, y_true, density in zip(
        fig.axes, _DIST_TITLES, _DIST_LOC, _DIST_Y_TRUE, densities
    ):
        assert _axis_texts(ax) == (title, "", "Probability density")
        assert _legend_labels(ax) == ["ground truth", "pred"]
        assert len(ax.lines) == 1  # probability density
        np.testing.assert_allclose(np.ravel(ax.lines[0].get_xdata()), x)
        np.testing.assert_allclose(ax.lines[0].get_ydata(), density)
        # Vertical lines from 0 to the peak density.
        assert len(ax.collections) == 2
        for collection, x_position in zip(ax.collections, (y_true, loc)):
            (segment,) = collection.get_segments()
            np.testing.assert_allclose(
                segment, [[x_position, 0.0], [x_position, density.max()]]
            )
        assert ax.get_xlim() == pytest.approx((x[0], x[-1]))
        assert ax.get_ylim()[1] == pytest.approx(y_top)

    assert_figure_matches_reference(fig, "sample_dist_figure.png")


def test_learning_curve_optuna(assert_figure_matches_reference):
    fig = _draw_learning_curve_optuna()

    assert len(fig.axes) == 1
    ax = fig.axes[0]
    assert _axis_texts(ax) == ("", "Trials", "Objective Values")
    # Matplotlib < 3.5 lists the line before the scatter in the legend.
    assert sorted(_legend_labels(ax)) == ["Best Value", "Objective Value"]
    assert len(ax.collections) == 1  # objective value of each trial
    assert len(ax.lines) == 1  # best value so far
    np.testing.assert_allclose(
        np.asarray(ax.collections[0].get_offsets()),
        list(enumerate(_TRIAL_VALUES)),
    )
    np.testing.assert_allclose(
        ax.lines[0].get_xdata(), range(len(_TRIAL_VALUES))
    )
    np.testing.assert_allclose(
        ax.lines[0].get_ydata(), list(accumulate(_TRIAL_VALUES, max))
    )

    assert_figure_matches_reference(fig, "sample_learning_curve_optuna.png")


@requires_ngboost
def test_learning_curve_ngboost(assert_figure_matches_reference):
    fig = _draw_learning_curve_ngboost()

    assert len(fig.axes) == 1
    ax = fig.axes[0]
    assert _axis_texts(ax) == ("", "n_iter", _NGBOOST_METRIC)
    assert _legend_labels(ax) == list(_NGBOOST_SCORES)
    assert len(ax.lines) == len(_NGBOOST_SCORES)
    for line, scores in zip(ax.lines, _NGBOOST_SCORES.values()):
        np.testing.assert_allclose(line.get_xdata(), range(len(scores)))
        np.testing.assert_allclose(line.get_ydata(), scores)

    assert_figure_matches_reference(fig, "sample_learning_curve_ngboost.png")


def test_learning_curve_lightgbm(assert_figure_matches_reference):
    lgbm = _fit_lgbm_regressor()
    fig = get_learning_curve_gb(lgbm, fontfamily=FONT_FAMILY)

    assert len(fig.axes) == 1
    ax = fig.axes[0]
    assert _axis_texts(ax) == ("", "n_iter", "l2")
    assert _legend_labels(ax) == list(_LGBM_EVAL_NAMES)
    assert len(ax.lines) == len(_LGBM_EVAL_NAMES)
    for line, name in zip(ax.lines, _LGBM_EVAL_NAMES):
        scores = lgbm.evals_result_[name]["l2"]
        assert len(scores) == _LGBM_N_ESTIMATORS
        np.testing.assert_allclose(line.get_xdata(), range(_LGBM_N_ESTIMATORS))
        np.testing.assert_allclose(line.get_ydata(), scores)

    assert_figure_matches_reference(fig, "sample_learning_curve_lightgbm.png")


#: Font family with metrics far from :data:`FONT_FAMILY`. Matplotlib ships
#: both, so every environment has them.
_OTHER_FONT_FAMILY = "DejaVu Sans Mono"

#: RMS accepted between the same figure drawn with the two font families,
#: far below the fixture's tolerance: once the texts are hidden, the fonts
#: must not move the other artists at all.
_FONT_RMS_TOLERANCE = 1.0


@pytest.mark.parametrize(
    "draw",
    [
        pytest.param(_draw_summarize_pi, id="summarize_pi"),
        pytest.param(
            _draw_dist_figure, id="dist_figure", marks=requires_ngboost
        ),
        pytest.param(_draw_learning_curve_optuna, id="learning_curve_optuna"),
        pytest.param(
            _draw_learning_curve_ngboost,
            id="learning_curve_ngboost",
            marks=requires_ngboost,
        ),
        pytest.param(
            _draw_learning_curve_lightgbm, id="learning_curve_lightgbm"
        ),
    ],
)
def test_figure_comparison_ignores_font_family(
    assert_figure_matches_reference, reference_figure_mode, tmp_path, draw
):
    name = "figure.png"
    reference = _save_reference(
        assert_figure_matches_reference,
        reference_figure_mode,
        draw(FONT_FAMILY),
        name,
        tmp_path / "default-font",
    )
    other = _save_reference(
        assert_figure_matches_reference,
        reference_figure_mode,
        draw(_OTHER_FONT_FAMILY),
        name,
        tmp_path / "other-font",
    )
    rms_failure = compare_images(
        str(reference), str(other), tol=_FONT_RMS_TOLERANCE
    )
    assert rms_failure is None, rms_failure

    assert_figure_matches_reference(
        draw(_OTHER_FONT_FAMILY), name, reference_dir=reference.parent
    )
