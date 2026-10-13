from __future__ import annotations

import platform
from itertools import accumulate
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.cbook import is_math_text
from matplotlib.container import ErrorbarContainer
from matplotlib.mathtext import MathTextParser
from matplotlib.testing.compare import compare_images
from matplotlib.text import Text
from matplotlib.ticker import PercentFormatter
from scipy.stats import norm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from yikit.visualize import (
    SummarizePI,
    set_font,
    with_custom_matplotlib_settings,
    yyplot,
)

# optuna, lightgbm and ngboost are optional: the tests that use them import
# them through ``pytest.importorskip`` and are skipped without them (ngboost
# is not installed on Python 3.14, which it does not support).
if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import optuna
    from lightgbm import LGBMRegressor
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from ngboost import NGBRegressor  # type: ignore[reportMissingImports]


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
    """Return a study holding completed trials with fixed values.

    The calling test is skipped when optuna is not installed.
    """
    optuna = pytest.importorskip("optuna")

    study = optuna.create_study(direction="maximize")
    for value in _TRIAL_VALUES:
        study.add_trial(optuna.trial.create_trial(value=value))
    return study


def _make_ngb_regressor() -> NGBRegressor:
    """Return an NGBRegressor whose ``evals_result`` is fixed.

    The calling test is skipped when ngboost is not installed.
    """
    ngboost = pytest.importorskip("ngboost")

    ngb = ngboost.NGBRegressor()
    ngb.evals_result = {
        name: {_NGBOOST_METRIC: list(scores)}
        for name, scores in _NGBOOST_SCORES.items()
    }
    return ngb


def _fit_lgbm_regressor() -> LGBMRegressor:
    """Fit LightGBM on a small fixed dataset with deterministic settings.

    The calling test is skipped when lightgbm is not installed.
    """
    lightgbm = pytest.importorskip("lightgbm")

    rng = np.random.RandomState(SEED)
    X = rng.uniform(-1.0, 1.0, size=(100, 3))
    y = X @ np.array([3.0, -2.0, 1.0]) + rng.normal(0.0, 0.1, size=100)
    X_train, X_test, y_train, y_test = X[:80], X[80:], y[:80], y[80:]

    lgbm = lightgbm.LGBMRegressor(
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
    distns = pytest.importorskip("ngboost.distns")
    from yikit.visualize import get_dist_figure

    # ngboost's Normal takes the rows (loc, log(scale)).
    y_dist = distns.Normal(np.array([_DIST_LOC, np.log(_DIST_SCALE)]))
    return get_dist_figure(
        y_dist,
        np.array(_DIST_Y_TRUE),
        titles=list(_DIST_TITLES),
        verbose=False,
        fontfamily=fontfamily,
    )


# The figure functions below exist in yikit.visualize only when their
# optional packages are installed, so each is imported after the model or
# study that skips the test without them.


def _draw_learning_curve_optuna(fontfamily: str = FONT_FAMILY) -> Figure:
    study = _make_study()
    from yikit.visualize import get_learning_curve_optuna

    return get_learning_curve_optuna(study, fontfamily=fontfamily)


def _draw_learning_curve_ngboost(fontfamily: str = FONT_FAMILY) -> Figure:
    ngb = _make_ngb_regressor()
    from yikit.visualize import get_learning_curve_gb

    return get_learning_curve_gb(ngb, fontfamily=fontfamily)


def _draw_learning_curve_lightgbm(fontfamily: str = FONT_FAMILY) -> Figure:
    lgbm = _fit_lgbm_regressor()
    from yikit.visualize import get_learning_curve_gb

    return get_learning_curve_gb(lgbm, fontfamily=fontfamily)


def _axis_texts(ax: Axes) -> tuple[str, str, str]:
    """Return the title, the x label and the y label of ``ax``."""
    return ax.get_title(), ax.get_xlabel(), ax.get_ylabel()


def _legend_labels(ax: Axes) -> list[str]:
    legend = ax.get_legend()
    assert legend is not None
    return [text.get_text() for text in legend.get_texts()]


def _error_bar_segments(ax: Axes) -> np.ndarray:
    """Return the segments of the only error bars on ``ax``.

    The array has the shape (n_bars, 2, 2): the two (x, y) end points of
    each error bar.
    """
    (container,) = [
        container
        for container in ax.containers
        if isinstance(container, ErrorbarContainer)
    ]
    _, _, (bar_lines,) = container.lines
    return np.asarray(bar_lines.get_segments())


def _assert_summarize_pi_bars(
    ax: Axes, names: Sequence[object], means: np.ndarray, stds: np.ndarray
) -> None:
    """Check the bars and the horizontal error bars drawn by SummarizePI.

    The i-th feature of ``names`` must be drawn at y = i, counted from the
    top: its tick label, its bar of width ``means[i]`` and its error bar
    from ``means[i] - stds[i]`` to ``means[i] + stds[i]``. The y range must
    hold the bars only, without margins.
    """
    positions = np.arange(len(names))
    np.testing.assert_array_equal(ax.get_yticks(), positions)
    assert [label.get_text() for label in ax.get_yticklabels()] == [
        str(name) for name in names
    ]
    # seaborn may add the bars in another order than their positions.
    centers = np.array(
        [bar.get_y() + bar.get_height() / 2 for bar in ax.patches]
    )
    widths = np.array([bar.get_width() for bar in ax.patches])
    np.testing.assert_allclose(np.sort(centers), positions)
    np.testing.assert_allclose(widths[np.argsort(centers)], means)
    segments = _error_bar_segments(ax)
    np.testing.assert_allclose(
        segments[np.argsort(segments[:, 0, 1])],
        np.stack(
            [
                np.column_stack([means - stds, positions]),
                np.column_stack([means + stds, positions]),
            ],
            axis=1,
        ),
    )
    assert ax.get_ylim() == (len(names) - 0.5, -0.5)


def test_summarize_pi(assert_figure_matches_reference):
    importances = _make_importances()
    # Each value is divided by the sum of the per-feature means: the bars
    # show the mean of each feature and the error bars the population
    # standard deviation (as scikit-learn's ``importances_std``).
    values = importances.to_numpy() / importances.mean(axis=1).sum()
    order = np.argsort(-values.mean(axis=1), kind="stable")

    fig = _draw_summarize_pi()

    assert len(fig.axes) == 1
    ax = fig.axes[0]
    assert _axis_texts(ax) == ("", "", "")
    assert ax.get_legend() is None
    _assert_summarize_pi_bars(
        ax,
        importances.index[order],
        values.mean(axis=1)[order],
        values.std(axis=1)[order],
    )

    assert_figure_matches_reference(fig, "sample_summarize_pi.png")


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
    from yikit.visualize import get_learning_curve_gb

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
#: well below the fixture's tolerance: once the texts are hidden, the fonts
#: must not move the other artists at all (CI measured 0 on every version).
_FONT_RMS_TOLERANCE = 0.1


@pytest.mark.parametrize(
    "draw",
    [
        pytest.param(_draw_summarize_pi, id="summarize_pi"),
        pytest.param(_draw_dist_figure, id="dist_figure"),
        pytest.param(_draw_learning_curve_optuna, id="learning_curve_optuna"),
        pytest.param(
            _draw_learning_curve_ngboost, id="learning_curve_ngboost"
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


# ===========================================================================
# SummarizePI.get_data and the font settings
# ===========================================================================

#: Global font settings in place before a call, unlike any value set by
#: yikit. Every test applies them in ``plt.rc_context``, which restores the
#: settings of the session afterwards.
_USER_FONT_SETTINGS = {"font.family": ["serif"], "font.size": 9.0}

#: Permutation importances with exact normalized values: the per-feature
#: means 0.6, 0.2 and 1.2 sum to 2.0.
_SMALL_IMPORTANCES = pd.DataFrame(
    [[0.4, 0.8], [0.2, 0.2], [1.0, 1.4]],
    index=pd.Index(["a", "b", "c"], name="feature"),
)


def _font_settings() -> tuple[list[str], float]:
    """Return the global font family and font size of matplotlib."""
    return list(plt.rcParams["font.family"]), plt.rcParams["font.size"]


def test_summarize_pi_get_data():
    importances = _SMALL_IMPORTANCES.copy()

    data = SummarizePI(importances).get_data()

    pd.testing.assert_frame_equal(
        data,
        pd.DataFrame(
            {
                "importances_mean": [0.6, 0.3, 0.1],
                "importances_std": [0.1, 0.1, 0.0],
            },
            index=pd.Index(["c", "a", "b"], name="feature"),
        ),
    )
    pd.testing.assert_frame_equal(importances, _SMALL_IMPORTANCES)


def test_summarize_pi_get_data_keeps_order_of_ties():
    # Unimportant features often tie at zero; ties keep the input order.
    names = [f"feature_{i}" for i in range(40)]
    values = np.zeros((len(names), 3))
    values[5] = (1.0, 2.0, 3.0)
    values[17] = (3.0, 4.0, 5.0)
    values[30] = (2.0, 2.0, 2.0)

    data = SummarizePI(pd.DataFrame(values, index=names)).get_data()

    assert list(data.index) == [
        "feature_17",
        "feature_5",
        "feature_30",
        *(name for i, name in enumerate(names) if i not in (5, 17, 30)),
    ]


@pytest.mark.parametrize(
    "index",
    [
        pytest.param(None, id="str"),
        # seaborn >= 0.13 sorts numeric names unless told the order.
        pytest.param(pd.RangeIndex(_N_FEATURES), id="int"),
        pytest.param([i + 0.5 for i in range(_N_FEATURES)], id="float"),
    ],
)
def test_summarize_pi_get_data_matches_figure(index):
    importances = _make_importances()
    if index is not None:
        importances.index = index
    summarize_pi = SummarizePI(importances)

    data = summarize_pi.get_data()
    fig, ax = summarize_pi.get_figure(fontfamily=FONT_FAMILY)

    try:
        assert fig is summarize_pi.fig
        assert ax is summarize_pi.ax
        # Sorting by importance must differ from sorting by name.
        assert list(data.index) != sorted(data.index)
        _assert_summarize_pi_bars(
            ax,
            data.index,
            data["importances_mean"].to_numpy(),
            data["importances_std"].to_numpy(),
        )
    finally:
        plt.close(fig)


def test_summarize_pi_get_figure_restores_font_settings():
    with plt.rc_context(_USER_FONT_SETTINGS):
        before = _font_settings()

        fig, ax = SummarizePI(_make_importances()).get_figure(
            fontfamily=_OTHER_FONT_FAMILY
        )

        try:
            assert _font_settings() == before
            # The font family was in effect while the figure was drawn.
            assert ax.get_yticklabels()[0].get_fontfamily() == [
                _OTHER_FONT_FAMILY
            ]
        finally:
            plt.close(fig)


def test_set_font_changes_global_settings():
    with plt.rc_context(_USER_FONT_SETTINGS):
        set_font(fontfamily=_OTHER_FONT_FAMILY, fontsize=17)

        # The settings stay changed after the call, for every new figure.
        assert _font_settings() == ([_OTHER_FONT_FAMILY], 17.0)
        fig = plt.figure()
        try:
            text = fig.text(0.5, 0.5, "text")
            assert text.get_fontfamily() == [_OTHER_FONT_FAMILY]
            assert text.get_fontsize() == 17.0
        finally:
            plt.close(fig)


@pytest.mark.parametrize(
    ("system", "expected"),
    [
        ("Darwin", "Helvetica"),
        ("Linux", "DejaVu Sans"),
        ("Windows", "DejaVu Sans"),
    ],
)
def test_set_font_defaults_to_platform_font(monkeypatch, system, expected):
    monkeypatch.setattr(platform, "system", lambda: system)

    # Only the settings are checked: nothing is drawn, so the font does not
    # need to be installed.
    with plt.rc_context(_USER_FONT_SETTINGS):
        set_font()

        assert _font_settings() == ([expected], 13.0)


@pytest.mark.parametrize("restore", [True, False])
def test_with_custom_matplotlib_settings(restore):
    @with_custom_matplotlib_settings(
        fontfamily=_OTHER_FONT_FAMILY, fontsize=17, restore=restore
    )
    def draw(value, *, scale):
        """Return ``value * scale`` and the font settings in effect."""
        return value * scale, _font_settings()

    with plt.rc_context(_USER_FONT_SETTINGS):
        before = _font_settings()

        result, inside = draw(2, scale=3)

        after = _font_settings()

    assert draw.__name__ == "draw"
    assert result == 6
    assert inside == ([_OTHER_FONT_FAMILY], 17.0)
    assert after == (before if restore else inside)


def test_with_custom_matplotlib_settings_restores_after_error():
    @with_custom_matplotlib_settings(fontfamily=_OTHER_FONT_FAMILY)
    def fail():
        raise RuntimeError("drawing failed")

    with plt.rc_context(_USER_FONT_SETTINGS):
        before = _font_settings()

        with pytest.raises(RuntimeError, match="drawing failed"):
            fail()

        assert _font_settings() == before


# ===========================================================================
# yyplot
# ===========================================================================

#: (true values, predicted values) of the data sets drawn by ``yyplot``.
_YY_TRAIN = ((1.0, 2.0, 3.0, 4.0, 5.0), (1.2, 1.9, 3.3, 3.8, 5.1))
_YY_TEST = ((1.5, 2.5, 3.5, 4.5), (1.0, 2.9, 3.4, 5.0))


def _metric_values(
    y_true: Sequence[float], y_pred: Sequence[float]
) -> tuple[str, str, str, str]:
    """Return the texts of R2, RMSE, MAE and MSE as ``yyplot`` formats them."""
    mse = mean_squared_error(y_true, y_pred)
    return (
        f"{r2_score(y_true, y_pred):#.2f}",
        f"{np.sqrt(mse):#.3g}",
        f"{mean_absolute_error(y_true, y_pred):#.3g}",
        f"{mse:#.3g}",
    )


def _displayed_characters(text: str) -> str:
    """Return the characters matplotlib draws for one line of ``text``.

    Matplotlib draws a string with an even number of unescaped dollar
    signs as mathtext (spaces inside math are dropped), and any other string
    as plain text in which ``\\$`` stands for ``$``.
    """
    if not is_math_text(text):
        return text.replace("\\$", "$")
    glyphs = MathTextParser("path").parse(text)[3]
    return "".join(chr(glyph[2]) for glyph in glyphs)


def _layout(fig: Figure) -> tuple[tuple[float, ...], list[tuple[float, ...]]]:
    """Return the subplot parameters and the axes positions of ``fig``."""
    params = fig.subplotpars
    return (
        (
            params.left,
            params.right,
            params.bottom,
            params.top,
            params.wspace,
            params.hspace,
        ),
        [tuple(ax.get_position(original=True).bounds) for ax in fig.axes],
    )


def _is_blank(ax: Axes) -> bool:
    return (
        len(ax.collections) == 0
        and len(ax.lines) == 0
        and len(ax.texts) == 0
        and ax.get_legend() is None
    )


def test_yyplot_draws_values_and_metrics():
    ax = yyplot(*_YY_TRAIN, *_YY_TEST, metrics=("r2", "rmse", "mae", "mse"))

    try:
        # One scatter per data set at (true, predicted), labeled by default.
        assert len(ax.collections) == 2
        for collection, (y_true, y_pred) in zip(
            ax.collections, (_YY_TRAIN, _YY_TEST)
        ):
            np.testing.assert_allclose(
                np.asarray(collection.get_offsets()),
                np.column_stack([y_true, y_pred]),
            )
        assert _legend_labels(ax) == ["train", "test"]
        assert (ax.get_xlabel(), ax.get_ylabel()) == (
            "$y_\\mathrm{true}$",
            "$y_\\mathrm{pred}$",
        )

        # The line y = x and both axes span all the values with a 5% margin.
        values = np.concatenate([*_YY_TRAIN, *_YY_TEST])
        margin = np.ptp(values) * 0.05
        limits = (values.min() - margin, values.max() + margin)
        (line,) = ax.lines
        np.testing.assert_allclose(line.get_xdata(), limits)
        np.testing.assert_allclose(line.get_ydata(), limits)
        assert ax.get_xlim() == pytest.approx(limits)
        assert ax.get_ylim() == pytest.approx(limits)

        # The metrics of each data set, in the order of ``metrics``.
        (text,) = ax.texts
        expected = []
        for label, (y_true, y_pred) in zip(
            ("train", "test"), (_YY_TRAIN, _YY_TEST)
        ):
            r2, rmse, mae, mse = _metric_values(y_true, y_pred)
            expected += [
                f"$R^2$ ({label})$ = {r2}$",
                f"RMSE ({label})$ = {rmse}$",
                f"MAE ({label})$ = {mae}$",
                f"MSE ({label})$ = {mse}$",
            ]
        assert text.get_text().split("\n") == expected
    finally:
        plt.close(ax.figure)


def test_yyplot_without_labels():
    ax = yyplot(*_YY_TRAIN)

    try:
        r2, rmse, _, _ = _metric_values(*_YY_TRAIN)
        (text,) = ax.texts
        assert text.get_text().split("\n") == [
            f"$R^2$$ = {r2}$",
            f"RMSE$ = {rmse}$",
        ]
        assert ax.get_legend() is None
    finally:
        plt.close(ax.figure)


def test_yyplot_treats_empty_label_as_no_label():
    # Alone: no legend at all, and the unlabeled metric lines.
    ax = yyplot(*_YY_TRAIN, labels=[""])

    try:
        r2, rmse, _, _ = _metric_values(*_YY_TRAIN)
        (text,) = ax.texts
        assert text.get_text().split("\n") == [
            f"$R^2$$ = {r2}$",
            f"RMSE$ = {rmse}$",
        ]
        assert ax.get_legend() is None
    finally:
        plt.close(ax.figure)

    # Next to a labeled data set: only that one is in the legend.
    ax = yyplot(*_YY_TRAIN, *_YY_TEST, labels=["", "test"])

    try:
        test_r2, test_rmse, _, _ = _metric_values(*_YY_TEST)
        (text,) = ax.texts
        assert text.get_text().split("\n") == [
            f"$R^2$$ = {r2}$",
            f"RMSE$ = {rmse}$",
            f"$R^2$ (test)$ = {test_r2}$",
            f"RMSE (test)$ = {test_rmse}$",
        ]
        assert _legend_labels(ax) == ["test"]
    finally:
        plt.close(ax.figure)


def test_yyplot_keeps_layout_of_figure_of_given_axes():
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 3.0))
    fig.subplots_adjust(left=0.3, right=0.9, bottom=0.25, top=0.8, wspace=0.6)
    layout = _layout(fig)

    try:
        ax = yyplot(*_YY_TRAIN, *_YY_TEST, ax=axes[0])

        assert ax is axes[0]
        assert len(ax.collections) == 2
        assert _layout(fig) == layout
        assert _is_blank(axes[1])
    finally:
        plt.close(fig)


def test_yyplot_legend_keeps_labeled_artists_of_given_axes():
    fig, ax = plt.subplots()
    ax.plot([0.0, 6.0], [0.5, 6.5], label="reference")

    try:
        yyplot(*_YY_TRAIN, *_YY_TEST, ax=ax)

        assert _legend_labels(ax) == ["reference", "train", "test"]
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "given_axes", [False, True], ids=["new-figure", "given-axes"]
)
def test_yyplot_rejects_unknown_metric_before_drawing(given_axes):
    fig, ax = plt.subplots() if given_axes else (None, None)
    figure_numbers = plt.get_fignums()

    try:
        with pytest.raises(ValueError, match="'rmsle'"):
            yyplot(*_YY_TRAIN, metrics=("r2", "rmsle"), ax=ax)

        assert plt.get_fignums() == figure_numbers
        if ax is not None:
            assert _is_blank(ax)
    finally:
        if fig is not None:
            plt.close(fig)


@pytest.mark.parametrize(
    "label",
    [
        "$x$",
        "cost $5",
        "y_train",
        "_valid",
        "50% CI",
        "a^b {c} #d &e ~f",
        "\\alpha",
        "\\$",
        "ends with \\",
    ],
)
def test_yyplot_shows_labels_as_written(label):
    ax = yyplot(*_YY_TRAIN, labels=[label])
    fig = ax.figure

    try:
        # Mathtext errors are raised when the texts are drawn.
        fig.canvas.draw()

        (legend_text,) = ax.get_legend().get_texts()
        assert _displayed_characters(legend_text.get_text()) == label

        r2, rmse, _, _ = _metric_values(*_YY_TRAIN)
        (text,) = ax.texts
        assert [
            _displayed_characters(line) for line in text.get_text().split("\n")
        ] == [f"R2 ({label})={r2}", f"RMSE ({label})={rmse}"]
    finally:
        plt.close(fig)
