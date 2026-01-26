import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMRegressor
from ngboost import NGBRegressor
from PIL import Image
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from yikit.models import Objective
from yikit.visualize import (
    SummarizePI,
    get_dist_figure,
    get_learning_curve_gb,
    get_learning_curve_optuna,
)

SEED = 334

# 参照画像のディレクトリ
REFERENCE_IMGS_DIR = Path(__file__).parent / "imgs"

# 実行モード: pytest では compare、直実行では save に切り替える
_FIGURE_MODE = "compare"  # "compare" | "save"

# 参照画像の出力を安定させるため dpi を固定
_SAVEFIG_DPI = 72


def compare_images(img_path1: Path, img_path2: Path) -> bool:
    """2つの画像ファイルが同じ（近い）かどうかを判定

    Parameters
    ----------
    img_path1 : Path
        比較する画像ファイル1のパス
    img_path2 : Path
        比較する画像ファイル2のパス

    Returns
    -------
    bool
        画像が同じ（近い）場合True、異なる場合False
    """
    # まず matplotlib の比較ユーティリティで判定（RMS ベースで環境差に強い）
    try:
        from matplotlib.testing.compare import compare_images as mpl_compare

        # tol は RMS 許容値。小さすぎるとフォントやAA差で落ちるため少し緩めに設定。
        result = mpl_compare(
            str(img_path2),  # expected
            str(img_path1),  # actual
            tol=20,
        )
        return result is None
    except Exception:
        # フォールバック: PIL + numpy の簡易比較
        pass

    img1 = Image.open(img_path1).convert("RGB")
    img2 = Image.open(img_path2).convert("RGB")

    # サイズが異なる場合はFalse
    if img1.size != img2.size:
        return False

    # 画像をnumpy配列に変換して比較
    arr1 = np.array(img1, dtype=np.int16)
    arr2 = np.array(img2, dtype=np.int16)

    # matplotlib の微小差（アンチエイリアス/フォント差）を許容するため、閾値付きで比較
    diff = np.abs(arr1 - arr2)
    max_diff = int(diff.max())
    mean_diff = float(diff.mean())

    # 経験的な許容範囲（完全一致に近いが、環境差で数値がわずかに揺れても落ちない）
    return (max_diff <= 8) and (mean_diff <= 0.5)


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
        assert compare_images(tmp_path, reference_path), (
            "生成された画像が参照画像と異なります"
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


def test_get_dist_figure():
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
    # pytest（conftest）と同じ backend に揃える（直実行で参照画像を作っても一致するように）
    plt.switch_backend("Agg")

    # `python3 tests/test_visualize.py` 直実行時は、pytest と同じ test 関数を呼ぶ
    test_summarize_pi()
    test_get_dist_figure()
    test_learning_curve_optuna()
    test_learning_curve_ngboost()
    test_learning_curve_lightgbm()
