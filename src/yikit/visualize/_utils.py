import platform
from functools import wraps
from typing import Callable, Optional, TypeVar

import matplotlib.pyplot as plt
from sklearn.utils import Bunch

F = TypeVar("F", bound=Callable)

COLORS = Bunch(
    train="#283655",
    test="#cf3721",
    val="#cf3721",
)


def set_font(fontfamily: Optional[str] = None, fontsize: int = 13):
    if fontfamily is None:
        fontfamily = _default_fontfamily()

    plt.rcParams["font.size"] = fontsize
    plt.rcParams["font.family"] = fontfamily


def _default_fontfamily():
    if platform.system() == "Darwin":
        return "Helvetica"
    else:  # Window or Linux
        return "DejaVu Sans"


def with_custom_matplotlib_settings(
    fontfamily: Optional[str] = None, fontsize: int = 13, restore: bool = True
):
    """matplotlibのデフォルト設定を適用するデコレータ

    Parameters
    ----------
    fontfamily : Optional[str], optional
        フォントファミリー。Noneの場合はデフォルトフォントを使用, by default None
    fontsize : int, optional
        フォントサイズ, by default 13
    restore : bool, optional
        関数実行後に設定を元に戻すかどうか, by default True

    Returns
    -------
    Callable
        デコレータ関数

    Examples
    --------
    >>> @with_matplotlib_defaults(fontfamily="Helvetica", fontsize=14)
    ... def plot_data():
    ...     plt.plot([1, 2, 3], [1, 4, 9])
    ...     return plt.gcf()
    """
    if fontfamily is None:
        fontfamily = _default_fontfamily()

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 現在の設定を保存
            if restore:
                original_font_size = plt.rcParams["font.size"]
                original_font_family = plt.rcParams["font.family"]

            # 設定を適用
            plt.rcParams["font.size"] = fontsize
            plt.rcParams["font.family"] = fontfamily

            try:
                # 関数を実行
                result = func(*args, **kwargs)
                return result
            finally:
                # 設定を復元
                if restore:
                    plt.rcParams["font.size"] = original_font_size
                    plt.rcParams["font.family"] = original_font_family

        return wrapper  # type: ignore

    return decorator
