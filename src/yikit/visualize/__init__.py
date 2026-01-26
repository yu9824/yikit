from yikit.helpers import is_installed

from ._ngboost import get_dist_figure, get_learning_curve_gb
from ._permutation_importance import SummarizePI
from ._utils import set_font, with_custom_matplotlib_settings

__all__ = [
    "SummarizePI",
    "set_font",
    "with_custom_matplotlib_settings",
    "get_dist_figure",
    "get_learning_curve_gb",
]

if is_installed("optuna"):
    from ._optuna import get_learning_curve_optuna

    __all__ += ["get_learning_curve_optuna"]
