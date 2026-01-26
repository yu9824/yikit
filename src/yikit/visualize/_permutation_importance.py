"""Permutation importance visualization.

This module provides utilities for visualizing permutation importance
results from machine learning models.
"""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from yikit.visualize._utils import set_font


class SummarizePI:
    def __init__(self, importances):
        """Summarize permutation importances

        Parameters
        ----------
        importances : pandas.DataFrame
            index: features
            columns: n_repeats
        """
        self.importances = importances

    def get_figure(self, fontfamily: Optional[str] = None):
        # 平均をとる．
        imp = self.importances.mean(axis=1)

        # 「規格化」したのち，大きい順に並べ替えて，sns.bairplotのためにtranspose()
        df_imp = (
            pd.DataFrame(imp / np.sum(imp), columns=["importances"])
            .sort_values("importances", ascending=False)
            .transpose()
        )

        # レイアウトについて
        set_font(fontfamily=fontfamily)

        # 重要度の棒グラフを描画
        self.fig = plt.figure(facecolor="white")
        self.ax = self.fig.add_subplot(111)

        sns.barplot(data=df_imp, ax=self.ax, orient="h")

        self.fig.tight_layout()

        return self.fig, self.ax

    def get_data(self):
        pass
