
'''
Copyright (c) 2021 yu9824

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import optuna
import seaborn as sns
import sys
from decimal import Decimal
from math import ceil
import warnings

from sklearn.utils.validation import check_array
from yikit.tools import is_notebook

if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

def get_learning_curve(study, loc = 'best', fontfamily='Helvetica', return_axis = False):
    """get_leraning_curve

    Parameters
    ----------
    study : optuna.study.Study
        
    loc : str, optional
        legend's location, by default 'best'
    fontfamily : str, optional
        fontfamily, by default 'Helvetica'
    return_axis : bool, optional
        return axis or not, by default False

    Returns
    -------
    if return_axis is True:
        tuple (matplotlib.pyplot.figure, matplotlib.pyplot.axis)
    else:
        matplotlib.pyplot.figure
    """
    # 2: maximize, 1: minimize
    if study.direction == optuna.study.StudyDirection(1):
        min_or_max = min
        temp = float('inf')
    elif study.direction == optuna.study.StudyDirection(2):
        min_or_max = max
        temp = - float('inf')
    df_values = pd.DataFrame([trial.value for trial in study.trials], columns = ['value'])
    
    best_values = []
    for v in df_values.loc[:, 'value']:
        temp = min_or_max(temp, v)
        best_values.append(temp)
    df_values['best_value'] = best_values

    plt.rcParams['font.size'] = 13
    plt.rcParams['font.family'] = fontfamily

    fig = plt.figure(facecolor = 'white')
    ax = fig.add_subplot(111)

    ax.scatter(df_values.index, df_values['value'], s = 10, c = '#00293c', label = 'Objective Value')
    ax.plot(df_values.index, df_values['best_value'], c = '#f62a00', zorder = 0, label = 'Best Value')

    ax.set_xlabel('Trials')
    ax.set_ylabel('Objective Values')

    ax.legend(facecolor = '#f0f0f0', edgecolor = 'None', fontsize = 10, loc = loc)

    fig.tight_layout()
    if return_axis:
        return fig, ax
    else:
        return fig

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

    def get_figure(self, fontfamily='Helvetica'):
        # 平均をとる．
        imp = self.importances.mean(axis = 1)

        # 「規格化」したのち，大きい順に並べ替えて，sns.bairplotのためにtranspose()
        df_imp = pd.DataFrame(imp / np.sum(imp), columns = ['importances']).sort_values('importances', ascending=False).transpose()

        # レイアウトについて
        plt.rcParams['font.size'] = 13
        plt.rcParams['font.family'] = 'Helvetica'

        # 重要度の棒グラフを描画
        self.fig = plt.figure(facecolor = 'white')
        self.ax = self.fig.add_subplot(111)

        sns.barplot(data = df_imp, ax = self.ax, orient = 'h')

        self.fig.tight_layout()

        return self.fig, self.ax

    def get_data(self):
        pass

def get_dist_figure(y_true, y_pred, y_dist, keep_y_range = True, return_axis = False, verbose = True, titles = []):
    offset = np.ptp(y_pred) * 0.05
    y_range = np.linspace(min(y_pred)-offset, max(y_pred)+offset, 200).reshape((-1, 1))
    dist_values = y_dist.pdf(y_range).transpose()

    y_true = check_array(y_true, ensure_2d=False)
    y_pred = check_array(y_pred, ensure_2d=False)

    if not np.allclose(y_dist.mean(), y_pred):
        warnings.warn('`y_dist.mean()` and `y_pred` is not close.')
        

    n_samples = len(y_true)
    n_rows = ceil(Decimal(n_samples).sqrt())
    n_cols = n_samples // n_rows + int(n_samples % n_rows > 0)
    fig, axes = plt.subplots(n_rows, n_cols, facecolor='white', dpi=144, figsize=(6.4*n_rows, 4.8*n_cols))
    
    if verbose:
        pbar = tqdm(total=n_cols * n_rows + 1)
    # 一つずつ分布を書いていく．
    for idx in range(n_samples):
        ax = axes[idx//n_cols][idx%n_cols]
        ax.plot(y_range, dist_values[idx], c = '#022C5E')
        
        prob_max_temp = max(dist_values[idx])   # このidxにおける確率密度の最大値
        ax.vlines(y_true[idx], 0, prob_max_temp, "#AB4E15", label="ground truth")
        ax.vlines(y_pred[idx], 0, prob_max_temp, "r", label="pred")
        ax.legend(loc="best", facecolor='#f0f0f0', edgecolor='None')
        if titles:
            ax.set_title("{0}".format(titles[idx]))
        else:
            ax.set_title("idx: {0}".format(idx))
            
        
        ax.set_xlim(y_range[0], y_range[-1])
        if keep_y_range:
            ax.set_ylim(None, np.max(dist_values) * 1.05)

        if verbose:
            pbar.update(1)
    # 余ったところを消す．
    for idx in range(n_samples, n_rows * n_cols):
        ax = axes[idx//n_cols][idx%n_cols]
        ax.axis("off")
        if verbose:
            pbar.update(1)

    fig.tight_layout()

    if verbose:
        pbar.update(1)
        pbar.close()
    if return_axis:
        return fig, axes
    else:
        return fig
