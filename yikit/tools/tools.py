
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

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import optuna
import seaborn as sns
import sys


def is_notebook():
    return 'ipykernel' in sys.modules
'''
例
if 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm
'''

def get_learning_curve(study, loc = 'best', fontfamily='Helvetica'):
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
    return fig, ax

class SummarizePI:
    def __init__(self, importances):
        '''

        Parameters
        ----------
        importances : pandas.DataFrame
            index: features
            columns: n_repeats
        '''
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

        return self.fig, self.ax

    def get_data(self):
        pass

if __name__ == '__main__':
    from sklearn.inspection import permutation_importance
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    from pdb import set_trace

    boston = load_boston()
    X = pd.DataFrame(boston.data, columns = boston.feature_names)
    y = pd.Series(boston.target, name = 'PRICEE')

    SEED = 334

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = SEED)
    
    rf = RandomForestRegressor(random_state = SEED, n_jobs = -1)

    rf.fit(X_train, y_train)

    pi = permutation_importance(rf, X_test, y_test, random_state = SEED, n_jobs = -1)
    spi = SummarizePI(pd.DataFrame(pi.importances, index = X.columns))
    spi.get_figure()
