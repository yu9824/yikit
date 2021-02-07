import matplotlib.pyplot as plt
import pandas as pd
import optuna

def get_learning_curve(study, loc = 'best'):
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
    plt.rcParams['font.family'] = 'Helvetica'

    fig = plt.figure(facecolor = 'white')
    ax = fig.add_subplot(111)

    ax.scatter(df_values.index, df_values['value'], s = 10, c = '#00293c', label = 'Objective Value')
    ax.plot(df_values.index, df_values['best_value'], c = '#f62a00', zorder = 0, label = 'Best Value')

    ax.set_xlabel('Trials')
    ax.set_ylabel('Objective Values')

    ax.legend(facecolor = '#f0f0f0', edgecolor = 'None', fontsize = 10, loc = loc)
    return fig, ax