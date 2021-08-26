if __name__ == '__main__':
    from yikit.tools import is_notebook
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    import pandas as pd
    import matplotlib.pyplot as plt
    from pdb import set_trace
    import os

    # データセットの準備等
    boston = load_boston()
    X = pd.DataFrame(boston.data, columns = boston.feature_names)
    y = pd.Series(boston.target, name = 'PRICE')

    SEED = 334

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = SEED)


    # SummarizePI test
    # from yikit.tools.visualization import SummarizePI
    # from sklearn.inspection import permutation_importance
    
    # rf = RandomForestRegressor(random_state = SEED, n_jobs = -1)

    # rf.fit(X_train, y_train)

    # pi = permutation_importance(rf, X_test, y_test, random_state = SEED, n_jobs = -1)
    # spi = SummarizePI(pd.DataFrame(pi.importances, index = X.columns))
    # spi.get_figure()
    # plt.show()
    # plt.close()

    # get_dist_figure test
    from yikit.tools.visualization import get_dist_figure
    from ngboost import NGBRegressor
    ngb = NGBRegressor(random_state=SEED, verbose=False).fit(X_train, y_train)
    get_dist_figure(ngb.pred_dist(X_test), y_test, titles = ['a'] * len(y_test), verbose=False).savefig(os.path.join(os.path.dirname(__file__), 'sample_dist_figure.png'))

    # get_learning_curve_optuna
    # from yikit.tools.visualization import get_learning_curve_optuna
    # import optuna
    # from yikit.models import Objective
    # rf = RandomForestRegressor(random_state=SEED, n_jobs = -1)
    # objective = Objective(rf, X_train, y_train, random_state=SEED, n_jobs=-1)
    # study = optuna.create_study(sampler = objective.sampler, direction='maximize')
    # study.optimize(objective, n_trials=10)
    # get_learning_curve_optuna(study).savefig(os.path.join(os.path.dirname(__file__), 'sample_learning_curve_optuna.png'))


    # get_learning_curve_gb
    # from yikit.tools.visualization import get_learning_curve_gb
    # from ngboost import NGBRegressor
    # from sklearn.tree import DecisionTreeRegressor
    # ngb = NGBRegressor(random_state=SEED, Base=DecisionTreeRegressor(random_state=SEED))
    # ngb.fit(X_train, y_train, X_val=X_test, Y_val=y_test)
    # get_learning_curve_gb(ngb).savefig(os.path.join(os.path.dirname(__file__), 'sample_learning_curve_ngboost.png'))

    # from yikit.tools.visualization import get_learning_curve_gb
    # from lightgbm import LGBMRegressor
    # lgbm = LGBMRegressor(random_state=SEED, n_jobs=-1)
    # lgbm.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], eval_names=['train', 'test'])
    # get_learning_curve_gb(lgbm).savefig(os.path.join(os.path.dirname(__file__), 'sample_learning_curve_lightgbm.png'))
    
    
