if __name__ == '__main__':
    from yikit.tools import is_notebook
    print(is_notebook())
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
    get_dist_figure(ngb.pred_dist(X_test), y_test, titles = ['a'] * len(y_test)).savefig(os.path.join(os.path.dirname(__file__), 'sample_dist_figure.png'))
    
    