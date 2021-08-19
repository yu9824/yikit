if __name__ == '__main__':
    # SummarizePI test
    from yikit.tools.visualization import SummarizePI
    from sklearn.inspection import permutation_importance
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.datasets import load_boston
    from sklearn.model_selection import train_test_split
    import pandas as pd
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
