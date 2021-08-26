if __name__ == '__main__':
    from sklearn.datasets import load_boston
    import pandas as pd
    from yikit.feature_selection import BorutaPy
    from sklearn.ensemble import RandomForestRegressor

    boston = load_boston()
    X = pd.DataFrame(boston['data'], columns = boston['feature_names'])
    y = pd.Series(boston['target'], name = 'PRICE')
    print(BorutaPy(RandomForestRegressor(n_jobs=-1, random_state=334), random_state=334).fit(X, y).support_)