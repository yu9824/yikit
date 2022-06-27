if __name__ == '__main__':
    from sklearn.datasets import load_diabetes
    import pandas as pd
    from yikit.feature_selection import BorutaPy
    from sklearn.ensemble import RandomForestRegressor

    diabetes = load_diabetes()
    X = pd.DataFrame(diabetes['data'], columns = diabetes['feature_names'])
    y = pd.Series(diabetes['target'])
    print(BorutaPy(RandomForestRegressor(n_jobs=-1, random_state=334), random_state=334).fit(X, y).support_)