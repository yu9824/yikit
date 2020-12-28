from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
import pandas as pd
import numpy as np

class WrapperMethod:
    def __init__(self, estimator = RandomForestRegressor(), random_state = None, max_iter = 100, perc = 80, two_step = False, verbose = 2, n_estimators = 'auto'):
        self.estimator = estimator
        self.random_state = random_state
        self.feature_selector = BorutaPy(self.estimator, n_estimators = n_estimators, two_step = two_step, verbose = verbose, random_state = self.random_state, perc = perc, max_iter = max_iter)

    def __call__(self, X, y, cv = None, threshold = 1):
        '''
        cv: int or None (default is None).
        threshold: int
        '''
        if type(X) in (pd.DataFrame, pd.Series):
            X = X.values
        if type(y) in (pd.DataFrame, pd.Series):
            y = y.values

        if cv is None:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = self.random_state)
            self.feature_selector.fit(X_train, y_train)
            boolean_feature_selection = np.array(self.feature_selector.support_)
        elif type(cv) == int:
            kf = KFold(n_splits = cv, shuffle = True, random_state = self.random_state)
            booleans_feature_selection = []
            for i_train, i_test in kf.split(X, y):
                X_train = X[i_train]
                X_test = X[i_test]
                y_train = y[i_train]
                y_test = y[i_test]
                self.feature_selector.fit(X_train, y_train)
                booleans_feature_selection.append(self.feature_selector.support_)
            
            # 閾値以上回出てきたら採用
            booleans_feature_selection = np.array(booleans_feature_selection)
            boolean_feature_selection = booleans_feature_selection.sum(axis = 0) >= threshold
        else:
            NotImplementedError

        return boolean_feature_selection

if __name__ == '__main__':
    from sklearn.datasets import load_boston
    boston = load_boston()
    X = pd.DataFrame(boston['data'], columns = boston['feature_names'])
    y = pd.Series(boston['target'], name = 'PRICE')
    wrapper_method = WrapperMethod(RandomForestRegressor(random_state = 334), random_state=334)
    wrapper_method(X, y, cv = 5, threshold = 1)