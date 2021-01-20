from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_selection import SelectorMixin
from sklearn.base import BaseEstimator
from sklearn.utils import check_array
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted
import pandas as pd
import numpy as np

class WrapperSelector(BaseEstimator, SelectorMixin):
    def __init__(self, estimator = RandomForestRegressor(), random_state = None, max_iter = 100, perc = 80, two_step = False, verbose = 2, n_estimators = 'auto', cv = 5, threshold = None):
        self.estimator = estimator
        self.random_state = random_state
        self.max_iter = max_iter
        self.perc = perc
        self.two_step = two_step
        self.verbose = verbose
        self.n_estimators = n_estimators
        self.cv = cv
        self.threshold = threshold

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        # np.ndarrayに変換
        X = np.array(X)
        y = np.array(y)

        # boruta
        self.feature_selector_ = BorutaPy(self.estimator, n_estimators = self.n_estimators, two_step = self.two_step, verbose = self.verbose, random_state = self.random_state, perc = self.perc, max_iter = self.max_iter)

        # cvの検査
        if self.cv is None:
            self.feature_selector_.fit(X, y)
            self.support_ = self.feature_selector_.support_
        elif isinstance(self.cv, int) and self.cv > 1:
            # thresholdの判定
            if self.threshold is None:
                self.threshold = self.cv
            elif isinstance(self.threshold, int):
                if self.threshold > self.cv:
                    raise ValueError('Threshold must be less than or equal to cv.')
            else:
                raise NotImplementedError
            
            # KFold
            kf = KFold(n_splits = self.cv, shuffle = True, random_state = self.random_state)
            flags = []
            for i_train, i_test in kf.split(X, y):
                X_selected = X[i_train]
                y_selected = y[i_train]

                self.feature_selector_.fit(X_selected, y_selected)
                flags.append(self.feature_selector_.support_)

            # 閾値以上回出てきたら採用
            self.support_ = np.array(flags).sum(axis = 0) >= self.threshold
        else:
            raise NotImplementedError
        
        return self

    def _get_support_mask(self):
        check_is_fitted(self, ['support_', 'feature_selector_'])
        return self.support_


if __name__ == '__main__':
    from sklearn.datasets import load_boston
    boston = load_boston()
    X = pd.DataFrame(boston['data'], columns = boston['feature_names'])
    y = pd.Series(boston['target'], name = 'PRICE')
    selector = WrapperSelector(cv = 5, max_iter = 100)
    print(selector.fit_transform(X, y))
    # print(selector.get_support())