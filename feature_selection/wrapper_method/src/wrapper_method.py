from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_selection import SelectorMixin
from sklearn.base import BaseEstimator
from sklearn.utils import check_array
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from tqdm import tqdm

#  cross_validateでやったほうがより現実的？

class WrapperSelector(BaseEstimator, SelectorMixin):
    def __init__(self, estimator = RandomForestRegressor(), random_state = None, max_iter = 100, perc = 'auto', two_step = False, verbose = 2, n_estimators = 'auto', max_shuf = 10000):
        self.estimator = estimator
        self.random_state = random_state
        self.max_iter = max_iter
        self.perc = perc
        self.two_step = two_step
        self.verbose = verbose
        self.n_estimators = n_estimators
        self.max_shuf = max_shuf

    def fit(self, X, y):
        X, y = check_X_y(X, y)

        # np.ndarrayに変換
        X = np.array(X)
        y = np.array(y)

        if self.perc == 'auto':
            # ランダムに並べ替えてどれくらいでてしまうのかを調べる
            np.random.seed(self.random_state)
            self.pears_ = []
            for shuf in tqdm(range(self.max_shuf), desc = 'Calc r_ccmax'):
                X_shuffled = shuffle(X, random_state = np.random.randint(0, 2 ** 32))
                temp = [pearsonr(X_shuffled[:, i], y)[0] for i in range(X_shuffled.shape[1])]
                self.pears_.extend(temp)
            r_ccmax = max(self.pears_)
            self.perc = 100 * (1 - r_ccmax)
        elif isinstance(self.perc, (int, float)):
            pass
        else:
            raise NotImplementedError("The 'perc' must be 'auto' or a number greater than 0 and less than or equal to 100.")

        # boruta
        self.feature_selector_ = BorutaPy(self.estimator, n_estimators = self.n_estimators, two_step = self.two_step, verbose = self.verbose, random_state = self.random_state, perc = self.perc, max_iter = self.max_iter)
        self.feature_selector_.fit(X, y)
        
        return self

    def _get_support_mask(self):
        check_is_fitted(self, 'feature_selector_')
        return self.feature_selector_.support_


if __name__ == '__main__':
    from sklearn.datasets import load_boston
    boston = load_boston()
    X = pd.DataFrame(boston['data'], columns = boston['feature_names'])
    y = pd.Series(boston['target'], name = 'PRICE')
    selector = WrapperSelector(cv = 5, max_iter = 100)
    print(selector.fit_transform(X, y))
    # print(selector.get_support())