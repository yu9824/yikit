
'''
Copyright © 2021 yu9824

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

from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_selection import SelectorMixin
from sklearn.base import BaseEstimator
from sklearn.utils import check_array
from sklearn.utils import check_X_y
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import shuffle
import pandas as pd
from scipy.stats import pearsonr
from yikit.tools import is_notebook
if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


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
        # checkかつnumpy.ndarrayに変換
        X, y = check_X_y(X, y)

        # random_state
        self.rng_ = check_random_state(self.random_state)

        if self.perc == 'auto':
            # ランダムに並べ替えてどれくらいでてしまうのかを調べ，自動で決める．
            self.pears_ = []    # 相関係数を足していくリスト
            for shuf in tqdm(range(self.max_shuf), desc = 'Calc r_ccmax') if self.verbose else range(self.max_shuf):
                X_shuffled = shuffle(X, random_state = self.rng_)
                temp = [pearsonr(X_shuffled[:, i], y)[0] for i in range(X_shuffled.shape[1])]
                self.pears_.extend(temp)
            r_ccmax = max(self.pears_)
            self.perc = 100 * (1 - r_ccmax)
        elif isinstance(self.perc, (int, float)):
            pass
        else:
            raise NotImplementedError("The 'perc' must be 'auto' or a number greater than 0 and less than or equal to 100.")

        # boruta
        self.feature_selector_ = BorutaPy(self.estimator, n_estimators = self.n_estimators, two_step = self.two_step, verbose = self.verbose, random_state = self.rng_, perc = self.perc, max_iter = self.max_iter)
        self.feature_selector_.fit(X, y)
        
        return self

    def _get_support_mask(self):
        check_is_fitted(self, 'feature_selector_')
        return self.feature_selector_.support_


if __name__ == '__main__':
    from pdb import set_trace
    from sklearn.datasets import load_boston

    boston = load_boston()
    X = pd.DataFrame(boston['data'], columns = boston['feature_names'])
    y = pd.Series(boston['target'], name = 'PRICE')
    selector = WrapperSelector(estimator = RandomForestRegressor(n_jobs = -1))
    print(selector.fit_transform(X, y))