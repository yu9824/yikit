
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

from sklearn.feature_selection import SelectorMixin
from sklearn.base import BaseEstimator
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from scipy.stats import pearsonr
import pandas as pd
import numpy as np



class FilterSelector(SelectorMixin, BaseEstimator):
    def __init__(self, r = 0.9, alpha = 0.05):
        '''
        r: 相関係数を高いとする閾値．これより高いと相関が高いとして，削除対象とする．
        alpha: 有意水準．これよりp値が低い時，相関係数の値が信頼に足ると判断．特別な理由がない限りはこれを変えないのがベター．
        '''
        self.r = r
        self.alpha = alpha

    def fit(self, X, y = None):
        '''
        X: pd.DataFrame or 2次元のnp.ndarray or 2次元のlist．すでに作成された特徴量空間．
        y: 不要．
        '''
        # checkかつnp.ndarrayに変換される．
        X = check_array(X)

        # よくつかうので列数を変数で保存
        n_features = X.shape[1]

        # 相関係数とそのp値を入れるリストの鋳型を作成
        # [i, j, 0]が相関係数， [i, j, 1]がp値
        self.corr_ = np.empty((n_features, n_features, 2), dtype = float)

        # i番目の要素は，i番目と相関係数がcorr以上かつp値が有意水準alpha未満のものの集合．これの鋳型を作成
        pair = [set() for _ in range(n_features)]

        for i in range(n_features):
            for j in range(i, n_features):
                if i == j:  # 一緒の時はそもそも相関係数1, p値0は自明なので例外処理
                    self.corr_[i, i] = [1.0, 0.0]
                else:
                    self.corr_[i, j] = list(pearsonr(X[:, i], X[:, j]))
                    self.corr_[j, i] = self.corr_[i, j]
                    if self.corr_[i, j, 0] > self.r and self.corr_[i, j, 1] < self.alpha:    # 相関係数が高く， かつp値が小さい (信頼に足る) とき．
                        # i, j番目の要素の集合にそれぞれj, iを追加
                        pair[i].add(j)
                        pair[j].add(i)

        # 何個相関係数が高い係数が存在するのかの値を求める．
        def _delete_recursive(pair, boolean = np.ones(n_features, dtype = bool)):
            # 相関係数が高いものがいくつあるのかのnp.ndarrayを作成
            order_pair = np.array([len(s) for s in pair])

            # これがすべて0のとき，もう削除は終わってるので終了処理
            if np.sum(order_pair) == 0:
                return boolean

            # 一番相関係数が高いペア数を多く持っているものの要素番号 (index) を取得
            i_max = np.argmax(order_pair)

            # 相関係数高いペアからその番号を削除
            pair = [p.difference({i_max}) for p in pair]

            # その集合は削除かつ，その特徴量はbooleanをFalseに．
            pair[i_max] = set()
            boolean[i_max] = False
            return _delete_recursive(pair, boolean)

        self.support_ = _delete_recursive(pair)

        return self

    def _get_support_mask(self):
        check_is_fitted(self, 'support_')
        return self.support_

if __name__ == '__main__':
    from pdb import set_trace
    from sklearn.datasets import load_boston

    data = load_boston()
    X = pd.DataFrame(data.data, columns = data.feature_names)
    # X = pd.concat([X, pd.Series(np.ones(X.shape[0]))], axis = 1)

    selector = FilterSelector()
    selector.fit(X)

    set_trace()
