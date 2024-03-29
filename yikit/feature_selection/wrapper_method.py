
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

import boruta
from sklearn.utils import check_X_y
from sklearn.utils import check_random_state
from sklearn.utils import shuffle
from sklearn.utils.validation import check_is_fitted
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.cross_decomposition import PLSRegression
import numpy as np
from scipy.stats import pearsonr
import sys
import warnings
from joblib import Parallel, delayed
from typing import Any

from yikit.tools import is_notebook


class BorutaPy(boruta.BorutaPy):
    def __init__(self, estimator, n_estimators='auto', perc='auto', alpha=0.05,
                 two_step=True, max_iter=100, random_state=None, verbose=1,
                 max_shuf=10000, n_jobs=None):
        """
        This docstring is modified from and uses parts of scikit-learn-contrib/boruta_py, which is a class inheritor under the BSD 3 clause license.
        https://github.com/scikit-learn-contrib/boruta_py/blob/master/boruta/boruta_py.py

        Improved Python implementation of the Boruta R package.
        The improvements of this implementation include:
        - Faster run times:
            Thanks to scikit-learn's fast implementation of the ensemble methods.
        - Scikit-learn like interface:
            Use BorutaPy just like any other scikit learner: fit, fit_transform and
            transform are all implemented in a similar fashion.
        - Modularity:
            Any ensemble method could be used: random forest, extra trees
            classifier, even gradient boosted trees.
        - Two step correction:
            The original Boruta code corrects for multiple testing in an overly
            conservative way. In this implementation, the Benjamini Hochberg FDR is
            used to correct in each iteration across active features. This means
            only those features are included in the correction which are still in
            the selection process. Following this, each that passed goes through a
            regular Bonferroni correction to check for the repeated testing over
            the iterations.
        - Percentile:
            Instead of using the max values of the shadow features the user can
            specify which percentile to use. This gives a finer control over this
            crucial parameter. For more info, please read about the perc parameter.
        - Automatic tree number:
            Setting the n_estimator to 'auto' will calculate the number of trees
            in each itartion based on the number of features under investigation.
            This way more trees are used when the training data has many feautres
            and less when most of the features have been rejected.
        - Ranking of features:
            After fitting BorutaPy it provides the user with ranking of features.
            Confirmed ones are 1, Tentatives are 2, and the rejected are ranked
            starting from 3, based on their feautre importance history through
            the iterations.
        We highly recommend using pruned trees with a depth between 3-7.
        For more, see the docs of these functions, and the examples below.
        Original code and method by: Miron B Kursa, https://m2.icm.edu.pl/boruta/
        Boruta is an all relevant feature selection method, while most other are
        minimal optimal; this means it tries to find all features carrying
        information usable for prediction, rather than finding a possibly compact
        subset of features on which some classifier has a minimal error.
        Why bother with all relevant feature selection?
        When you try to understand the phenomenon that made your data, you should
        care about all factors that contribute to it, not just the bluntest signs
        of it in context of your methodology (yes, minimal optimal set of features
        by definition depends on your classifier choice).
        Parameters
        ----------
        estimator : object
            A supervised learning estimator, with a 'fit' method that returns the
            feature_importances_ attribute. Important features must correspond to
            high absolute values in the feature_importances_.
        n_estimators : int or string, default = 1000
            If int sets the number of estimators in the chosen ensemble method.
            If 'auto' this is determined automatically based on the size of the
            dataset. The other parameters of the used estimators need to be set
            with initialisation.
        perc : int | 'auto', default = 'auto'
            Instead of the max we use the percentile defined by the user, to pick
            our threshold for comparison between shadow and real features. The max
            tend to be too stringent. This provides a finer control over this. The
            lower perc is the more false positives will be picked as relevant but
            also the less relevant features will be left out. The usual trade-off.
            The default is essentially the vanilla Boruta corresponding to the max.
        alpha : float, default = 0.05
            Level at which the corrected p-values will get rejected in both
            correction steps.
        two_step : Boolean, default = True
            If you want to use the original implementation of Boruta with Bonferroni
            correction only set this to False.
        max_iter : int, default = 100
            The number of maximum iterations to perform.
        random_state : int, RandomState instance or None; default=None
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by `np.random`.
        verbose : int, default=1
            Controls verbosity of output:
            - 0: no output
            - 1: displays iteration number | if tqdm package exists, progress bar
            - 2: which features have been selected already
        max_shuf : int, default=10000
            How many times to calculate the Pearson coefficient when `perc` is 'auto'.
        n_jobs : int, default=None
            The number of jobs to run in parallel for both `fit` and `predict`.
            If -1, then the number of jobs is set to the number of cores.


        Attributes
        ----------
        n_features_ : int
            The number of selected features.
        support_ : array of shape [n_features]
            The mask of selected features - only confirmed ones are True.
        support_weak_ : array of shape [n_features]
            The mask of selected tentative features, which haven't gained enough
            support during the max_iter number of iterations..
        ranking_ : array of shape [n_features]
            The feature ranking, such that ``ranking_[i]`` corresponds to the
            ranking position of the i-th feature. Selected (i.e., estimated
            best) features are assigned rank 1 and tentative features are assigned
            rank 2.
        importance_history_ : array-like, shape [n_features, n_iters]
            The calculated importance values for each feature across all iterations.
        Examples
        --------

        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
        from boruta import BorutaPy

        # load X and y
        X = pd.read_csv('examples/test_X.csv', index_col=0).values
        y = pd.read_csv('examples/test_y.csv', header=None, index_col=0).values
        y = y.ravel()

        # define random forest classifier, with utilising all cores and
        # sampling in proportion to y labels
        rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5)

        # define Boruta feature selection method
        feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)

        # find all relevant features - 5 features should be selected
        feat_selector.fit(X, y)

        # check selected features - first 5 features are selected
        feat_selector.support_

        # check ranking of features
        feat_selector.ranking_

        # call transform() on X to filter it down to selected features
        X_filtered = feat_selector.transform(X)
        References
        ----------
        [1] Kursa M., Rudnicki W., "Feature Selection with the Boruta Package"
            Journal of Statistical Software, Vol. 36, Issue 11, Sep 2010
        """
        self.max_shuf = max_shuf
        self.n_jobs = n_jobs
        super().__init__(estimator=estimator, n_estimators=n_estimators, perc=perc, alpha=alpha,
                 two_step=two_step, max_iter=max_iter, random_state=random_state, verbose=verbose)
        self.random_state = check_random_state(self.random_state)
        if verbose > 0:
            try:
                import tqdm
            except ImportError as e:
                mess = '{}\nIf exists, a progress bar can be displayed.'.format(e)
                warnings.warn(mess)
                self._flag_tqdm = False
            else:
                self._flag_tqdm = True
        else:
            self._flag_tqdm = False

    def fit(self, X, y):
        """
        This docstring uses the same one as scikit-learn-contrib/boruta_py, which is a class inheritor under the BSD 3 clause license.
        https://github.com/scikit-learn-contrib/boruta_py/blob/master/boruta/boruta_py.py

        Fits the Boruta feature selection with the provided estimator.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values.
        """
        X, y = check_X_y(X, y)
        if self.perc == 'auto':
            self.perc = self._calc_auto_perc(X, y)
        if self._flag_tqdm and self.verbose == 1:
            if is_notebook():
                from tqdm.notebook import tqdm
            else:
                from tqdm import tqdm
            self.pbar = tqdm(total=self.max_iter, desc='BorutaPy')
        return self._fit(X, y)

    def get_support(self, weak=False)->np.ndarray:
        """get support

        Parameters
        ----------
        weak : bool, optional
            If set to true, the tentative features are also used to reduce X., by default False

        Returns
        -------
        support : array
        """
        check_is_fitted(self, 'support_')
        if weak:
            return self.support_weak_
        else:
            return self.support_


    def _calc_auto_perc(self, X, y)->float:
        """
        This docstring is based on scikit-learn-contrib/boruta_py, which is a class inheritor under the BSD 3 clause license.
        https://github.com/scikit-learn-contrib/boruta_py/blob/master/boruta/boruta_py.py

        calculate `perc` if 'auto'

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values.

        Returns
        -------
        float
            calculated `perc`
        """
        if self.verbose > 0 and self._flag_tqdm:
            if is_notebook():
                from tqdm.notebook import trange
            else:
                from tqdm import trange
            _range = trange(self.max_shuf, desc = 'Calc r_ccmax')
        else:
            _range = range(self.max_shuf)

        # ランダムに並べ替えてどれくらい相関がでてしまうのかを調べ，自動で決める．
        parallel = Parallel(n_jobs=self.n_jobs, verbose=0)
        def _get_pearsonrs()->list:
            X_shuffled = shuffle(X, random_state = self.random_state)
            return [pearsonr(X_shuffled[:, i], y)[0] for i in range(X_shuffled.shape[1])]
        self.pears_ = parallel(delayed(_get_pearsonrs)() for _ in _range)

        # self.pears_ = []    # 相関係数を足していくリスト
        # for _ in _range:
        #     self.pears_.extend(_get_pearsonrs())

        self.r_ccmax_ = np.nanmax(self.pears_)
        perc = 100 * (1 - self.r_ccmax_)
        if self.verbose > 0:
            sys.stdout.write("Assgigned perc = {:.1f}\n".format(perc))
        return perc

    def _print_results(self, dec_reg, _iter, flag):
        n_iter = str(_iter) + ' / ' + str(self.max_iter)
        n_confirmed = np.where(dec_reg == 1)[0].shape[0]
        n_rejected = np.where(dec_reg == -1)[0].shape[0]
        cols = ['Iteration: ', 'Confirmed: ', 'Tentative: ', 'Rejected: ']

        # still in feature selection
        if flag == 0:
            n_tentative = np.where(dec_reg == 0)[0].shape[0]
            content = map(str, [n_iter, n_confirmed, n_tentative, n_rejected])
            if self.verbose == 1:
                if self._flag_tqdm:
                    self.pbar.update()
                    output = None
                else:
                    output = cols[0] + n_iter
            elif self.verbose > 1:
                output = '\n'.join([x[0] + '\t' + x[1] for x in zip(cols, content)])

        # Boruta finished running and tentatives have been filtered
        else:
            n_tentative = np.sum(self.support_weak_)
            n_rejected = np.sum(~(self.support_|self.support_weak_))
            content = map(str, [n_iter, n_confirmed, n_tentative, n_rejected])
            result = '\n'.join([x[0] + '\t' + x[1] for x in zip(cols, content)])
            output = "\n\nBorutaPy finished running.\n\n" + result
            if self.verbose == 1 and self._flag_tqdm:
                self.pbar.update(self.max_iter - _iter + 1)
                self.pbar.close()

        if not(flag == 0 and self.verbose == 1 and self._flag_tqdm):
            sys.stdout.write(output + '\n')

def calc_vip(estimator:PLSRegression)->np.ndarray:
    """Calculate VIP (The Variable Importance in Projection)

    References
    - Mukherjee, R., Sengupta, D., & Sikdar, S. K. (2015). Selection of Sustainable Processes Using Sustainability Footprint Method: A Case Study of Methanol Production from Carbon Dioxide. In Computer Aided Chemical Engineering (Vol. 36, pp. 311-329). Elsevier. DOI: [10.1016/B978-0-444-63472-6.00012-4](https://doi.org/10.1016/B978-0-444-63472-6.00012-4)
    - [Variable Importance in Projection](https://www.sciencedirect.com/topics/engineering/variable-importance-in-projection)

    Parameters
    ----------
    estimator : PLSRegression
        fitted pls regression model

    Returns
    -------
    np.ndarray
        VIP of each variable (n_features,)
    """
    # score: 潜在変数、主成分
    t = estimator.x_scores_     # (n_samples, n_components)
    w = estimator.x_weights_    # (n_features, n_components)
    # loading: 因子負荷量
    q = estimator.y_loadings_   # (n_targets, n_components)
    p, h = w.shape
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    return np.sqrt(p * (np.square(w / np.linalg.norm(w, axis=0)) @ s).ravel() / np.sum(s))


if __name__ == '__main__':
    import os
    from urllib import request
    from pdb import set_trace
    import pandas as pd
    from sklearn.feature_selection import VarianceThreshold

    # 配布ページ: https://datachemeng.com/pythonassignment/
    url = 'https://datachemeng.com/wp-content/uploads/2017/07/logSdataset1290.csv'

    dirpath_cache = os.path.abspath('./_cache')
    if not os.path.isdir(dirpath_cache):
        os.mkdir(dirpath_cache)

    fpath_csv_cache = os.path.join(dirpath_cache, os.path.basename(url))
    if not os.path.isfile(fpath_csv_cache):
        with request.urlopen(url) as response:
            content = response.read().decode('utf-8-sig')
        with open(fpath_csv_cache, 'w', encoding='utf-8-sig') as f:
            f.write(content)

    df_data = pd.read_csv(fpath_csv_cache, index_col=0)
    df_data.head()
