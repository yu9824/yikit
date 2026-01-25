"""
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
"""

from typing import List, Optional, Set, Tuple, Union

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import pearsonr
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin  # >= 0.23.2
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

from yikit.helpers import is_installed, tqdm_joblib
from yikit.logging import get_child_logger

logger = get_child_logger(__name__)


class FilterSelector(SelectorMixin, BaseEstimator):
    def __init__(
        self,
        r: float = 0.9,
        alpha: float = 0.05,
        verbose: Union[int, bool] = True,
        n_jobs: Optional[int] = None,
    ):
        """
        Filter method for feature selection using correlation coefficient and significance.

        Parameters
        ----------
        r : float, optional
            The correlation coefficient threshold. Features with a correlation higher than this value
            will be considered highly correlated and thus will be subject to removal. Default is 0.9.
        alpha : float, optional
            Significance level. If the p-value is lower than this threshold,
            the correlation coefficient is considered statistically significant.
            It is recommended not to change this without a specific reason. Default is 0.05.
        verbose : bool or int, optional
            Whether to show progress during batch processing. Default is True.
        n_jobs : int, optional
            Number of jobs to run in parallel. If None, 1 is used.
            If -1, all CPUs are used. Default is None.
        """
        self.r = r
        self.alpha = alpha
        self.verbose = verbose
        self.n_jobs = n_jobs

    def fit(self, X, y=None):
        """
        Fit the FilterSelector.

        This method identifies features in `X` that are highly correlated (above correlation threshold `r`
        and with significant p-value below `alpha`). Features that are highly correlated with others
        will be recursively removed, keeping only minimally redundant features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix. Can be a pandas DataFrame, 2D numpy array, or 2D list.
        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        # Convert and validate X as np.ndarray
        X: np.ndarray = check_array(X)

        # Store the number of features for later use
        n_features = X.shape[1]

        # Prepare a tensor to store correlation coefficients and p-values.
        # [i, j, 0] is corr coef, [i, j, 1] is p-value
        self.corr_ = np.empty((n_features, n_features, 2), dtype=float)

        # Prepare a list of sets. Each i-th set holds features highly correlated with i-th feature.
        pairs_highly_correlated: List[Set[int]] = [
            set() for _ in range(n_features)
        ]

        # Decide on progress display based on availability of tqdm and verbosity
        use_tqdm = is_installed("tqdm") and self.verbose > 0
        use_logging = not is_installed("tqdm") and self.verbose > 0

        # Compute correlation matrix in one go (p-values are computed later as needed)
        corr_matrix = np.corrcoef(X.T)

        # Fill corr_ array with coefficients and set diagonal
        for i in range(n_features):
            self.corr_[i, i, 0] = 1.0
            self.corr_[i, i, 1] = 0.0
            for j in range(i + 1, n_features):
                corr_value = corr_matrix[i, j]
                self.corr_[i, j, 0] = corr_value
                self.corr_[j, i, 0] = corr_value

        # Identify pairs that need p-value computation (correlation above threshold)
        pairs_to_compute: Set[Tuple[int, int]] = set()
        for i in range(n_features):
            for j in range(i + 1, n_features):
                if corr_matrix[i, j] > self.r:
                    pairs_to_compute.add((i, j))

        # Compute p-values in parallel for those pairs that need it
        if pairs_to_compute:

            def _compute_pvalue(
                pair_idx: Tuple[int, int],
            ) -> Tuple[int, int, float]:
                """
                Compute the p-value for the Pearson correlation between two features given by the indices.

                Parameters
                ----------
                pair_idx : tuple of int
                    Indices (i, j) of the feature pair.

                Returns
                -------
                tuple
                    (i, j, p-value)
                """
                i, j = pair_idx
                _, p_value = pearsonr(X[:, i], X[:, j])
                return (i, j, p_value)

            # Compute p-values using parallel processing, with optional progress bar if available
            if use_tqdm:
                with tqdm_joblib(
                    total=len(pairs_to_compute),
                    desc="computing p-values",
                    silent=False,
                ):
                    parallel = Parallel(n_jobs=self.n_jobs, verbose=0)
                    pvalue_results = parallel(
                        delayed(_compute_pvalue)(pair_idx)
                        for pair_idx in pairs_to_compute
                    )
            else:
                parallel = Parallel(n_jobs=self.n_jobs, verbose=0)
                pvalue_results = parallel(
                    delayed(_compute_pvalue)(pair_idx)
                    for pair_idx in pairs_to_compute
                )

            # Store p-value results in the corr_ tensor, and update highly correlated feature sets
            for i, j, p_value in pvalue_results:
                self.corr_[i, j, 1] = p_value
                self.corr_[j, i, 1] = p_value
                if (
                    p_value < self.alpha
                ):  # If correlation is high and p-value is significant
                    pairs_highly_correlated[i].add(j)
                    pairs_highly_correlated[j].add(i)
        else:
            if use_logging:
                logger.info("No pairs with correlation above threshold.")

        # For pairs below correlation threshold, assign default p-value of 1.0
        for i in range(n_features):
            for j in range(i + 1, n_features):
                if corr_matrix[i, j] <= self.r:
                    self.corr_[i, j, 1] = 1.0
                    self.corr_[j, i, 1] = 1.0

        if use_logging:
            logger.info(
                f"Finished! Processed {n_features} features, computed {len(pairs_to_compute)} p-values."
            )

        def _delete_recursive(
            pairs_highly_correlated: List[Set[int]],
            boolean: np.ndarray = np.ones(n_features, dtype=bool),
        ) -> np.ndarray:
            """
            Recursively remove features with the highest number of highly correlated pairs.

            At each step, find the feature involved in the largest number of highly correlated pairs,
            remove it (by setting its mask to False), and repeat until there are no more such pairs.

            Parameters
            ----------
            pairs_highly_correlated : List[Set]
                List of sets, where each set contains the indices of features highly correlated with the i-th feature.
            boolean : np.ndarray, optional
                Boolean mask for features (True if the feature is kept), by default all True.

            Returns
            -------
            np.ndarray
                Boolean mask of selected (kept) features.
            """
            order_pair = np.array(
                [len(s) for s in pairs_highly_correlated], dtype=int
            )

            # If there are no more high correlation connections, stop recursion
            if np.sum(order_pair) == 0:
                return boolean

            # Identify the feature with the largest number of highly correlated connections
            i_max = np.argmax(order_pair)

            # Remove connections to this feature from all other sets
            pairs_highly_correlated = [
                p.difference({i_max}) for p in pairs_highly_correlated
            ]

            # Mark the feature as removed and set its mask to False
            pairs_highly_correlated[i_max] = set()
            boolean[i_max] = False
            return _delete_recursive(pairs_highly_correlated, boolean)

        self.support_ = _delete_recursive(pairs_highly_correlated)

        return self

    def _get_support_mask(self) -> np.ndarray:
        """
        Get the boolean mask indicating which features are selected.

        Returns
        -------
        np.ndarray
            Boolean mask array of shape (n_features,), where True means the feature is selected.
        """
        check_is_fitted(self, "support_")
        return self.support_
