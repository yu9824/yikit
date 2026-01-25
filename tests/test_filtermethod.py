import numpy as np
from sklearn.datasets import load_iris

from yikit.feature_selection import FilterSelector


def test_filter_method():
    X = load_iris().data

    selector = FilterSelector(verbose=0)
    selector.fit(X)
    assert np.all(selector.support_ == np.array([True, True, False, True]))
