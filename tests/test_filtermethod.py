from yikit.feature_selection import FilterSelector


def test_filter_method(X_regression, y_regression):
    X = X_regression

    selector = FilterSelector()
    selector.fit(X)
    assert selector.support_.sum() > 0
