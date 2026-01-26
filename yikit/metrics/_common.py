import importlib


def root_mean_squared_error(y_true, y_pred, **kwargs):
    """
    Compute the root mean squared error for a set of predictions.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True target values.
    y_pred : array-like of shape (n_samples,)
        Predicted target values.
    **kwargs
        Additional keyword arguments to pass to the root mean squared error function.

    Returns
    -------
    float
        The root mean squared error computed by the estimator.

    Notes
    -----
    This function is a wrapper around the root mean squared error function from the scikit-learn library.
    If the root mean squared error function is not available, the mean squared error function is used instead.
    """
    sklearn_metrics_module = importlib.import_module("sklearn.metrics")
    if hasattr(sklearn_metrics_module, "root_mean_squared_error"):
        return sklearn_metrics_module.root_mean_squared_error(
            y_true, y_pred, **kwargs
        )
    else:
        return sklearn_metrics_module.mean_squared_error(
            y_true, y_pred, squared=False, **kwargs
        )
