# Changelog

Notable changes to yikit. Git tags use `vX.Y.Z-rc.N`, and PyPI shows the same versions as `X.Y.ZrcN`.

## [0.4.0-rc.1] - 2026-10-14

This release candidate makes these changes:

- Hyperparameter tuning is rebuilt so that it respects the parameters you pass.
- `EnsembleRegressor` is rebuilt on scikit-learn's ensembles.
- `GBDTRegressor` works with LightGBM 4.
- Every module now has docstrings and tests.

Many of these changes break existing code, and they are marked **Breaking**. If you upgrade from 0.3.x, read [0.4.0-rc.0](#040-rc0---2026-01-28) as well.

### Supported environments

- Python 3.8 is supported again (Python 3.8–3.14). ngboost is still not installed on Python 3.14.
- The `optional` extra now requires `Boruta>=0.4.3` and `optuna>=3.0`, and includes `optuna-integration`. The `test` extra no longer installs `optuna-integration`.

### `yikit.models`: hyperparameter tuning

- `Objective` no longer overwrites the parameters of the estimator you pass.
  - It does not inject `n_jobs`, and it keeps an explicit `random_state`.
  - A `random_state` left as None gets one integer drawn from `random_state`.
  - The implicit fixed values are gone: `n_jobs=-1`, a `RandomState` for `random_state`, `gamma="auto"` and `kernel="rbf"` for SVR, and `objective="regression"` for LightGBM.
- `Objective` and `ParamDistributions` now share one table of search spaces.
  - They find the predicting model inside a `Pipeline` (its last step) or a `TransformedTargetRegressor` (its `regressor`), and prefix the names, such as `regressor__svr__C`.
  - New search spaces: `PLSRegression`, `LinearSVR`, `Ridge`, `Lasso` and `ElasticNet`.
- New `RecommendedParams(estimator)`: a dict of the values that yikit recommends to fix. It is opt-in. Pass it yourself as `fixed_params` of `Objective`, or to `set_params`. For now it only holds `gamma="auto"` for SVR.
- `Objective.get_best_estimator(study)` returns a copy of the estimator you passed, with the best parameters set. It no longer rebuilds the estimator from its type, so it also keeps the `Base` of `NGBRegressor`.
- Other `Objective` changes:
  - A fixed name now also covers the names below it.
  - The `random_state` rule also applies to fixed estimators.
  - A trial whose fit fails is recorded as FAIL instead of stopping the study.
  - A non-empty `custom_params` now takes priority even for models that have a search space in the table.
  - `custom_params` and `fixed_params` now default to None.
- `ParamDistributions` changes:
  - `n_features` is keyword-only.
  - A callable `custom_params` is evaluated with `FixedTrial({})`, and its exceptions propagate.
- The same seed gives different search results than in 0.4.0-rc.0.
- The choices of `Base__criterion` for `NGBRegressor` depend on the scikit-learn version.
- **Breaking**: `LinearModelRegressor` and `SupportVectorRegressor` are removed.
  - Instead of `LinearModelRegressor`, use `Ridge`, `Lasso` or `ElasticNet`.
  - Instead of `SupportVectorRegressor`, use `TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(), SVR()), transformer=StandardScaler())`. To keep its `gamma`, pass `RecommendedParams` (or `fixed_params={"regressor__svr__gamma": "auto"}`).
- **Breaking**: some attributes and arguments are removed or changed.
  - `Objective` no longer has the attributes `model`, `fixed_params_` and `rng`.
  - `ParamDistributions` no longer has the attributes `distributions` and `rng`, and no longer takes the arguments `fixed_params` and `random_state`.
  - The `__repr__` of `ParamDistributions` changed.

### `yikit.models`: `EnsembleRegressor`

- `EnsembleRegressor` now builds a scikit-learn ensemble and delegates `fit` and `predict` to it.
  - `"average"` builds a `VotingRegressor`; `"stacking"` and `"blending"` build a `StackingRegressor`.
  - Before, it failed to fit on scikit-learn >= 1.4 and on NumPy 1.24–1.26.
- With `opt=True`, each model is tuned with `OptunaSearchCV` over `ParamDistributions`, on the training data of each split only.
  - `n_trials` sets the number of trials.
  - Models keep the parameters you pass. A `random_state` left unspecified gets one drawn integer, as in `Objective`.
- New fitted attributes: `estimator_`, `estimators_`, `named_estimators_`, `final_estimator_`, `weights_`, `model_random_state_`, `n_features_in_` and `feature_names_in_`.
- **Breaking**: the `boruta` argument and the `results_` attribute are removed. To select features, pass `Pipeline([("boruta", BorutaPy(...)), ("model", model)])` as an estimator.
- **Breaking**: predictions change.
  - Predictions come from the models refitted on all the training data, instead of the average of the models of each fold.
  - The weights of `"blending"` are non-negative least squares and do not sum to 1.
  - `n_jobs` defaults to None.
- A plain `NGBRegressor` is not accepted by scikit-learn >= 1.6 ensembles. Wrap it in `TransformedTargetRegressor(regressor=NGBRegressor(...))`.

### `yikit.models`: `GBDTRegressor`

- `GBDTRegressor` works with LightGBM 3 and 4.
  - Early stopping uses the `lightgbm.early_stopping` callback.
  - `silent` is mapped to LightGBM's `verbosity`.
  - New `best_iteration_`.
- It now trains on the training split only. Before, the 20% held out for early stopping was also used for training, so results change.
- **Breaking**: extra keyword arguments (`**kwargs`) are no longer accepted. For other LightGBM parameters, use `lightgbm.LGBMRegressor`.

### `yikit.metrics`

- **Breaking**: `root_mean_squared_error` is removed. Use `sklearn.metrics.root_mean_squared_error` (scikit-learn >= 1.4), or `np.sqrt(mean_squared_error(...))` on older versions.
- `log_likelihood` returns the mean log-likelihood whatever `Score` the model was trained with. It returned the negated mean CRPS for `Score=CRPScore`. It raises ValueError for a distribution without a `LogScore` implementation.

### `yikit.feature_selection`

- `FilterSelector`:
  - It removes features by the absolute correlation (`|r| > r` and `p < alpha`), so strong negative correlations are removed too.
  - It no longer hits the recursion limit when many features are removed.
  - Constant columns are treated as uncorrelated, without warnings. Their pairs in `corr_` are now r=0.0 and p=1.0 instead of NaN.
  - Negative pairs with `|r| > r` now have their real p-value in `corr_`.
  - New `n_features_in_` and `feature_names_in_`, and `get_feature_names_out` works. So `transform` is checked by scikit-learn:
    - A different number of features raises ValueError (it was IndexError).
    - A DataFrame with other column names raises ValueError.
    - An ndarray after fitting on a DataFrame gives a warning.
- `BorutaPy`:
  - `random_state` is stored as passed. `get_params()["random_state"]` no longer returns a `RandomState`, so the same instance and its clones give the same results.
  - For `perc="auto"`, `perc_` no longer depends on `n_jobs`. Its value (and which borderline features are selected) can differ from 0.4.0-rc.0 for the same seed.
  - `get_support(weak=True)` returns the confirmed and the tentative features, as `transform(X, weak=True)` keeps them. It returned the tentative features only.
  - `set_params(verbose=...)` takes effect at the next fit.
  - The progress bar is closed when `fit` fails.
  - A selector fitted with `verbose>=1` can be pickled.
  - A fit with a numeric `perc` removes `r_ccmax_` and `pears_` left by a previous `perc="auto"` fit.

### `yikit.visualize`

- `yyplot`:
  - Labels are shown as written: they are placed outside mathtext, so a metric line reads `R² (train)` instead of a subscript.
  - An empty label is treated like None.
  - With `ax`, it no longer applies `tight_layout` to the figure.
  - An unknown metric raises ValueError before anything is drawn.
- `SummarizePI.get_figure` and `get_learning_curve_gb` now restore the font settings after the call, like the other `get_*` functions. `set_font` still changes the global settings.
- `SummarizePI`:
  - `get_data` returns the plotted values (`importances_mean` and `importances_std` of the normalized importances, sorted by importance).
  - `get_figure` draws the standard deviation as error bars; they were NaN before.
  - Numeric feature names are sorted by importance on seaborn 0.13.
- `get_learning_curve_optuna`:
  - It plots complete trials only.
  - It raises ValueError for multi-objective studies, studies without a direction and studies without a complete trial.
- `get_dist_figure`:
  - It works with one or two samples.
  - `titles` defaults to None.
  - With `return_axis=True`, the axes are always 2-D.
- `get_learning_curve_gb`:
  - It accepts any eval set names, such as LightGBM's `training` and `valid_0`; unknown names use matplotlib's color cycle.
  - It raises ValueError for models without an evaluation record, and TypeError for other models.
  - It accepts every `LGBMModel` and `NGBoost` model.

### `yikit.helpers` and `yikit.logging`

- `tqdm_joblib`:
  - It advances with any `n_jobs`, including 1 and None.
  - It counts only the outermost `Parallel` started on the thread that entered it. Calls that return a generator (`return_as="generator"`) are no longer counted.
  - It works when nested and from several threads.
  - Entering an active instance again, or exiting before entering, raises RuntimeError.
- `catch_default_handler` restores the state from before entering. The same instance can be nested. Exiting before entering raises RuntimeError.
- A missing default handler now raises RuntimeError instead of AssertionError.
- The logger keeps its behavior. The docstrings describe how to silence or replace the default handler.

### Documentation

- Every public name of `yikit.helpers`, `logging`, `metrics`, `feature_selection` and `visualize` has an English numpy-style docstring and tests.
- The examples are updated for the new API.

## [0.4.0-rc.0] - 2026-01-28

- New `ParamDistributions` (search spaces for `OptunaSearchCV`).
- **Breaking**: `NNRegressor` is removed. Use `sklearn.neural_network.MLPRegressor`.
- `get_dist_figure` is exported only when ngboost is installed, and `get_learning_curve_gb` only when ngboost or LightGBM is installed.
- Python 3.13 and 3.14 are supported, and Python 3.7 and 3.8 are dropped (3.8 is restored in 0.4.0-rc.1). ngboost is not installed on Python 3.14.

[0.4.0-rc.1]: https://github.com/yu9824/yikit/compare/v0.4.0-rc.0...v0.4.0-rc.1
[0.4.0-rc.0]: https://github.com/yu9824/yikit/compare/v0.3.19...v0.4.0-rc.0
