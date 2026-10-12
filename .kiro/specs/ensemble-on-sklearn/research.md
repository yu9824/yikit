# Research & Design Decisions

## Summary
- **Feature**: `ensemble-on-sklearn`
- **Discovery Scope**: Extension（既存の `EnsembleRegressor` を scikit-learn の部品で作り直す。`BorutaPy` の最小限の修正を含む）
- **Key Findings**:
  - scikit-learn 1.9 では `is_regressor(OptunaSearchCV(...))` が False になり、VotingRegressor と StackingRegressor が "The estimator OptunaSearchCV should be a regressor." で拒む。1.6 から回帰モデルの判定が `__sklearn_tags__` に移ったのに、optuna-integration の `OptunaSearchCV` が tags を定義していないため。brief の「`OptunaSearchCV(model, ...)` で包んで渡す」はそのままでは成り立たない
  - yikit の `BorutaPy` は `perc="auto"` を `fit` でしか解決しない。boruta 0.4.3 の `fit_transform` は `self._fit` を直接呼ぶので、Pipeline の中では `perc` が文字列のまま `_check_params` に渡り TypeError になる。さらに `fit` が引数 `perc` を計算した値に書き換える（clone すると "auto" が失われる）
  - optuna-tuning で入れた `tests/test_optuna.py` の `test_ensemble_builds_the_tuned_estimator_with_get_best_estimator` は、作り直した EnsembleRegressor が `Objective` を使わなくなるので役目を終える

## Research Log

### OptunaSearchCV を sklearn のアンサンブルに入れる
- **Context**: 要件 2.1（各モデルを `OptunaSearchCV` で調整してからまとめる）
- **Sources Consulted**: scikit-learn 1.9.1 と optuna-integration（yikit-dev 環境）での実行、`OptunaSearchCV.__init__` の引数
- **Findings**:
  - `is_regressor(OptunaSearchCV(Ridge(), ...))` は False。StackingRegressor と VotingRegressor の `fit` が ValueError を出す
  - scikit-learn 1.6 未満の `is_regressor` は `_estimator_type` 属性を見る。1.6 以上は `__sklearn_tags__().estimator_type` を見る
  - `OptunaSearchCV` の引数は `estimator, param_distributions, cv, enable_pruning, error_score, max_iter, n_jobs, n_trials, random_state, refit, return_train_score, scoring, study, subsample, timeout, verbose, callbacks, catch`。`random_state` は TPESampler の種になる。`study=None` なら `fit` のたびに新しい study を作るので、clone した写しどうしで study を共有しない
  - `OptunaSearchCV` は optuna のログの設定を変えない。optuna は既定で試行ごとに INFO のログを出す
- **Implications**: `OptunaSearchCV` を継承し、回帰モデルであることだけを示す小さなクラスを yikit に置く。`_estimator_type = "regressor"` と、`__sklearn_tags__` で `estimator_type = "regressor"` を返す（1.6 以上でだけ呼ばれる）。`__init__` は上書きしないので `get_params`・`clone` はそのまま。pickle のためにモジュールの直下に定義する

### BorutaPy の perc="auto"
- **Context**: 要件 4.2、4.3
- **Sources Consulted**: boruta 0.4.3 の `boruta/boruta_py.py`（`fit` L209、`fit_transform` L249、`_fit` L286、`np.percentile(..., self.perc)` L350、`_check_params` L603-609）、yikit の `feature_selection/_wrapper.py`（`fit` L216-236、`_print_results` の `_pbar`）
- **Findings**:
  - boruta の `fit` と `fit_transform` は、どちらも `self._fit(X, y)` を呼ぶ。`self.perc` は `_check_params` と、影の特徴量の重要度の百分位で使われる
  - yikit の `fit` は、`perc` の自動決定と tqdm の進捗バー（`_pbar`）の用意をしてから `_fit` を呼ぶ。`fit_transform` の経路ではどちらも行われない（`_print_results` が `_pbar` を使うので、tqdm があると別の AttributeError にもなりうる）
- **Implications**: 自動決定と進捗バーの用意を、`fit` から `_fit` の上書きへ移す。boruta が読む `self.perc` には、`_fit` の間だけ解決した値を入れ、終わったら元の値（"auto" など）に戻す（try/finally）。解決した値は `perc_` に残す

### 既存の利用者
- **Findings**: `examples/simulate.py`（`wip/0.4.0-draft`）と `examples/demo_ensemble.py` は `fit`・`predict` と `results_.estimators` だけを使う。src の中で EnsembleRegressor の `results_` や `weights_` を使うものはない
- **Implications**: `results_` の削除は examples の書き直し（release-0.4.0）で吸収できる

## Architecture Pattern Evaluation

| Option | Description | Strengths | Risks / Limitations | Notes |
|--------|-------------|-----------|---------------------|-------|
| 薄いクラスが Voting/Stacking を組み立てる | `fit` で引数から sklearn のアンサンブルを作り、学習を委ねる | 既存の名前と使い方を保つ。sklearn の決まり（clone、Pipeline）に乗る | アンサンブルの属性を写して公開する手間 | 採用（利用者の回答） |
| 組み立てる関数 | `make_ensemble(...)` が Voting/Stacking を返す | 最小のコード | EnsembleRegressor の名前がなくなる | 不採用 |
| 全データで一度だけ調整して Stacking に入れる | 調整済みのモデルを固定して渡す | 学習が軽い | まとめ役の学習に使う予測の調整に検証データが混ざる（要件 2.4 に反する） | 不採用 |

## Design Decisions

### Decision: OptunaSearchCV の回帰モデル版を yikit に置く
- **Context**: scikit-learn 1.6 以上で `OptunaSearchCV` がアンサンブルに拒まれる
- **Alternatives Considered**:
  1. `RegressorMixin` を混ぜて継承する — `score` が R² に置き換わり、`OptunaSearchCV.score`（scoring による）と意味が変わる
  2. 回帰モデルの印（`_estimator_type` と tags）だけを足す
- **Selected Approach**: 2。`src/yikit/models/_search_cv.py` に `OptunaSearchRegressor(OptunaSearchCV)` を置く（公開しない）。`verbose=0` のときは自身の `fit` の間だけ optuna のログを WARNING にして、終わったら戻す
- **Rationale**: 振る舞いは `OptunaSearchCV` のまま。`fit` は並列のワーカーの中でも動くので、ログの抑制がワーカーでも効く
- **Trade-offs**: optuna-integration が tags を定義したら不要になる。モジュールは optuna-integration がないと import できないので、`EnsembleRegressor.fit` の中で `opt=True` のときだけ import する

### Decision: random_state の規則は Objective と同じ部品を使う
- **Context**: 要件 3.2
- **Selected Approach**: `fit` で `check_random_state(random_state)` から整数を1つ引き、各モデルの `find_unspecified_random_states` の名前に `apply_params` で入れる（optuna-tuning の `_params.py`）。続けて、モデルごとに `OptunaSearchCV` の `random_state`（TPESampler の種）を引く
- **Rationale**: 規則と、ngboost を含む引数の入れ方を1か所に保つ

### Decision: 入力は変換せずに中のアンサンブルへ渡す
- **Context**: 要件 5.4（`feature_names_in_`）、Pipeline の中で列名を使う前処理
- **Selected Approach**: EnsembleRegressor は X を `check_X_y` で配列に変えない。`ParamDistributions` に渡す特徴量の数だけを X の形から読み、検査と `n_features_in_`・`feature_names_in_` は中のアンサンブルの値を写す
- **Rationale**: 列名に頼る Pipeline（ColumnTransformer など）を壊さない

### Decision: n_jobs の既定は None
- **Context**: 要件 1.7。並列数は Ensemble、調整、各モデルの層ごとに掛け算になる
- **Selected Approach**: EnsembleRegressor の `n_jobs` の既定は None。`OptunaSearchCV` の試行は並列にしない（`n_jobs=1`、TPE の再現性のため）。各モデルの `n_jobs` は渡された値のまま

## Risks & Mitigations
- stacking と blending の調整は、分割ごとと全データで (cv+1) 回行われる（既定で1モデルあたり 3000 回の学習） — docstring に書き、`n_trials` と `cv` で調整してもらう
- optuna-integration が `OptunaSearchCV` に tags を足すと、派生クラスの印が不要になる — 害はないので残す
- `BorutaPy._fit` が `self.perc` を一時的に変える間に例外が出ても、try/finally で元に戻す
- 作り直しで EnsembleRegressor の予測は 0.4.0-rc.0 と変わる（各分割のモデルの平均をやめる） — CHANGELOG（release-0.4.0）に書く

## References
- scikit-learn: `VotingRegressor`、`StackingRegressor`、`is_regressor`、`__sklearn_tags__`（1.6 以降）
- optuna-integration: `OptunaSearchCV`
- boruta 0.4.3: `boruta/boruta_py.py`
