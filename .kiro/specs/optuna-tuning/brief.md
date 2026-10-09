# Brief: optuna-tuning

## Problem
- **licond**（yikit を使う側）は、モデルの並列数と乱数を自分で決めて `Objective` に渡したい。しかし今の `Objective` は次のように上書きするので、指定が効かない
  - LGBMRegressor・GBDTRegressor・RandomForestRegressor の固定値に `"n_jobs": -1` を入れ、探索の各試行と `get_best_params` の結果に入れる
  - LightGBM・ランダムフォレスト・LinearModelRegressor・MLPRegressor・NGBRegressor の固定値に `"random_state": self.rng`（RandomState のオブジェクト）を入れる。NGBRegressor では、Base の DecisionTreeRegressor にも同じオブジェクトを入れる。`get_best_params` の結果にもこのオブジェクトが入り、最後のモデルと入れ子のモデルが1つの RandomState を共有する
- **探索範囲が2か所にある**: `Objective.__call__`（`trial.suggest_*`）と `ParamDistributions._build_distributions`（optuna の分布）に同じ範囲を書いている。途中まで進めた変更（`wip/0.4.0-draft`）では LinearSVR・PLS を `Objective` にだけ足していて、ずれ始めている
- **sklearn の組み合わせを探索できない**: `Pipeline` や `TransformedTargetRegressor` に包んだモデルは探索範囲が見つからない。そのため、引数名を平らにするだけの包み（LinearModelRegressor、SupportVectorRegressor）が必要になっていた
- **固定値を暗黙に入れる**: SVR の `gamma`・`kernel` なども、利用者が明示したかどうかに関係なく上書きする。モデルの値からは、明示したのか既定値のままなのかを判定できない

## Current State
- `src/yikit/models/_optuna.py` に `ParamDistributions`（dict を継承、`OptunaSearchCV` 用）と `Objective` がある
- `Objective.__init__` で TPESampler を作る（`self.rng.randint(2**31 - 1)` で種を引く）
- `Objective.__call__` はモデルの型ごとに分岐し、`params_` と `self.fixed_params_` を組み立てて `clone(self.estimator).set_params(...)` し、`cross_validate(..., n_jobs=self.n_jobs)` の平均を返す
- `get_best_estimator` は `self.model(**best_params)`（型から作り直す）なので、探索しない引数が既定値に戻る。NGBRegressor は `get_best_params` に専用の処理がある
- LinearModelRegressor の探索範囲は `max_iter` を浮動小数で出すので、scikit-learn 1.2 以上の Lasso では失敗する
- NGBRegressor の探索範囲 `Base__criterion` に含まれる `"friedman_mse"` は、scikit-learn 1.11 で削除される予定（今は FutureWarning）
- `EnsembleRegressor` が `objective.model` と `objective.fixed_params_` を使っている
- テストは `test_optuna.py` の2件（RandomForest で study と OptunaSearchCV の最良値を確かめる）だけ

## Desired Outcome
1. **探索範囲の共通ファイル**: モデルの型ごとの探索範囲（optuna の分布）を1つのモジュールで持つ。`Objective` も `ParamDistributions` もそこから作る。両者の範囲が同じであることをテストで確かめる
2. **入れ子の解決**: `Pipeline` は最後の段、`TransformedTargetRegressor` は `regressor` を再帰的にたどって探索するモデルを見つけ、引数名に前置き（例 `regressor__svr__C`）を付ける。`Objective`（`set_params`）でも `OptunaSearchCV` でもそのまま使える。Boruta の中のモデルなど、予測を担わない estimator は探索しない
3. **探索範囲の追加**: PLSRegression（`n_components` は 1〜min(10, 特徴量の数)）、LinearSVR、Ridge・Lasso（`alpha`）、ElasticNet（`alpha`、`l1_ratio` は 0.1〜1.0）。`max_iter`・`tol`・`fit_intercept` は探索しない。`ParamDistributions` に `n_features` を足し、データで決まる範囲に使う
4. **RecommendedParams**: yikit の推奨する固定値を返す dict を継承したクラス。利用者が `Objective(..., fixed_params=RecommendedParams(est))` や `est.set_params(**RecommendedParams(est))` で明示的に使う。入れ子の解決を共有する。最初は少なく（例: SVR の `gamma="auto"`）、後から足す
5. **`Objective` の引数の扱い**
   - 固定値を暗黙に入れない。`n_jobs` は入れず、渡したモデル（`clone(self.estimator)`）の値を使う
   - `random_state`: モデルで明示されている値（None 以外）はそのまま使う。None のときだけ、Objective の乱数から引いた整数を入れる。整数は `__init__` で、TPESampler の種を引いた後に1回だけ引き、全試行と `get_best_params` で同じ値を使う（試行どうしを同じ乱数で比べられるように）。入れ子のモデル（NGBRegressor の Base など）も同じ規則にする。RandomState のオブジェクトを、最後のモデルと入れ子のモデルで共有させない
   - 利用者が `fixed_params` で渡した値は最優先で使う
   - `get_best_params` は「探索した引数」と「固定値（上の規則の `random_state` と、利用者の `fixed_params`）」だけを返す。`get_best_estimator` は `clone(self.estimator).set_params(**best_params)` で作り、探索しない引数は渡したモデルの値のまま残す
6. **包みの削除**: LinearModelRegressor（`_linear.py`）と SupportVectorRegressor（`_svm.py`）を削除する。代わりの書き方（`TransformedTargetRegressor` + Pipeline + SVR、Ridge・Lasso・ElasticNet）を docstring と CHANGELOG で案内する
7. **テスト**（licond の要件を含む）
   - `n_jobs=2` の LGBMRegressor と RandomForestRegressor: 試行のモデル、`get_best_params`、`get_best_estimator` のどれでも `n_jobs` が 2 のまま
   - `random_state=7` のモデル: 試行と最後のモデルで 7 のまま
   - `random_state=None` のモデル: 試行と `get_best_params` の `random_state` が整数で、`Objective` の `random_state` が同じなら同じ値になり、study の結果も同じになる
   - NGBRegressor（Base の `random_state` が None の場合と明示した場合）: Base の `random_state` が上の規則に従い、最後のモデルと Base が同じ RandomState のオブジェクトを持たない
   - `fixed_params={"n_jobs": 3, "random_state": 5}` を渡すと、その値が使われる
   - 探索範囲の共通ファイル: `Objective` と `ParamDistributions` の範囲が一致する。入れ子（Pipeline、TransformedTargetRegressor）で正しい前置きが付く。追加したモデルを探索できる
8. **docstring**: 英語の numpy 形式。型の書き方は steering に従う

## Approach
探索範囲を「モデルの型 → 引数名 → optuna の分布」の辞書として1つのモジュールに置き、入れ子を解く関数で前置きを付けます。`ParamDistributions` はその辞書をそのまま返し、`Objective` は分布の種類に応じて `trial.suggest_*` を呼ぶ小さな変換を通して使います。`Objective` の引数の扱いは `clone` + `set_params` に統一し、NGBRegressor の専用処理をなくします。`RecommendedParams` も同じモジュールに置き、同じ入れ子の解決を使います。

## Scope
- **In**: `src/yikit/models/_optuna.py` と、新しく作る探索範囲のモジュール。`src/yikit/models/__init__.py` の公開する名前。`_linear.py`・`_svm.py` の削除。`test_optuna.py` の拡充。EnsembleRegressor を壊さないための最小限の変更（`get_best_estimator` を使うなど）
- **Out**: EnsembleRegressor の作り直し（ensemble-on-sklearn）、GBDTRegressor の修正（gbdt-fix）、CI と依存（dev-foundation）、`visualize` の学習曲線（module-quality）

## Boundary Candidates
- 探索範囲のデータ（モデルの型ごとの分布と推奨値）
- 入れ子の解決（探索するモデルと前置きを見つける）
- 利用の窓口（`Objective`、`ParamDistributions`、`RecommendedParams`）
- `Objective` の引数の扱い（固定値、`random_state` の規則、`get_best_*`）

## Out of Boundary
- 試行の評価の方法（`cross_validate` と `n_jobs=self.n_jobs`）と TPESampler の種の引き方は変えない
- 既存のモデルの探索範囲は変えない。例外は、scikit-learn の版によって使えない値（`"friedman_mse"` など）への対応
- `Objective` を包む estimator は作らない（アンサンブルの中では `OptunaSearchCV` を使う）

## Upstream / Downstream
- **Upstream**: dev-foundation（optuna>=3.0、`optuna_integration` の import、型の方針、CI）
- **Downstream**: ensemble-on-sklearn（`ParamDistributions` と入れ子の解決を使う）、licond（`Objective` を使う。リリースした版番号を伝える）

## Existing Spec Touchpoints
- **Extends**: 初期化だけしていた `objective-respect-estimator-params` を吸収した（その spec のディレクトリは削除）。元の「変えないこと: n_jobs・random_state 以外の固定値（例 SVR の gamma・kernel）」は、固定値を暗黙に入れない方針に置き換えた
- **Adjacent**: ensemble-on-sklearn（`_ensemble.py` の Objective への依存）、gbdt-fix（GBDTRegressor の探索範囲はこの spec の共通ファイルに置く）

## Constraints
- 振る舞いの変更（同じ種でも結果が変わる、n_jobs を上書きしなくなった、固定値を入れなくなった、包みを削除した）を CHANGELOG に書く（release-0.4.0 で）
- optuna>=3.0、scikit-learn>=0.24.1、Python 3.8 でも動く書き方
- `wip/0.4.0-draft` ブランチの `_optuna.py` にある LinearSVR・PLS・型ヒントの下書きは参考にする。sampler の property 化と `early_stopping_callback` は使わない
