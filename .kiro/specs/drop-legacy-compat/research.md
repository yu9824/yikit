# Research & Design Decisions

## Summary
- **Feature**: `drop-legacy-compat`
- **Discovery Scope**: Extension（既存のコードから古い版のための分岐を消し、依存の宣言を合わせる。新しい依存はない。各ライブラリの版の事実は、リリースノート・changelog・タグを付けたソースで確かめた）
- **Key Findings**:
  - 古い版のための分岐は brief に挙がったものの他に、ngboost<0.4 の判定（`models/_search_space.py` の `set_params` の比較とそのテスト）、optuna<3.1 の判定（`tests/test_visualize.py` の `create_new_study` の引数）、ngboost<0.4 の skip（Student's t）がある。新しい版のための分岐（scikit-learn 1.6・1.9、LightGBM 4.7、ngboost 0.5、joblib 1.3、matplotlib 3.10）は残す
  - ngboost 0.5.9・0.5.10 は、`log_likelihood` が使う部分（`Distn._params`・`manifold`・`uncensor`・`total_score`・各分布の score）も、`set_params`・`get_params` も 0.5.8 と同じ。0.5.11 は `set_params` が入れ子の名前（`Base__max_depth`）を `Base` に渡し、`get_params(deep=True)` が `Base__*` を返すように変わった。yikit の「ngboost は入れ子の名前を無視する」という判定と警告と docstring がずれるので、上限は `<0.5.11` にする
  - optuna-integration は 4.1.0 から `OptunaSearchCV.predict` が普通のメソッドになった（4.0.0 までは property）。その `sklearn.py` は `optuna.terminator.report_cross_validation_scores`（optuna 3.2.0 から）を import するので、optuna-integration>=4.1 には optuna>=3.2 が要る

## Research Log

### 古い版のための分岐の洗い出し（2026-10-14、ブランチ `docs/spec-0.5.0`）
- **Context**: brief の Current State を、実際のコードとテストで確かめ、漏れを探す
- **Sources Consulted**: `src/yikit` と `tests` への `grep`（`__version__`、`version_info`、`hasattr(`、`except ImportError`・`AttributeError`・`TypeError`、`inspect.signature`、`_FIT_PARAMETERS`、`is_argument(`、ライブラリ名と版の番号、`older`・`old`・`before`・`since`・`or later`・`deprecat`）、各ファイルの該当の箇所
- **Findings**:
  - scikit-learn<1.0: `helpers/_validation.py` の `_sklearn_has_feature_names` と `feature_names_of` の判定、`models/_ensemble.py` の同じ判定の複製（`_sklearn_has_feature_names`・`_feature_names`。中身は `feature_names_of` と同じ）、`models/_search_space.py` の `_criterion_choices` の `"mse"`。docstring とコメント: `_ensemble.py` の `feature_names_in_`、`_search_cv.py` の `feature_names_in_`、`feature_selection/_filter.py` の 3 か所（`SelectorMixin` の import の `# >= 0.23.2`、`feature_names_in_` の説明、例の `# scikit-learn >= 1.0`）、`visualize/_yyplot.py` の `_rmse`（`works with scikit-learn >= 0.24.1`）。テスト: `test_helpers.py`（版を差し替える 3 つのテストと各テストの差し替え）、`test_filtermethod.py` の `requires_feature_names_out`、`test_ensemble.py` と `test_search_cv.py` の `HAS_FEATURE_NAMES` と skip、`test_search_space.py` の `0.24.1`・`0.24.2` の行、`test_optuna.py` の許容値のコメント（`0.24.1`）
  - optuna 3.0〜3.6 と optuna-integration≤4.0.0: `models/_search_cv.py` の import の代わりの場所と `predict` の上書き、`models/_optuna.py` の `ParamDistributions` の例の import。テスト: `test_search_cv.py`（import の代わり、`_PropertyPredictSearchCV`・`_OldParentSearchRegressor` と 1 つのテスト）、`test_optuna.py` と `test_ensemble.py` の import の代わり、`test_ensemble.py` の import を塞ぐテストの `"optuna.integration"`
  - optuna<3.1（新しく見つけた）: `tests/test_visualize.py` の `_make_study_without_direction` が `create_new_study` に `directions` があるかで分けている
  - LightGBM 3: `models/_gbdt.py` の `verbose` の判定と、`best_iteration_` を booster から読む理由、class と関数の docstring。`visualize/_gbdt.py` の `None` についてのコメント（コードの `not evals_result` は 4 の `{}` にも要るので残す）。テスト: `test_gbdt.py` の `_fit_3_2`・`_fit_3_3` と `passes_verbose`、`test_visualize.py` の 2 つのコメント（`4.6 on Python 3.8`、`None before`）
  - ngboost 0.3（新しく見つけたものを含む）: `models/_search_space.py` の `_build_ignores_nested_set_params` が `NGBRegressor.set_params is not BaseEstimator.set_params` で 0.4.0 より前を見分けている。docstring の「Older ngboost inherits」。`models/_ensemble.py` の「ngboost 0.3.x」。`metrics/_ngboost.py` の「0.3.6 through 0.5.8」。テスト: `test_optuna.py` の `_ngboost_set_params_ignores_nested_names` と分岐・skip、`test_search_space.py` のコメント、`test_visualize.py` の Student's t の skip とコメント、Python 3.14 についてのコメント、`test_metrics.py` の 0.3.6 の注記、`test_params.py` の `deep=False` の注記
  - seaborn<0.12: `visualize/_permutation_importance.py` の `errorbar` と `ci` の書き分け（`is_argument` を使う）
  - numpy<1.21: `visualize/_yyplot.py` の `NDArray` のコメントだけ。`NDArray` は他のモジュールでも型のためだけに `TYPE_CHECKING` の中で import している
  - Boruta<0.4（下限 0.4.3 はすでにある）: `feature_selection/_wrapper.py` のコメント 1 行
  - 新しい版のためのもの（残す）: `_ensemble.py` の `is_regressor` の `AttributeError`（scikit-learn 1.6）、`_search_cv.py` の `_estimator_type` と `__sklearn_tags__`（1.6 の前と後）、`_search_space.py` の 1.9 の `criterion`、`_params.py` の `clone` が運ぶ状態（1.2・1.3・1.9）、`_gbdt.py` の `eval_X`（LightGBM 4.7）、`visualize/_gbdt.py` の `eval_X` の案内、`_permutation_importance.py` の seaborn 0.13 の並び、`_yyplot.py` の matplotlib<3.10 の凡例、`visualize/_optuna.py` の `LegendLocType`（matplotlib に下限を書かない）、`_helpers.py` の joblib 1.3 の `return_as`、テストの scikit-learn 1.2・1.3・1.6・1.9、PLSRegression<1.3、joblib<1.3、matplotlib<3.5、seaborn 0.13、ngboost 0.5 の tags
  - 版ではないもの（残す）: `_ensemble.py` の `n_features_in_` の `AttributeError`（最初のモデルが記録しない）、`_search_cv.py` の `best_estimator_` の `AttributeError`、`test_search_space.py` の「Old scikit-learn validates the steps in `__init__`」（scikit-learn 1.0 の `Pipeline.__init__` も検証するので、下限の範囲でも要る）、`test_boruta.py` の観察の記録（scikit-learn 0.24.1 を含む環境ごとの結果）
- **Implications**: 分岐の判断は design.md の「分岐の一覧」に 1 行ずつ書く。ファイルが複数のライブラリにまたがるので、タスクの境界はライブラリではなくファイルのまとまりで切る

### scikit-learn の下限（1.0）
- **Context**: `feature_names_in_`、`get_feature_names_out`、`"squared_error"` が入った版
- **Sources Consulted**: `scikit-learn` の `doc/whats_new/v1.0.rst`（1.0.X ブランチ）、`sklearn/feature_selection/_base.py`・`sklearn/base.py`・`sklearn/ensemble/_voting.py`・`sklearn/pipeline.py`（1.0.X、1.1.X）
- **Findings**:
  - 1.0.0 の changelog: 「All estimators store `feature_names_in_` when fitted on pandas Dataframes」、「`get_feature_names_out` has been added to the transformer API」、木の `criterion="mse"` は非推奨で `"squared_error"` が既定
  - 1.0.X の `SelectorMixin` に `get_feature_names_out`、`BaseEstimator` に `_check_feature_names`、`VotingRegressor` に `feature_names_in_` の受け渡しがある
  - 1.0.X の `Pipeline.__init__` は `_validate_steps()` を呼ぶ（1.1 から呼ばない）
  - Python 3.10 で入る scikit-learn は 1.7.x まで（1.9 は 3.11 以上）。1.0.2 が cp310 の wheel を持つ最初の版
- **Implications**: `scikit-learn>=1.0`。1.6 と 1.9 の分岐は Python 3.10〜3.14 の CI で両方の道が通る

### lightgbm の下限（4.0）
- **Context**: `verbose` の判定と、`best_iteration_`・`evals_result_` の LightGBM 3 での違い
- **Sources Consulted**: `microsoft/LightGBM` の `python-package/lightgbm/sklearn.py`・`callback.py`（タグ v3.3.5、v4.0.0）
- **Findings**:
  - 4.0.0 の `LGBMModel.fit` の引数に `verbose` と `early_stopping_rounds` はない。`early_stopping` の callback はある
  - 4.0.0 の `fit` は `self._best_iteration = self._Booster.best_iteration` と `self._evals_result = evals_result`（初期値 `{}`）を必ず入れる。`best_iteration_` と `evals_result_` はそれを返す
  - 3.3.5 は `early_stopping_rounds` を渡したときだけ `_best_iteration` を入れ、それ以外は None。検証データがないと `_evals_result` は None。`verbose` の既定は `'warn'`
  - lightgbm 4.7.0 は Python 3.10 以上。CI の 5 つの版はどれも 4.7 を入れる
- **Implications**: `lightgbm>=4.0`。`verbose` の判定を消しても 4.x の引数は変わらない。`best_iteration_` は `estimator_.best_iteration_` と同じ値。`eval_X` の分岐（4.7）は残し、4.0〜4.6 の道は署名を差し替える単体テストで確かめる

### seaborn の下限（0.12）
- **Context**: `barplot` の `errorbar`
- **Sources Consulted**: `mwaskom/seaborn` の `doc/whatsnew/v0.12.0.rst`、`seaborn/categorical.py`（タグ v0.12.0）、PyPI の JSON API
- **Findings**: 0.12.0 で `barplot` に `errorbar=("ci", 95)` が入り、`ci` は `"deprecated"` になった。0.12.0 は Python 3.7 以上。Python 3.10 で入る最新は 0.13.2
- **Implications**: `seaborn>=0.12`。`errorbar=None` を常に渡す。`is_argument` の import が `_permutation_importance.py` から消える（`is_argument` は helpers の公開 API のまま）

### ngboost の下限（0.4）と上限（<0.5.11）、Python 3.14
- **Context**: brief の「`log_likelihood` が 0.5.9〜0.5.11 で正しく動くか」と、ngboost<0.4 の判定
- **Sources Consulted**: PyPI の JSON API（各版の `requires_python`・`requires_dist`・公開日）、`stanfordmlgroup/ngboost` のタグ v0.3.6・v0.3.13・v0.4.0・v0.4.2・v0.5.0〜v0.5.11 の `ngboost/ngboost.py`・`api.py`・`manifold.py`・`scores.py`・`distns/{distn,normal,lognormal,exponential,categorical,__init__}.py`・`learners.py`・`helpers.py`、GitHub の compare API（v0.5.8...v0.5.9、v0.5.9...v0.5.10、v0.5.10...v0.5.11、v0.4.0...v0.5.8）、yikit-dev（ngboost 0.5.8）での読み取りだけの実行
- **Findings**:
  - 版と Python: 0.4.0〜0.4.2 は `>=3.7.1,<3.11`、0.5.0〜0.5.6 は `<3.13`、0.5.7・0.5.8 は `<3.14`、0.5.9（2026-02-25）・0.5.10（2026-03-24）・0.5.11（2026-06-26）は `<3.15`。0.5.5 以降は scikit-learn>=1.6 を求め、0.5.9 以降は sympy と（Python 3.10 以上で）matplotlib>=3.10 を求める。どの版も lifelines>=0.25 を求め、最新の lifelines 0.30.3 は pandas<3.0 を求める
  - `set_params`: 0.3.6・0.3.13 の `NGBoost` は `set_params`・`get_params` を持たず、`BaseEstimator` のものを使う（入れ子の名前を渡す）。0.4.0〜0.5.10 は `setattr` だけの `set_params` と、`deep` を無視する `get_params` を持つ。0.5.11 は `set_params` が入れ子の名前を `Base`（またはその列）の `set_params` に渡し、`get_params(deep=True)` が `Base__*` を返す
  - `log_likelihood` が使う部分: `Distn.__init__` の `self._params`、`Distn.uncensor` と `DistWithUncensoredScore`、`manifold`、`NGBoost.score`（`self.Manifold(self.pred_dist(X)._params).total_score(Y)`）は 0.4.0・0.5.8・0.5.9・0.5.10・0.5.11 で同じ。`scores.py` は 0.4.0 から 0.5.8 の間に自然勾配の解き方（`_natural_gradient`）だけが変わり、0.5.8〜0.5.11 は同じ。`normal.py`・`lognormal.py`・`exponential.py`・`categorical.py` は 0.5.8〜0.5.11 で同じ
  - 0.5.9 は分布の追加（Beta など）と 3.14 への対応だけ。0.5.10 は `helpers.py` に古い pickle を読む `load_ngboost_model` を足しただけ（`Y_from_censored` は同じ）。0.5.11 は `set_params`・`get_params` の変更、`Base` に学習器の列を渡す機能、`validation_fraction`・`early_stopping_rounds` の受け渡しの修正、既定の木の `criterion` を `"squared_error"` にした
  - `NGBRegressor._estimator_type = "regressor"` は 0.5.2 から（0.3.6〜0.5.1 にはない）。`ngboost.distns.T` は 0.4.0 にある
  - LogNormal の CRPScore: `lognormal.py` は 0.3.6〜0.5.8 で書式の違いしかない。`scores.py` は 0.3.6 と 0.4.0 で docstring の改行だけが違い、0.5.0〜0.5.7 にも `_natural_gradient` はないので、自然勾配の特異な行列への対処（`pinv` への切り替え）は 0.5.8 から。yikit-dev の 0.5.8 では学習でき、`log_likelihood` は `pred_dist(X).logpdf(y)` の平均と一致した
  - yikit-dev（0.5.8）では `NGBRegressor.set_params is not BaseEstimator.set_params` が True、`is_regressor(NGBRegressor())` が False（scikit-learn 1.9.1）
- **Implications**:
  - 下限 `ngboost>=0.4`: `_build_ignores_nested_set_params` の比較を消すと、0.4.0 以降の `set_params` を前提にする。Student's t の skip も消せる
  - `log_likelihood` は 0.5.9〜0.5.11 でもソースの上では同じ値を返す。上限を決めるのは models の側で、0.5.11 の `set_params` の変更が「入れ子の名前は無視される」という `ignores_nested_set_params`・`ParamDistributions` の警告・`EnsembleRegressor` の docstring とずれる。0.5.11 では `OptunaSearchCV` で `Base__*` を探索すると実際に効くので、0.5.10 までと結果も変わる
  - 上限は `<0.5.11`。0.5.9・0.5.10 は Python 3.14 に対応しているので、環境の条件（`python_version < '3.14'`）を消しても 3.14 で ngboost が入る。上限を残したまま条件だけ消すと、3.14 では入る版がなくて解決に失敗するので、両方を同時に変える
  - 0.5.11 への対応（入れ子の名前を渡すかどうかを振る舞いで判定し、警告と docstring を版に合わせる）は、この spec の外の作業として残す
  - 3.14 でも ngboost が入ると、lifelines を通じて pandas<3.0 になる。CI のどの版でも pandas 3 は入らなくなる

### optuna と optuna-integration の下限（3.2 と 4.1）
- **Context**: import の代わりの場所と `predict` の上書きを消すのに要る版
- **Sources Consulted**: PyPI の JSON API（optuna・optuna-integration の各版）、`optuna/optuna-integration` の `optuna_integration/sklearn/sklearn.py`（タグ v3.6.0、v4.0.0、v4.1.0、v4.2.0）と `optuna_integration/__init__.py`（v4.1.0）、`optuna/optuna` の `optuna/terminator/__init__.py`（v3.0.0、v3.1.0、v3.2.0）と `optuna/storages/_base.py`（v3.0.0、v3.1.0、v3.2.0）
- **Findings**:
  - optuna-integration は 3.2.0（2023-05）が最初。どの版も `optuna` を版の指定なしで求める。4.1.0 は Python 3.8 以上、4.6.0 以降は 3.9 以上
  - `OptunaSearchCV.predict`・`predict_proba`・`predict_log_proba`・`decision_function` は 4.0.0 まで `@property`、4.1.0 から `predict(self, X, **kwargs)` の普通のメソッド。どちらも `self._check_is_fitted()` の後に `best_estimator_` の同じメソッドを呼ぶ。4.1.0 は `param_distributions` に `Mapping` も受け付け、`catch` を足した
  - optuna-integration の `sklearn.py`（3.6.0〜4.2.0）は `from optuna.terminator import report_cross_validation_scores` を持つ。`optuna.terminator` は optuna 3.2.0 から（3.0.0・3.1.0 にはない）。他に使う `optuna._experimental`・`optuna._imports.try_import`・`_convert_old_distribution_to_new_distribution`・`Trial._suggest` は 3.0 からある
  - optuna の `BaseStorage.create_new_study` の `directions` は 3.1.0 から
  - yikit の `ParamDistributions` は `dict` の子クラスなので、4.0.0 の「dict でなければ TypeError」にも 4.1.0 の `Mapping` にも合う
  - Python 3.10 で入る最新は optuna 5.0.0 と optuna-integration 5.0.0
- **Implications**: `optuna-integration>=4.1`、`optuna>=3.2`。`optuna>=3.0` のままだと、optuna 3.0・3.1 と optuna-integration 4.1 の組み合わせで `OptunaSearchCV` の import に失敗する。`create_new_study` の分岐（テスト）も消せる。`optuna-integration` を入れずに optuna 3.2〜3.6 だけを入れた環境では、今まで動いていた `EnsembleRegressor(opt=True)` が ImportError になる（optional の extra を使えば起きない。CHANGELOG に書く）

### 下限を書かないライブラリ
- **Context**: 作者の決定（numpy・pandas・scipy・matplotlib・joblib は、消す分岐がなければ下限を書かない）
- **Sources Consulted**: 上の洗い出し、discovery の PyPI の事実（cp310 の最初の wheel: numpy 1.21.2、pandas 1.3.3、scipy 1.7.2、matplotlib 3.5.0）
- **Findings**:
  - numpy: 消すのはコメントだけ。`NDArray` は型のためだけの import なので、実行時に numpy 1.21 を要らない。`numpy.typing.ArrayLike`（1.20）は `_yyplot.py` で実行時に import しているが、分岐はない
  - matplotlib・joblib の分岐（matplotlib<3.10 の凡例、`LegendLocType`、joblib<1.3）は下限を書かないので残す
- **Implications**: 下限を書くのは scikit-learn・seaborn・lightgbm・ngboost・optuna・optuna-integration の 6 つだけ（Boruta は今のまま）

### 使ったスキルと規則
- kiro-spec-requirements、kiro-spec-design（design-principles、design-discovery-light、design-synthesis、design-review-gate）、kiro-spec-tasks（tasks-generation、tasks-parallel-analysis）の各ファイル。共有の文脈（`spec-0.5.0-context.md`）の決定と禁止事項（インストールとダウンロードをしない、spec のディレクトリの外に書かない）に従った。ソースは GitHub の raw のファイルと API を読むだけにした

## Architecture Pattern Evaluation

| Option | Description | Strengths | Risks / Limitations | Notes |
|--------|-------------|-----------|---------------------|-------|
| ライブラリごとの作業単位 | brief の Boundary Candidates のとおり、scikit-learn・optuna・LightGBM・ngboost・seaborn ごとに分ける | 下限の理由と削除が 1 つにまとまる | 同じファイル（`_ensemble.py`、`_search_space.py`、`_search_cv.py`、`test_visualize.py` など）を複数の単位が触り、並行できない | 下限の理由は design の表で 1 か所にまとめれば足りる |
| ファイルのまとまりごとの作業単位（採用） | モジュールとそのテストの組ごとに、含まれるすべてのライブラリの分岐を消す | ファイルが重ならず並行できる。レビューもモジュール単位 | 1 つの単位に複数のライブラリが入る | 下限の宣言は最初の 1 つの単位にまとめる |

## Design Decisions

### Decision: 上限を `<0.5.11` にし、Python 3.14 の除外をやめる
- **Context**: Requirement 2。brief は「0.5.9〜0.5.11 で `log_likelihood` が動くか確かめてから上げるか外す」
- **Alternatives Considered**:
  1. 上限を外す（0.5.11 を入れる）
  2. `<0.5.11`（0.5.10 まで）
  3. `<0.5.9` のまま、3.14 の除外も残す
- **Selected Approach**: 2。0.5.9・0.5.10 は yikit が使う部分が 0.5.8 と同じで、3.14 に対応している
- **Rationale**: 0.5.11 は `log_likelihood` には影響しないが、`set_params`・`get_params` の変更で yikit の判定・警告・docstring がずれ、`OptunaSearchCV` での探索の結果も変わる。合わせるには新しい判定と警告の変更が要り、この spec の範囲（新しい機能を足さない、宣言した範囲で振る舞いを変えない）を越える
- **Trade-offs**: 最新の ngboost（0.5.11）を入れた環境では、pip が 0.5.10 に下げる。dependabot が上限を広げる PR を出すことがある
- **Follow-up**: 作者の承認を得て、手元で 0.5.10 のテストを回す。0.5.11 で落ちるテストを記録し、対応の作業を作者に確認する

### Decision: 下限を書く 6 つのライブラリと版
- **Context**: Requirement 1。作者の決定は「消した分岐が要る分だけ」
- **Selected Approach**: `scikit-learn>=1.0`、`seaborn>=0.12`、`lightgbm>=4.0`、`ngboost>=0.4,<0.5.11`、`optuna>=3.2`、`optuna-integration>=4.1`。numpy・pandas・scipy・matplotlib・joblib・colorlog・tqdm には書かない。`Boruta>=0.4.3` は変えない
- **Rationale**: 各版は、残るコードが使う機能が入った最初の版（上の Research Log）。optuna の 3.2 は、yikit 自身の分岐ではなく、optuna-integration>=4.1 が import する `optuna.terminator` から決まる（import の代わりの場所を消した結果、`OptunaSearchCV` は optuna-integration からしか来なくなるため）
- **Trade-offs**: 下限の版で動くことは CI で確かめない（作者の決定）。CI の 3.10 の job でも、これらは最新に近い版が入る（scikit-learn 1.7、lightgbm 4.7、ngboost 0.5.10、optuna 5.0、seaborn 0.13）

### Decision: 特徴量の名前の判定を `feature_names_of` 1 つにする
- **Context**: Requirement 3.4。`models/_ensemble.py` の `_feature_names` は `helpers/_validation.py` の `feature_names_of` と同じ処理の複製
- **Selected Approach**: `_ensemble.py` は `yikit.helpers._validation.feature_names_of` を import し、`_sklearn_has_feature_names`・`_feature_names` を消す
- **Rationale**: 版の判定を消すと両者は同じ。models は基盤（helpers）に依存してよい（structure.md）。`feature_selection/_filter.py` も同じ非公開のモジュールから import している

### Decision: `NDArray` は `TYPE_CHECKING` の中に残す
- **Context**: brief の「numpy < 1.21: `NDArray` を `TYPE_CHECKING` の中で import している」
- **Alternatives Considered**: 1. 実行時に import する（numpy>=1.21 の下限が要る） 2. 型のための import として残し、numpy<1.21 のコメントだけを消す
- **Selected Approach**: 2
- **Rationale**: `from __future__ import annotations` の下では実行時に要らない。他のモジュール（`_gbdt.py`、`_ensemble.py`、`_validation.py`）も同じ書き方をしている。numpy に下限を書かないという作者の決定と合う
- **Trade-offs**: なし（`tech.md` の「型」は packaging-and-ci が持ち、この spec では触らない）

### Decision: `best_iteration_` は `estimator_.best_iteration_` から読む
- **Context**: LightGBM 3 では `best_iteration_` が None のままになるので booster から読んでいた
- **Selected Approach**: `self.best_iteration_ = self.estimator_.best_iteration_`。docstring から LightGBM 3 の説明を消す
- **Rationale**: 4.0.0 以降の `best_iteration_` は `self._Booster.best_iteration` を写した値で、同じになる（ソースで確認）。公開の属性を使う方が読みやすい
- **Follow-up**: `test_gbdt.py` で `best_iteration_ == estimator_.best_iteration_ == estimator_.booster_.best_iteration` を確かめる

## Risks & Mitigations
- 下限の版の組み合わせを CI で試さない — 各下限はソースや changelog で確かめた API だけに頼る。3.10 の job でも最新に近い版しか入らないことを tech.md に書く
- ngboost 0.5.10 を手元で試していない（yikit-dev は 0.5.8） — 作者の承認のもとで入れて試すタスクを置く。承認がなくても、CI の 5 つの版が 0.5.10 を入れて試す
- 3.14 の job で ngboost の図のテストが初めて走る — dev-foundation の比べ方（文字を消す、RMS 1.0）は版に左右されにくい。失敗したら原因を調べ、図の比べ方を変える必要があれば作者に相談する
- 3.14 でも pandas<3.0 になる（lifelines） — pandas 3 を CI で試さなくなることを報告する（作者への確認事項）
- dependabot が ngboost の上限を広げる PR を出す — 0.5.11 に対応するまでマージしないことを tech.md の Key Libraries に書く
- optuna-integration なしで optuna 3.2〜3.6 だけを入れていた利用者は、`opt=True` が ImportError になる — 案内のメッセージは今のまま。CHANGELOG の材料に書く

## References
- [scikit-learn 1.0 changelog](https://github.com/scikit-learn/scikit-learn/blob/1.0.X/doc/whats_new/v1.0.rst) — `feature_names_in_`、`get_feature_names_out`、`squared_error`
- [LightGBM v4.0.0 sklearn.py](https://github.com/microsoft/LightGBM/blob/v4.0.0/python-package/lightgbm/sklearn.py) — `fit` の引数、`_best_iteration`、`_evals_result`
- [seaborn v0.12.0 release notes](https://github.com/mwaskom/seaborn/blob/v0.12.0/doc/whatsnew/v0.12.0.rst) — `errorbar`
- [ngboost compare v0.5.10...v0.5.11](https://github.com/stanfordmlgroup/ngboost/compare/v0.5.10...v0.5.11) — `set_params`・`get_params` の変更
- [ngboost v0.4.0 ngboost.py](https://github.com/stanfordmlgroup/ngboost/blob/v0.4.0/ngboost/ngboost.py) — `setattr` だけの `set_params`
- [optuna-integration v4.1.0 sklearn.py](https://github.com/optuna/optuna-integration/blob/v4.1.0/optuna_integration/sklearn/sklearn.py) — 普通のメソッドの `predict`、`optuna.terminator` の import
- [optuna v3.2.0 terminator](https://github.com/optuna/optuna/blob/v3.2.0/optuna/terminator/__init__.py) — `report_cross_validation_scores`
- PyPI の JSON API（`https://pypi.org/pypi/<pkg>/json`）— 各版の `requires_python`・`requires_dist`・公開日
