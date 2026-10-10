# Research & Design Decisions

## Summary
- **Feature**: `optuna-tuning`
- **Discovery Scope**: Extension（既存の `models/_optuna.py` の作り直しと、探索範囲のモジュールの追加）
- **Key Findings**:
  - ngboost の `NGBRegressor` は `get_params` と `set_params` を独自に定義している。`set_params` は `setattr` するだけなので、`Base__max_depth` のような入れ子の名前は効かない（変な属性が増えるだけ）。また `random_state` は `__init__` の中で `check_random_state` により RandomState に変換されるため、`set_params(random_state=7)` で整数を入れると、`minibatch_frac < 1` のときに fit が `'int' object has no attribute 'choice'` で失敗する。コンストラクタから作り直す（`clone` と同じ方式）と、どちらも正しく動く
  - EnsembleRegressor は、今の scikit-learn（1.4 以上）では探索の有無に関係なく動かない。非公開の `sklearn.model_selection._validation._score` の引数に `score_params` が増えたため。また `boruta=False` のときは `np.bool` を使うので、NumPy 1.24〜1.26（Python 3.8 の CI）でも動かない。要件 9 の「動き続ける」は満たせないので、この spec の責任を「探索の結果からモデルを作る部分を直す」に絞った
  - optuna-integration の `OptunaSearchCV` は、`cross_validate` が全分割の学習の失敗で ValueError を出したとき、それを捕まえて `error_score`（既定は nan）で埋め、試行を FAIL として記録して探索を続ける。`Objective` も同じ扱いにする

## Research Log

### ngboost の get_params / set_params
- **Context**: 入れ子の解決と `random_state` の規則が NGBRegressor の Base にも効くかを確かめる
- **Sources Consulted**: ngboost 0.5.8 の `ngboost/ngboost.py`（`set_params`・`get_params`・`__init__`）、`ngboost/learners.py`。yikit-dev 環境での実行
- **Findings**:
  - `get_params(deep=True)` の `deep` は無視され、`Base__*` は出てこない。`set_params(**p)` は `setattr` だけ
  - `NGBRegressor(random_state=None).get_params()["random_state"]` は NumPy の大域の RandomState（`check_random_state(None)` が返すもの、`np.random.mtrand._rand`）そのもの。None を渡したかどうかは、この同一性でしか判定できない
  - `clone(NGBRegressor(random_state=7))` は RandomState を deepcopy して作り直すので、学習の結果は再現する
  - 既定の Base（`default_tree_learner`）は `criterion="friedman_mse"` の DecisionTreeRegressor で、モジュールの中の1つのインスタンスを共有している
- **Implications**:
  - 引数をモデルに入れる処理は、`set_params` に頼らず、入れ子の先を写して作り直し、外側をコンストラクタから作り直す
  - `random_state` が未指定かどうかは、写す前の元のモデルで「None か、大域の RandomState と同一か」で判定する
  - `OptunaSearchCV` は `set_params` を直接呼ぶので、NGBRegressor の `Base__*` は効かない。yikit では直せないので、`ParamDistributions` で警告する

### EnsembleRegressor の現状
- **Context**: 要件 9（探索ありの EnsembleRegressor が動き続ける）の確認
- **Sources Consulted**: `src/yikit/models/_ensemble.py`、scikit-learn 1.9.1 での実行
- **Findings**:
  - `EnsembleRegressor(estimators=[SVR()], boruta=False, opt=True).fit(X, y)` は `TypeError: _score() missing 1 required positional argument: 'score_params'` で失敗する（scikit-learn 1.4 で非公開の `_score` の引数が変わった）
  - `boruta=False` の分岐は `np.bool` を使う（NumPy 1.24〜1.26 では AttributeError、2.0 以降は再び存在する）
  - `objective.model(**objective.fixed_params_, **study.best_params)` は、型から作り直すので探索しない引数が既定値に戻る。NGBRegressor では `Base__max_depth` をコンストラクタに渡すので TypeError になる
- **Implications**: この spec では、探索の結果を `get_best_estimator` で作る形にだけ直す。全体を今の scikit-learn で動かすのは ensemble-on-sklearn の作り直しに任せる（要件 9 をそのように直した）

### 学習に失敗した試行の扱い
- **Context**: 要件 1.8。PLS の `n_components` が特徴量の数を超えるときなど
- **Sources Consulted**: optuna-integration の `optuna_integration/sklearn/sklearn.py`（`_Objective.__call__`）、scikit-learn の `cross_validate`（`_warn_or_raise_about_fit_failures`）、実行での確認
- **Findings**:
  - `cross_validate` は、一部の分割の失敗では警告と nan、全分割の失敗では ValueError を出す（scikit-learn 1.0 以降）
  - `OptunaSearchCV` は ValueError を捕まえて nan で埋める。optuna は目的関数が nan を返すと、その試行を FAIL として記録し、警告を出して次へ進む（最良の試行には選ばれない）
  - `OptunaSearchCV(PLSRegression(), {"n_components": IntDistribution(1, 10)})` を特徴量 3 のデータで実行すると、上限を超えた試行は FAIL になり、探索は最後まで進む
- **Implications**: `Objective` も `cross_validate` の ValueError を捕まえ、元のメッセージを警告として出してから nan を返す

### optuna の分布と suggest
- **Context**: 探索範囲を optuna の分布として1か所に持ち、`Objective` では `trial.suggest_*` に読み替える
- **Sources Consulted**: optuna 3.0 の `IntDistribution`・`FloatDistribution`・`CategoricalDistribution`、`Trial.suggest_int`・`suggest_float`・`suggest_categorical`、`FixedTrial`。optuna 5.0.0 での実行
- **Findings**:
  - `suggest_loguniform(a, b)` は `suggest_float(log=True)` と同じ分布（`FloatDistribution(log=True)`）を記録する。読み替えで探索の値の列は変わらない
  - 試行が使った分布は `trial.distributions` に残る。`Objective` と `ParamDistributions` の一致は、これを比べればテストできる
  - `FixedTrial(params)` に `suggest_*` すると、渡した値をそのまま返す。最良の試行の値から、試行と同じ手順で引数の辞書を作り直せる
- **Implications**: 分布から suggest への読み替えは、3種類の分布の分岐だけで済む

### scikit-learn の版と criterion
- **Context**: 要件 3.7。NGBRegressor の Base の `criterion` の候補
- **Findings**:
  - `"squared_error"` は scikit-learn 1.0 で追加された（それまでは `"mse"`。`"mse"` は 1.0 で非推奨、1.2 で削除）
  - `"friedman_mse"` は 1.9 で非推奨になり、1.11 で削除される（「`"squared_error"` と常に同じだった」という FutureWarning）
- **Implications**: 候補は、1.0 未満で `["mse", "friedman_mse"]`、1.0 以上 1.9 未満で `["squared_error", "friedman_mse"]`、1.9 以上で `["squared_error"]` とする。引数の名前（`Base__criterion`）は版によらず同じにする

### 型の照合の順番
- **Findings**: scikit-learn では `Lasso` が `ElasticNet` の派生クラスである
- **Implications**: 探索範囲の表は上から順に `isinstance` で照合するので、`Lasso` を `ElasticNet` より前に置く。そうしないと、Lasso に存在しない `l1_ratio` を探索してしまう

## Architecture Pattern Evaluation

| Option | Description | Strengths | Risks / Limitations | Notes |
|--------|-------------|-----------|---------------------|-------|
| Python のモジュールに表として持つ | 型と、optuna の分布の辞書の組の表 | 依存が増えない。optional な型と版による分岐を書ける。型検査が効く | モデルを足すたびに表が長くなる | 採用 |
| YAML・TOML の外部ファイル | 型の名前と範囲をデータとして書く | 編集しやすい | 依存（PyYAML、3.11 未満の tomli）とパッケージへの同梱の設定が増える。optional な型と版による分岐はコードに残る | 不採用 |
| Hydra | アプリの設定の組み立て | コマンドラインからの上書き | アプリ向けの仕組みで、ライブラリが中に持つ既定値には合わない。omegaconf などの依存が増える | 不採用。使う側（licond など）が Hydra で範囲を管理し、`custom_params` で渡せばよい |

## Design Decisions

### Decision: 探索範囲は分布の表として1か所に持つ
- **Context**: 要件 3、6.1。今は `Objective` と `ParamDistributions` に同じ範囲を2回書いている
- **Alternatives Considered**:
  1. モデルごとに分布を返す関数を書く
  2. 型と分布の辞書の組を並べた表（データ）にする
- **Selected Approach**: 2。表の値は optuna の分布そのもの。特徴量の数で決まる上限（PLS の `n_components`）は表に定数 10 で書き、特徴量の数が分かるときだけ `min` で切り詰める
- **Rationale**: 関数を含まないので読みやすい。`Objective` と `ParamDistributions` は同じ表から作るので、ずれようがない
- **Trade-offs**: 特徴量の数による切り詰めは、該当する引数名を表の外で知っている必要がある（今は `n_components` だけ）
- **Follow-up**: 表の照合の順番（`Lasso` を `ElasticNet` より前）をテストで確かめる

### Decision: 引数はコンストラクタから作り直して入れる
- **Context**: 要件 1.1、1.7、2.5。ngboost の `set_params` は入れ子の名前を解かず、`random_state` を変換しない
- **Alternatives Considered**:
  1. `clone(estimator).set_params(**params)`（brief の案）
  2. 入れ子の先を写して作り直し、外側を `type(est)(**{**est.get_params(deep=False), **params})` で作り直す
- **Selected Approach**: 2。入れ子の名前は先頭の名前ごとにまとめ、中のモデルに再帰的に入れてから、外側の引数として渡す。コンストラクタの引数でない名前（`Pipeline` の段の名前）だけは `set_params` で入れる
- **Rationale**: `clone` が使う方式なので、sklearn 互換のすべてのモデルで動く。ngboost でも、入れ子の名前と整数の `random_state` が正しく入る。入れ子の先も写すので、元のモデルと物を共有しない
- **Trade-offs**: 独自の処理が1つ増える。存在しない名前の検査を自分で行う必要がある
- **Follow-up**: ngboost で `Base__max_depth` と `random_state` が効くこと、元のモデルが変わらないことをテストする

### Decision: random_state が未指定かどうかは元のモデルで判定する
- **Context**: 要件 2.1、2.2。ngboost は None を大域の RandomState に変換する
- **Selected Approach**: `Objective` を作るときに、元のモデルの入れ子をすべてたどり、`random_state` の値が None か、`check_random_state(None)` が返す大域の RandomState と同一のものを「未指定」として名前を集める。`get_params(deep=True)` が入れ子を出さないモデル（ngboost）は、値が estimator の引数を自分で再帰してたどる
- **Rationale**: 写した後では大域の RandomState が deepcopy されて同一性で判定できなくなる。None と大域の RandomState は、どちらも「大域の乱数を使う」という同じ意味
- **Trade-offs**: 利用者が `random_state=np.random.mtrand._rand` を明示した場合も未指定として扱う（意味は同じなので問題にしない）

### Decision: Objective の公開する属性を減らす
- **Context**: `Objective.model`（型）と `fixed_params_` は EnsembleRegressor だけが使っていた。`model` は `Pipeline` では意味がない
- **Selected Approach**: `model` と `fixed_params_` をなくす。最後の試行のモデル `estimator_` は残す（試行のモデルの引数をテストで確かめるため）。乱数の整数を `model_random_state`、探索範囲を `param_distributions` として公開する
- **Trade-offs**: `model`・`fixed_params_` を使う外部のコードは壊れる。CHANGELOG（release-0.4.0）に書く

### Decision: custom_params の関数を FixedTrial で評価する
- **Context**: 要件 1.6、5.3。`custom_params` の関数は、返す辞書の名前と suggest の名前が違うことがある
- **Selected Approach**: `get_best_params` は、最良の試行の値を持つ `FixedTrial` で、試行と同じ引数の組み立てをもう一度行う。`ParamDistributions` の `custom_params` が関数のときは `FixedTrial({})` を渡して1回呼ぶ（分布を返すだけの関数なら動き、suggest を呼ぶ関数ならエラーになる）
- **Rationale**: 試行と最良の引数が同じ手順で作られる。今の `ParamDistributions` の、使い捨ての study を作って例外を握りつぶす処理をなくせる

## Risks & Mitigations
- 0.4.0-rc.0 と同じ種でも探索の結果が変わる（`random_state` と `n_jobs` を上書きしなくなるため）— 既存のテストの期待値を作り直し、CHANGELOG に書く
- ngboost の `Base__*` は `OptunaSearchCV` では効かない — `ParamDistributions` で警告し、`Objective` を使うよう案内する
- `Pipeline` の中で特徴量の選択が前にあると、PLS の上限が実際の特徴量の数を超えうる — 学習に失敗した試行は FAIL として記録して探索を続ける（要件 1.8、6.4）
- EnsembleRegressor は今の scikit-learn では動かないまま — ensemble-on-sklearn で作り直す。brief にこの事実を書き足した

## References
- scikit-learn: `sklearn.base.clone`、`BaseEstimator.set_params`、`TransformedTargetRegressor`、`Pipeline`
- optuna: `optuna.distributions`、`optuna.trial.FixedTrial`、`Trial.distributions`
- optuna-integration: `OptunaSearchCV`（`optuna_integration/sklearn/sklearn.py`）
- ngboost 0.5.8: `ngboost/ngboost.py`（`get_params`・`set_params`）
