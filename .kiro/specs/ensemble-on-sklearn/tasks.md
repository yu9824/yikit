# Implementation Plan

- [x] 1. BorutaPy を Pipeline の中でも perc="auto" のまま使えるようにする
  - `perc` の自動決定と進捗バーの用意を、`fit` から boruta の `_fit` の上書きへ移し、`fit` と `fit_transform` が同じ経路を通るようにする
  - 解決した値を学習後の属性 `perc_` に持ち、boruta が読む `self.perc` には `_fit` の間だけ解決した値を入れて、終わったら（例外のときも）元に戻す
  - 数値の `perc` を渡したときは、その値を `perc_` に入れる
  - 変えた振る舞いを英語の numpy 形式の docstring に書く（`perc_` の説明を含む）
  - `tests/test_boruta.py` の先頭の BorutaPy の import を、boruta がなければ飛ばす形（`pytest.importorskip`）にする
  - テスト（`tests/test_boruta.py` に追記。BorutaPy は `max_shuf` を小さく、`verbose=0`、`n_jobs=1`、`random_state` を整数にして速く・決定的にする）: `perc="auto"` での `fit_transform` が `fit` と同じ特徴量を選ぶ、`fit` の後も `get_params()["perc"] == "auto"` で `perc_` が数値、`perc=80` なら `perc_ == 80`、`Pipeline([("boruta", BorutaPy(...)), ("ridge", Ridge())])` で学習・予測できる。小さなデータと少ない `max_iter` で速くする
  - 完了の状態: Pipeline に入れた `BorutaPy(perc="auto")` が学習・予測でき、学習の後も引数 `perc` が "auto" のまま、`tests/test_boruta.py` が通る
  - _Boundary: BorutaPy_
  - _Requirements: 4.2, 4.3, 7.3, 7.4_

- [x] 2. (P) OptunaSearchCV をアンサンブルが回帰モデルとして受け付けるようにする
  - 新しいモジュールに、`OptunaSearchCV` を継承し、回帰モデルの印（`_estimator_type` と `__sklearn_tags__`）だけを足したクラスを作る。`__init__` は上書きせず、モジュールの直下に置く（pickle のため）。`OptunaSearchCV` は optuna-integration から、なければ `optuna.integration` から import する
  - `verbose=0` のときは `fit` の間だけ optuna のログを WARNING にし、終わったら（例外のときも）元に戻す
  - 英語の numpy 形式の docstring を付ける
  - テスト（`tests/test_search_cv.py`、optuna-integration がなければ飛ばす）: `is_regressor` が True、VotingRegressor と StackingRegressor に入れて学習・予測できる、`clone` と pickle の往復、`verbose=0` で optuna の INFO のログが出ず `fit` の後に verbosity が戻る、`verbose=1` では抑えない
  - 完了の状態: `StackingRegressor([("ridge", OptunaSearchRegressor(Ridge(), ParamDistributions(Ridge()), n_trials=2))])` が scikit-learn 1.9 でも学習でき、`tests/test_search_cv.py` が通る
  - _Boundary: OptunaSearchRegressor_
  - _Requirements: 2.4, 2.8, 7.3_

- [ ] 3. EnsembleRegressor を scikit-learn のアンサンブルの上に作り直す
- [ ] 3.1 引数・検査・組み立て・学習後の属性を作り、古い実装を置き換える
  - 引数を design のとおりにする（`boruta` をなくし、`n_trials` を足し、`n_jobs` の既定を None にする）。`__init__` は属性に入れるだけにする
  - `fit` で `method`・空の `estimators`・回帰モデルでないものを検査し、名前と型を示す ValueError にする。モデルだけの並びには `make_pipeline` と同じ規則で名前を付け、`(名前, モデル)` の組はそのまま使う
  - `method` に応じて VotingRegressor か StackingRegressor（stacking は `LinearRegression()`、blending は `LinearRegression(positive=True, fit_intercept=False)`、`cv` を渡す）を組み立て、`n_jobs`・`verbose` を渡し、X と y を変換せずに学習する。`opt=True` のときは各モデルを 2 のクラスと `ParamDistributions`（X の列の数）で包む。調整に使う import（optuna、`yikit.models._optuna`、2 のモジュール）は、すべて `opt=True` のときだけ `fit` の中で行う（optuna のない環境でも `import yikit.models` を壊さないため）
  - 学習後の属性（`estimator_`、`estimators_`、`named_estimators_`、`final_estimator_`、`weights_`、`n_features_in_`、あれば `feature_names_in_`）を写し、`predict` は学習を確かめてから委ねる。`results_` と独自の交差検証・各分割のモデルの平均・非公開の `_score` を使う古い実装を消す
  - 英語の numpy 形式の docstring を付ける（`n_jobs` が層ごとに掛け算になること、外側で並列にするときは各モデルを `n_jobs=1` にすること、stacking と blending では調整が (cv+1) 回行われる計算量の目安を含む）。Examples は `opt=False` の最小の例にする（Boruta と調整の例は 4 で足す）
  - `tests/test_optuna.py` の EnsembleRegressor の AST テストを削除し、それでだけ使っていた import（`ast`・`inspect`・`textwrap`・`EnsembleRegressor` など）も消す（`Objective` を使わなくなるため。ruff の F401 を通す）
  - テスト（`tests/test_ensemble.py`、`opt=False`）: 各まとめ方の予測が直接作った VotingRegressor・StackingRegressor と一致する（`random_state` を明示したモデル）、モデル1つ、名前の付け方（重なりに番号、組）、blending の `weights_` が非負、average の `weights_ is None`、属性、`results_` がない、DataFrame で学習したときの `feature_names_in_`（scikit-learn 1.0 以上）、`n_jobs` の既定と受け渡し、検査の ValueError、`boruta=` の TypeError、`clone`・`get_params` の往復、学習前の `predict` の NotFittedError
  - 完了の状態: `EnsembleRegressor(estimators=[Ridge(), SVR()], method=m, opt=False)` が3つのまとめ方で scikit-learn 1.9 でも学習・予測でき、直接作ったアンサンブルと予測が一致し、全体のテストが通る
  - _Depends: 2_
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 4.4, 5.1, 5.2, 5.4, 5.5, 5.6, 6.1, 6.2, 6.3, 6.4, 7.4_

- [ ] 3.2 渡したモデルの引数を保ち、random_state の規則を当てる
  - `fit` で `random_state` から整数を1つ引き（`model_random_state_` に持つ）、各モデルの未指定の `random_state`（入れ子を含む、元のモデルで判定）に `apply_params` で入れた写しを使う。続けてモデルごとに調整の種を引く
  - 渡したモデルのオブジェクトは変えない
  - テスト: `RandomForestRegressor(n_jobs=1, random_state=7)` の `n_jobs`・`random_state` が `estimators_` でも同じ、`random_state=None` のモデル（入れ子の Pipeline の中を含む）に `model_random_state_` が入る、同じ `random_state` で2回学習すると同じ予測、渡したモデルの `get_params` が学習の前後で同じ、`opt=False` では規則以外の引数が渡したまま
  - 完了の状態: 上のテストが `tests/test_ensemble.py` で通る
  - _Requirements: 2.5, 3.1, 3.2, 3.3, 3.4_

- [ ] 3.3 調整の振る舞いと、調整にまつわる誤りを確かめる
  - 調整に使う import（optuna、`yikit.models._optuna`、2 のモジュール）のどれかに失敗したときは、optuna と optuna-integration を入れるか `opt=False` にするよう案内する ImportError を、元の例外につないで出す
  - `ParamDistributions` の NotImplementedError は、モデルの名前と型を加えた NotImplementedError にして、元の例外につないで出す
  - テスト（optuna-integration がなければ飛ばす。`n_trials=2`、`cv=3` 程度）: `estimators_` の各要素が `OptunaSearchCV` で、`study_` の試行が `n_trials` 個、`best_params_` の名前が `ParamDistributions` と同じ。`n_trials` の既定が 100、`scoring` と `cv` が調整に渡る。stacking で、調整付きのモデルの `fit` が cv 回（分割の学習データの大きさ）と1回（全データ）呼ばれる。optuna-integration の import を失敗させたときと `yikit.models._optuna` の import を失敗させたときの ImportError と、どちらでも `opt=False` なら学習できること。調整の種が整数として各 `OptunaSearchCV` の `random_state` に入り、同じ `random_state` で2回学習すると同じ `best_params_` になること。`KNeighborsRegressor` と `opt=True` の NotImplementedError。`verbose=0` の調整で optuna の INFO のログが出ない
  - 完了の状態: 上のテストが `tests/test_ensemble.py` で通る
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.6, 2.7, 2.8, 3.3, 5.3, 7.3_

- [ ] 4. Boruta を前段に置いた Pipeline をアンサンブルで使えることを確かめ、使い方を示す
  - boruta 0.4.3 の変換（`_transform`）は `return_df=False` のとき `X[:, mask]` を使うので、DataFrame を入れた Pipeline が失敗する（タスク1のレビューで判明。作り直した EnsembleRegressor は入力を配列に変えずに渡すので、DataFrame で使うと必ず当たる）。yikit の BorutaPy で、`return_df=False` のときは DataFrame を配列にしてから変換するよう、最小限に直す
  - テスト: `Pipeline([("boruta", BorutaPy(..., max_iter 小、max_shuf 小、verbose=0、n_jobs=1、random_state 整数)), ("ridge", Ridge())])` を3つのまとめ方で学習・予測でき（ndarray と DataFrame の両方）、`opt=True` では `best_params_` の名前が `ridge__alpha` になる（boruta と optuna-integration がなければ飛ばす）
  - EnsembleRegressor の docstring の Examples に、BorutaPy を前段に置いた Pipeline と調整の使い方を示す（doctest で通るか、重い例は `# doctest: +SKIP`）
  - 完了の状態: 上のテストと EnsembleRegressor の doctest が通る
  - _Depends: 1, 3.3_
  - _Requirements: 4.1, 7.4_

- [ ] 5. 検証
- [ ] 5.1 全体の検査を通す
  - 手元で `pytest`、`ruff check src tests`、`ruff format --check src tests`、`mypy`、変えたモジュールの doctest を通す
  - 完了の状態: すべてエラーなく終わる
  - _Requirements: 7.2_

- [ ] 5.2 作業ブランチで CI を実行し、全版で通ることを確かめる
  - 作業ブランチを push し、CI を手動で実行して Python 3.8〜3.14 の7つの版の結果を確かめる。版による失敗は原因を直す。振る舞いを変える必要があるときは作者に相談する
  - 完了の状態: 7つの版のテストがすべて成功する
  - _Requirements: 7.1_

## Implementation Notes
- 手元の検査は yikit-dev の環境の実行ファイルを絶対パスで使う（`/Users/yu9824/opt/miniforge3/envs/yikit-dev/bin/{python,pytest,ruff,mypy}`）。`ruff format` は `src tests` に限る
- scikit-learn 1.6 以上では `is_regressor(OptunaSearchCV(...))` が False で、VotingRegressor と StackingRegressor が拒む（research.md）
- boruta 0.4.3 の `fit_transform` は `self._fit` を直接呼ぶ（yikit の `fit` を通らない）
- テストの小さなデータは各テストファイルの中に置く。`tests/conftest.py` は変えない
- GBDTRegressor は gbdt-fix（PR #32）がマージされるまで LightGBM 4 で学習できないので、このテストでは使わない
- タスク1（BorutaPy の perc）: `_fit` で `perc_` を決め、`self.perc` は `_fit` の間だけ差し替えて finally で戻す。boruta の `_transform` は DataFrame に対応しない（`X[:, mask]`）ので、タスク4で直す
- タスク2: OptunaSearchRegressor は回帰モデルの印に加えて、普通のメソッドの `predict`（古い optuna の property 対策）、`best_estimator_` から読む `n_features_in_`・`feature_names_in_`、親と同じ `fit(X, y=None, groups=None, **fit_params)`、参照を数えるログの抑制を持つ。タスク3では、`opt=True` でもアンサンブルの `n_features_in_`・`feature_names_in_` をそのまま写せる
