# Implementation Plan

- [x] 1. 探索範囲の表と入れ子の解決
- [x] 1.1 探索範囲の表と、モデルの型との照合を作る
  - 登録済みの 10 種類のモデル（LightGBM と GBDTRegressor は同じ行）の探索範囲を、design の表のとおりに optuna の分布で持つ。0.4.0-rc.0 から引き継ぐ範囲の値は変えない
  - 上から順に型で照合し、最初に当たった行を使う。Lasso を ElasticNet より前に置き、派生クラスにも同じ行が当たるようにする
  - lightgbm・ngboost の行は、入っているときだけ表に入れる
  - NGBRegressor の Base の `criterion` の候補を、インストールされた scikit-learn の版で使える値（1.0 未満・1.9 未満・それ以上の3通り）にする。版の判定に新しい依存を使わない
  - PLSRegression の `n_components` の上限を、特徴量の数が渡されたときだけ `min(10, 特徴量の数)` に切り詰める
  - 表に `n_jobs`・`random_state` などの固定値を入れない
  - 英語の numpy 形式の docstring を付ける
  - テスト: 各モデルの引数名と分布、Lasso に `l1_ratio` がないこと、派生クラスの照合、PLS の上限の切り詰め、`criterion` の各候補で DecisionTreeRegressor が FutureWarning なしで学習できること、各行の下限・上限・各選択肢で小さなデータを学習できること（モデルはコンストラクタに値を渡して組み立てる。NGBRegressor の行は `Base` の DecisionTreeRegressor を手で組み立てて渡す。GBDTRegressor は学習させない。lightgbm・ngboost の行は入っていなければ飛ばす）
  - 完了の状態: 探索範囲を問い合わせると、登録済みのモデル（派生クラスを含む）には design の表の分布が、それ以外には「なし」が返り、`tests/test_search_space.py` のこれらのテストが通る
  - _Requirements: 1.3, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 10.4, 10.6_

- [x] 1.2 入れ子の解決と前置き、推奨値の表を作る
  - `Pipeline` は最後の段、`TransformedTargetRegressor` は `regressor` を再帰的にたどり、引数名に前置き（`svr__`、`regressor__`、`regressor__svr__` など）を付ける。前段と `transformer` はたどらない
  - 探索範囲の問い合わせが、入れ子を解いた前置き付きの辞書を毎回新しく作って返すようにする
  - 推奨値の表を作り、最初は SVR の `gamma="auto"` だけを持つ。問い合わせは入れ子を同じ規則で解き、当たらなければ空の辞書を返す
  - ngboost の NGBRegressor のように、自身の `set_params` が入れ子の名前を受け付けないモデルを判定できるようにする
  - 英語の numpy 形式の docstring を付ける
  - テスト: `Pipeline`・`make_pipeline`・`TransformedTargetRegressor`・その組み合わせの前置き、前段の引数が含まれないこと、`TransformedTargetRegressor(regressor=None)` が「なし」になること、推奨値（SVR、入れ子の SVR、それ以外は空）、NGBRegressor の判定
  - 完了の状態: `TransformedTargetRegressor(Pipeline([..., ("svr", SVR())]))` の探索範囲が `regressor__svr__C`・`regressor__svr__epsilon` の2つになり、推奨値が `{"regressor__svr__gamma": "auto"}` になる。`tests/test_search_space.py` が通る
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 6.6, 7.2, 7.4, 7.5, 10.6_

- [x] 2. 引数をモデルに入れる処理と、未指定の random_state の判定
- [x] 2.1 (P) 前置き付きの引数を、元を変えずにモデルの写しへ入れる処理を作る
  - 入れ子の名前は中のモデルの写しに再帰的に入れ、コンストラクタの引数はコンストラクタから作り直して入れ、`Pipeline` の段の名前は `set_params` で入れる
  - どこにも当てはまらない名前は、モデルの型と名前を示す ValueError にする
  - 英語の numpy 形式の docstring を付ける
  - テスト（`tests/test_params.py`）: 入れ子の名前（`Pipeline`、`TransformedTargetRegressor`、その組み合わせ）が入り元のモデルが変わらないこと、存在しない名前の ValueError、ngboost で `Base__max_depth` と `random_state=7` が効き `minibatch_frac=0.5` で学習できること、返すモデルと Base が元のモデルと RandomState を共有しないこと。ngboost のテストは入っていなければ飛ばす
  - 完了の状態: ngboost の NGBRegressor に `{"Base__max_depth": 5, "random_state": 7}` を入れたモデルが学習でき、元のモデルは変わらず、`tests/test_params.py` が通る
  - _Boundary: EstimatorParams_
  - _Requirements: 1.1, 1.7, 2.5, 4.5, 10.4, 10.6_

- [x] 2.2 未指定の random_state の名前を見つける処理を作る
  - 入れ子をすべてたどり、`random_state` の値が None か NumPy の大域の RandomState（ngboost が None を変換したもの）である引数の名前を、前置き付きで返す。`get_params(deep=True)` が入れ子を出さないモデル（ngboost の Base）も自分でたどる
  - 英語の numpy 形式の docstring を付ける
  - テスト（`tests/test_params.py`）: None、明示した値、`random_state` を持たないモデル（SVR、PLSRegression）、`Pipeline` の各段、ngboost の大域の RandomState と Base
  - 完了の状態: `NGBRegressor()` に対して `random_state` と `Base__random_state` の2つが返り（並びは問わない。テストは集合で比べる）、`random_state=7` と明示したモデルでは返らず、`tests/test_params.py` が通る
  - _Boundary: EstimatorParams_
  - _Requirements: 2.1, 2.2, 2.6, 10.4, 10.6_

- [ ] 3. 利用の窓口
- [x] 3.1 Objective を、表と引数の処理を使う形に作り直す
  - 作るとき: TPESampler の種を今と同じ順番で引いた後に、未指定の `random_state` に入れる整数を1回だけ引いて公開する。探索範囲は渡された X の列の数で作り、`fixed_params` の名前を除く
  - 範囲がなく `custom_params` もないときは、作るときに型の名前と `custom_params` の案内を含む NotImplementedError にする。`fixed_params` と未指定の `random_state` の名前は作るときに一度当てて、誤りを ValueError にする
  - 試行: `custom_params` が空でない辞書を返せばそれを、そうでなければ探索範囲の各分布を対応する suggest に読み替えた値を使う。`{未指定の random_state の値, 探索した値, fixed_params}` の順に重ねてモデルの写しに入れ、最後の試行のモデルとして公開し、交差検証の平均を返す
  - 交差検証が ValueError を出したときは、元のメッセージを FitFailedWarning で出して nan を返す
  - `n_jobs` や推奨値など、規則以外の固定値を入れる処理を残さない。`model`・`fixed_params_`・`rng` の属性をなくす
  - 試行で `custom_params` が空を返し、探索範囲もないときは NotImplementedError にする
  - 最良の引数は、最良の試行の値を持つ FixedTrial で試行と同じ組み立てをもう一度行って作る（探索した値・未指定の `random_state` の値・`fixed_params` だけを含む）。最良のモデルは、渡したモデルの写しに最良の引数を入れた学習前のモデルとする。`_optuna.py` の中に、なくした属性を参照する古い実装を残さない
  - `custom_params` と `fixed_params` の既定値を None にする（今の値を渡しても同じ意味）。引数の並びと名前は変えない
  - 英語の numpy 形式の docstring を付け、Examples に Ridge・Lasso・ElasticNet と `TransformedTargetRegressor` + `Pipeline` + SVR の使い方を示す
  - 既存の `test_create_study` の期待値を、新しい振る舞いでの値に作り直す。期待値の定数はテストごとに分け、`test_optuna_search_cv` の期待値は 3.3 まで変えない
  - テスト: 同じ `random_state` の2つの Objective で整数と探索の結果が同じ、`fixed_params` の名前が試行の params に出ない、`custom_params` が RandomForest の範囲より優先される、範囲がないと NotImplementedError、学習に失敗する試行（特徴量 3 で `SelectKBest(k=2)` の後に PLS を置いた Pipeline）を含む探索が最後まで進み失敗した試行が FAIL になる
  - 完了の状態: `Objective` を `study.optimize` に渡して探索でき、上のテストと既存の `test_create_study` が `tests/test_optuna.py` で通る
  - _Depends: 1.2, 2.2_
  - _Requirements: 1.1, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.3, 2.4, 4.6, 5.1, 5.2, 5.3, 5.4, 8.2, 10.6_

- [x] 3.2 licond の場合を確かめる
  - 試行のモデル・最良の引数・最良のモデルについて、渡したモデルの引数が保たれることを確かめるテストを書き、見つかった不具合を直す
  - テスト（交差検証は固定のスコアを返す関数に差し替える）: `n_jobs=2` の LGBMRegressor と RandomForestRegressor で、試行のモデル・最良の引数（`n_jobs` を含まない）・最良のモデルの `n_jobs` が 2。`random_state=7` が残る。`random_state=None` では公開した整数が入る。NGBRegressor の Base が None なら整数、明示すればその値で、最良のモデルと Base が RandomState を共有しない。`fixed_params={"n_jobs": 3, "random_state": 5}` が使われる。`SVR(kernel="linear")` の最良のモデルで `kernel`・`gamma` が渡した値のまま。lightgbm・ngboost のテストは入っていなければ飛ばす
  - 完了の状態: licond の要件のテストがすべて `tests/test_optuna.py` で通る
  - _Requirements: 1.2, 1.3, 1.6, 1.7, 2.1, 2.2, 2.5, 10.1, 10.4_

- [x] 3.3 ParamDistributions を、表を使う形に作り直す
  - 中身を、探索範囲の問い合わせ（入れ子の解決と前置きを含む）の結果にする。`n_features` はキーワード専用で受け取り、渡されなければ上限は表の値のまま
  - `custom_params` が分布の辞書ならそれを、関数なら空の FixedTrial を渡した結果を使い、空でなければ表の代わりに使う。分布でない値を含むときは TypeError にする
  - 範囲も `custom_params` もないときは NotImplementedError にする。`fixed_params`・`random_state` の引数をなくす
  - 入れ子の名前を自身の `set_params` で受け付けないモデル（NGBRegressor）の入れ子の名前を含むときは、`OptunaSearchCV` では効かないことと `Objective` を使うことを UserWarning で伝える
  - 英語の numpy 形式の docstring を付け、Examples に `OptunaSearchCV` との組み合わせと代わりの書き方を示す
  - 既存の `test_optuna_search_cv` を、`random_state` を渡さない形に直し、期待値を作り直す
  - テスト: 登録済みのすべてのモデルで、Objective の1試行の `trial.distributions` が `ParamDistributions(est, n_features=X.shape[1])` と等しい（交差検証は差し替える）。`fixed_params`・`random_state` を渡すと TypeError、NGBRegressor で UserWarning、`custom_params` の dict と関数、TypeError
  - 完了の状態: `OptunaSearchCV(est, param_distributions=ParamDistributions(est))` がそのまま動き、上のテストと `test_optuna_search_cv` が通る
  - _Requirements: 4.6, 5.1, 5.2, 5.3, 5.4, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 8.2, 10.2, 10.6_

- [ ] 3.4 RecommendedParams を作る
  - dict を継承し、渡したモデルに対する推奨値の問い合わせの結果を詰める
  - Objective と ParamDistributions が RecommendedParams を参照しないことを保つ
  - 英語の numpy 形式の docstring を付ける
  - テスト（公開は 4 で行うので `yikit.models._optuna` から import する）: SVR・入れ子の SVR・それ以外（空の辞書）、`Objective(..., fixed_params=RecommendedParams(est))` で `gamma` が "auto" になり、渡さなければ変わらない、`est.set_params(**RecommendedParams(est))` が通る
  - 完了の状態: `RecommendedParams(SVR())` が `{"gamma": "auto"}` と等しく、上のテストが通る
  - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.6, 10.6_

- [ ] 4. 公開する名前と利用側の付け替え
  - `yikit.models` から LinearModelRegressor と SupportVectorRegressor をなくし、その2つのモジュールを削除する。optuna があるときに RecommendedParams を公開する
  - EnsembleRegressor の、探索の後のモデルを作る1か所を、Objective の最良のモデルを返す処理に置き換える。それ以外は変えない
  - テスト: `from yikit.models import LinearModelRegressor` と `SupportVectorRegressor` が ImportError、`from yikit.models import RecommendedParams` が通る
  - 完了の状態: `src/` と `tests/` に、削除した2つのクラスと、Objective の `model`・`fixed_params_`・`rng` への参照が残らず、上のテストが通る（`examples/` は release-0.4.0 で更新する）
  - 3.1 から 4 までの間は、EnsembleRegressor の1か所がなくした属性を参照したままになる（EnsembleRegressor はもともと今の scikit-learn で fit が失敗し、テストもないので、テストは落ちない）
  - _Depends: 3.1, 3.4_
  - _Requirements: 8.1, 9.1, 9.2_

- [ ] 5. 検証
- [ ] 5.1 OptunaSearchCV との組み合わせを確かめ、全体の検査を通す
  - PLSRegression・LinearSVR・Ridge・Lasso・ElasticNet と、`TransformedTargetRegressor(Pipeline(StandardScaler, SVR))` を、ParamDistributions を渡した OptunaSearchCV で数試行探索するテストを書く。`best_params_` の名前に前置きが付き、最良のモデルで予測できることを確かめる
  - 同じモデルを Objective で数試行探索できることも確かめる
  - 完了の状態: 手元で `pytest`、`ruff check src tests`、`ruff format --check src tests`、`mypy` がすべて通る
  - _Requirements: 10.3, 10.4_

- [ ] 5.2 作業ブランチで CI を実行し、全版で通ることを確かめる
  - 作業ブランチを push し、CI を手動で実行して、Python 3.8〜3.14 の7つの版の結果を確かめる
  - 版による失敗が出たら、原因（依存の版の違い、Python 3.8 の書き方など）を直す。振る舞いを変える必要があるときは作者に相談する
  - 完了の状態: 作業ブランチでの CI の実行で、7つの版のテストがすべて成功する
  - _Requirements: 10.5_

## Implementation Notes
- 手元の検査は yikit-dev の環境の実行ファイルを絶対パスで使う（`/Users/yu9824/opt/miniforge3/envs/yikit-dev/bin/{python,pytest,ruff,mypy}`）。`conda run` をシェルの変数経由で呼ぶと失敗する。`ruff format` は `src tests` に限る（Markdown も書き換えるため）
- ngboost の NGBRegressor は `set_params` が `setattr` だけで、入れ子の名前を解かず、`random_state` を RandomState に変換しない。None は NumPy の大域の RandomState に変換される（research.md）
- GBDTRegressor は LightGBM 4 で fit が失敗する（gbdt-fix で直す）。この spec のテストでは GBDTRegressor を学習させない
- EnsembleRegressor は scikit-learn 1.4 以上で fit が失敗する（非公開の `_score`。ensemble-on-sklearn で直す）。この spec のテストでは EnsembleRegressor を学習させない
- テストの小さなデータ（40〜60 サンプル、3〜5 特徴量、seed 334）は各テストファイルの中に置く。`tests/conftest.py` は変えない（module-quality と並行して触らないため）
- ngboost の `set_params` を `setattr` だけにする上書きは 0.4.0 から。0.3.x（古い環境の 0.3.6 を含む）は sklearn 標準の `set_params` で入れ子の名前も効く。判定は版ではなく `NGBRegressor.set_params is not BaseEstimator.set_params` で行う。NGBRegressor の既定の Base はモジュール大域の1つのオブジェクトなので、写していない NGBRegressor に `set_params` しない（テストでは Base を明示する）
- `apply_params` はコンストラクタから作り直すので、`clone` が引き継ぐ引数以外の状態（`_sklearn_output_config`・`_metadata_request`・`_skl_callbacks`）を作り直したモデルへ移している。sklearn の `clone` がこの一覧を増やしたら合わせる
- release-0.4.0 の CHANGELOG に書くこと（optuna-tuning で公開の振る舞いから消した・変えたもの）: Objective の `model`・`fixed_params_`・`rng` 属性。ParamDistributions の `distributions`・`rng` 属性、`fixed_params`・`random_state` 引数、`__repr__` の形。登録済みのモデルでも空でない `custom_params` が優先されること。NGBRegressor の `Base__criterion` の候補が scikit-learn の版で変わること
