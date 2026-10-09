# Requirements Document

## Project Description (Input)
下流の licond は、モデルの `n_jobs` と `random_state` を自分で決めて `Objective` に渡したいが、今の `Objective` はそれらを上書きし、`get_best_params` の結果にも RandomState のオブジェクトを入れる。また、探索範囲が `Objective` と `ParamDistributions` の2か所に書かれてずれ始めており、`Pipeline` や `TransformedTargetRegressor` に包んだモデルは探索できない。そのため、引数名を平らにするだけの包み（LinearModelRegressor、SupportVectorRegressor）が必要になっていた。

この spec では、探索範囲を1つのモジュールにまとめて `Objective` と `ParamDistributions` の両方で使い、入れ子を解いて前置きを付ける。PLS・LinearSVR・Ridge・Lasso・ElasticNet の探索範囲を加え、推奨の固定値は `RecommendedParams` で明示的に適用する形にする。`Objective` は固定値を暗黙に入れず、渡したモデルの引数を保ち（`random_state` が None のときだけ整数を入れる）、利用者の `fixed_params` を最優先にする。LinearModelRegressor と SupportVectorRegressor は削除する。詳細は `brief.md` を参照。

## Introduction
yikit で optuna を使ってモデルの引数を探索する利用者（licond など）のために、探索が「探索する引数」以外を変えないようにし、`Objective` と `OptunaSearchCV` のどちらで探索しても同じ範囲になるようにする。sklearn の `Pipeline` と `TransformedTargetRegressor` に包んだモデルもそのまま探索できるようにし、それまで必要だった包みのクラスを削除する。この spec は 0.4.0 の破壊的変更（同じ種でも結果が変わる、固定値を入れなくなる、包みのクラスと `ParamDistributions` の一部の引数を削除する）を含む。

## Boundary Context
- **In scope**: 登録済みのモデルの探索範囲（既存のものと、PLSRegression・LinearSVR・Ridge・Lasso・ElasticNet の追加）、入れ子（`Pipeline`・`TransformedTargetRegressor`）の解決、`custom_params` の扱い、`Objective` の引数の扱い（固定値、`random_state` の規則、`fixed_params`、`get_best_params`・`get_best_estimator`、学習に失敗した試行の扱い）、`ParamDistributions`（`n_features` の追加、`fixed_params`・`random_state` の削除）、`RecommendedParams`、LinearModelRegressor・SupportVectorRegressor の削除、EnsembleRegressor が動き続けるための最小限の変更、これらのテストと英語の docstring
- **Out of scope**: 試行の評価の方法（交差検証の分け方、scoring、交差検証の `n_jobs`）と TPESampler の種の引き方の変更、0.4.0-rc.0 から引き継ぐモデルの探索範囲の変更（scikit-learn の版で使えない値への対応を除く）、`Pipeline`・`TransformedTargetRegressor` 以外の meta-estimator（MultiOutputRegressor、BaggingRegressor、Voting・Stacking など）の入れ子の解決、EnsembleRegressor の作り直し（ensemble-on-sklearn）、GBDTRegressor 自体の修正（gbdt-fix）、examples の更新と CHANGELOG・リリースノート（release-0.4.0）、分類のモデル
- **Adjacent expectations**:
  - dev-foundation が用意した optuna>=3.0、`optuna_integration` からの import、Python 3.8〜3.14 の CI の上で動く
  - `OptunaSearchCV` は、学習に失敗した試行を失敗として記録し、探索を続ける（optuna-integration の既定の振る舞い）。`ParamDistributions` に `n_features` を渡さないときは、これに頼る
  - ensemble-on-sklearn は `ParamDistributions` と入れ子の解決を使う。gbdt-fix が GBDTRegressor の引数を変えるときは、この spec の探索範囲も合わせる
  - licond への版番号の連絡は release-0.4.0 で行う

## Requirements

### Requirement 1: Objective が渡されたモデルの引数を保つ
**Objective:** As licond の開発者, I want `Objective` に渡したモデルの `n_jobs` などの引数が探索で変わらないこと, so that 並列数や乱数を自分で決めて探索できる

#### Acceptance Criteria
1. When `Objective` が試行を評価するとき, the Objective shall 渡されたモデルの写しに、その試行で探索した引数、`random_state` の規則（Requirement 2）で決めた値、利用者の `fixed_params` だけを設定したモデルを評価する
2. The Objective shall 渡されたモデルの `n_jobs` を、試行のモデル、`get_best_params` の結果、`get_best_estimator` のモデルのどれでも変えない（利用者が `fixed_params` で指定した場合を除く）
3. The Objective shall 利用者が指定していない固定値（SVR の `gamma`・`kernel`、LightGBM の `objective` など）をモデルに設定しない
4. When 利用者が `fixed_params` を渡したとき, the Objective shall その値を、探索した値と `random_state` の規則の値より優先して、試行のモデル、`get_best_params` の結果、`get_best_estimator` のモデルに使う
5. When `fixed_params` に探索範囲の引数が含まれるとき, the Objective shall その引数を探索せず、固定値を使う
6. When `get_best_params` を呼んだとき, the Objective shall 最良の試行で探索した引数、`random_state` の規則で入れた値、利用者の `fixed_params` だけを返す
7. When `get_best_estimator` を呼んだとき, the Objective shall 渡されたモデルの写しに `get_best_params` の結果を設定した、学習前のモデルを返し、探索しない引数は渡されたモデルの値のまま残す
8. If 試行のモデルの学習が交差検証のいずれかの分割で失敗したとき, the Objective shall その試行を失敗として記録して探索を続け、その試行を最良の試行に選ばない

### Requirement 2: random_state の規則
**Objective:** As licond の開発者, I want 乱数を明示したモデルはその値のまま、明示しないモデルは再現できる整数で探索されること, so that 同じ種から同じ結果を得られ、試行どうしを同じ乱数で比べられる

#### Acceptance Criteria
1. Where 渡されたモデル、またはその中の入れ子のモデル（NGBRegressor の Base、Pipeline の各段など）が `random_state` を引数に持ち、その値が None 以外のとき, the Objective shall その値を変えずに使う
2. Where 渡されたモデル、またはその中の入れ子のモデルが `random_state` を引数に持ち、その値が None のとき, the Objective shall `Objective` の `random_state` から引いた整数をその引数に入れる
3. The Objective shall 2.2 の整数を `Objective` を作るときに、TPESampler の種を引いた後で1回だけ引き、すべての試行と `get_best_params` で同じ値を使う
4. When 同じ `random_state` を渡した `Objective` を2つ作り、同じデータと同じ回数で探索したとき, the Objective shall 同じ整数を入れ、同じ最良の引数と最良の値を得る
5. The Objective shall 1つの RandomState のオブジェクトを、最後のモデルと入れ子のモデルなど、複数のモデルに共有させない
6. Where モデルが `random_state` を引数に持たないとき（SVR、PLSRegression など）, the Objective shall そのモデルに `random_state` を設定しない

### Requirement 3: 登録済みのモデルの探索範囲
**Objective:** As yikit の利用者, I want よく使う回帰モデルに既定の探索範囲が用意されていること, so that 範囲を自分で書かずに探索を始められる

#### Acceptance Criteria
1. The yikit パッケージ shall LGBMRegressor・GBDTRegressor、RandomForestRegressor、SVR、LinearSVR、MLPRegressor、NGBRegressor、PLSRegression、Ridge、Lasso、ElasticNet（それぞれの派生クラスを含む）に探索範囲を持つ
2. The yikit パッケージ shall 0.4.0-rc.0 から引き継ぐモデル（LGBMRegressor・GBDTRegressor、RandomForestRegressor、SVR、MLPRegressor、NGBRegressor）の探索する引数と範囲を変えない（3.7 の場合を除く）
3. The yikit パッケージ shall PLSRegression の `n_components` を、1 から min(10, 特徴量の数) までの整数で探索する。`Objective` では、渡された X の列の数を特徴量の数とする
4. The yikit パッケージ shall LinearSVR の `C` と `epsilon` を、SVR と同じ範囲で探索する
5. The yikit パッケージ shall Ridge と Lasso の `alpha` を 0.1〜10 の対数の範囲で、ElasticNet の `alpha` を同じ範囲で、`l1_ratio` を 0.1〜1.0 の範囲で探索する
6. The yikit パッケージ shall 追加した線形のモデル（PLSRegression・LinearSVR・Ridge・Lasso・ElasticNet）の `max_iter`・`tol`・`fit_intercept` を探索しない
7. Where インストールされた scikit-learn の版が、探索範囲のある値を受け付けない、または非推奨にしているとき（NGBRegressor の Base の `criterion` の `"friedman_mse"` など）, the yikit パッケージ shall その版で警告なしに使える値だけを探索する

### Requirement 4: 入れ子の解決
**Objective:** As yikit の利用者, I want `Pipeline` や `TransformedTargetRegressor` に包んだモデルもそのまま探索できること, so that 前処理や目的変数の変換を、包みのクラスなしで組み合わせられる

#### Acceptance Criteria
1. When 渡された estimator が `Pipeline` のとき, the yikit パッケージ shall 最後の段のモデルの探索範囲を使い、引数名に段の名前の前置き（例: `svr__C`）を付ける
2. When 渡された estimator が `TransformedTargetRegressor` のとき, the yikit パッケージ shall `regressor` のモデルの探索範囲を使い、引数名に `regressor__` の前置きを付ける
3. When 入れ子が重なっているとき（`TransformedTargetRegressor` の中の `Pipeline` など）, the yikit パッケージ shall 内側までたどり、前置きをつなげる（例: `regressor__svr__C`）
4. The yikit パッケージ shall `Pipeline` の最後の段以外の段（前処理、特徴量の選択など）と、`TransformedTargetRegressor` の `transformer` の引数を探索しない
5. The yikit パッケージ shall 前置きを付けた引数名を、`Objective` が試行のモデルに設定するときと、`ParamDistributions` を `OptunaSearchCV` に渡したときの両方で、そのまま使えるようにする
6. When 渡された estimator が入れ子になっているとき, the yikit パッケージ shall データで決まる範囲（3.3）の特徴量の数に、包みの外側に入る X の列の数を使う

### Requirement 5: 利用者の探索範囲（custom_params）
**Objective:** As yikit の利用者, I want 自分の探索範囲を渡したら、それが必ず使われること, so that 範囲を調整でき、渡した範囲が黙って無視されることがない

#### Acceptance Criteria
1. When `custom_params` が空でない探索範囲を返すとき, the Objective and ParamDistributions shall 登録済みのモデルでも、yikit の探索範囲の代わりにそれを使う
2. The Objective and ParamDistributions shall `custom_params` の引数名に前置きを付けず、そのまま使う
3. The Objective and ParamDistributions shall `custom_params` として、今と同じ形（`Objective` では試行を受け取って値の辞書を返す関数、`ParamDistributions` では分布の辞書か、それを返す関数）を受け付ける
4. If 入れ子をたどった先のモデルに探索範囲がなく、`custom_params` も空のとき, the Objective and ParamDistributions shall そのモデルの型の名前と、`custom_params` を渡すことを示すエラー（NotImplementedError）を出す

### Requirement 6: ParamDistributions
**Objective:** As `OptunaSearchCV` を使う利用者, I want `Objective` と同じ探索範囲を `OptunaSearchCV` に渡せること, so that どちらの方式で探索しても範囲がずれない

#### Acceptance Criteria
1. The ParamDistributions shall 同じ estimator と同じ特徴量の数に対して、`Objective` と同じ引数名と同じ範囲の分布を持つ
2. The ParamDistributions shall `OptunaSearchCV` の `param_distributions` にそのまま渡せる辞書として振る舞う
3. When `n_features` を渡したとき, the ParamDistributions shall それを特徴量の数として、データで決まる範囲（3.3）に使う
4. If `n_features` を渡さずに、データで決まる範囲を持つモデルを探索するとき, the ParamDistributions shall その上限を 10 にする。特徴量の数を超えた試行は `OptunaSearchCV` の中で失敗として記録され、探索は続く
5. The ParamDistributions shall `fixed_params` と `random_state` の引数を持たない（0.4.0 で削除し、渡すと TypeError になる）

### Requirement 7: RecommendedParams
**Objective:** As yikit の利用者, I want yikit の推奨する固定値を、自分で選んだときだけ適用できること, so that 推奨値が黙って入ることなく、必要なときに1行で使える

#### Acceptance Criteria
1. When `RecommendedParams(estimator)` を作ったとき, the RecommendedParams shall そのモデルに対する yikit の推奨の固定値を、引数名から値への辞書として持つ
2. The RecommendedParams shall 入れ子を Requirement 4 と同じ規則で解き、同じ前置きを付ける
3. The RecommendedParams shall `Objective` の `fixed_params` と、estimator の `set_params` の両方にそのまま渡せる
4. Where 推奨の固定値がないモデルのとき（探索範囲のない型を含む）, the RecommendedParams shall エラーにせず、空の辞書になる
5. The RecommendedParams shall 最初の版では、SVR に対する `gamma="auto"` だけを持つ
6. The Objective and ParamDistributions shall RecommendedParams の値を、利用者が渡さない限り使わない

### Requirement 8: 包みのクラスの削除
**Objective:** As yikit の作者, I want 入れ子の解決で置き換えられる包みのクラスをなくすこと, so that 保守するコードを減らし、sklearn の標準の部品で組み立ててもらえる

#### Acceptance Criteria
1. The yikit.models shall LinearModelRegressor と SupportVectorRegressor を提供しない（import すると Python の通常の ImportError になる）
2. The yikit パッケージ shall 代わりの書き方（Ridge・Lasso・ElasticNet、`TransformedTargetRegressor` と `Pipeline` と SVR の組み合わせ）を、`Objective` と `ParamDistributions` の docstring の例で示す

### Requirement 9: EnsembleRegressor が動き続けること
**Objective:** As EnsembleRegressor の利用者, I want ensemble-on-sklearn で作り直すまで、探索ありの EnsembleRegressor が動き続けること, so that この spec の変更で既存の使い方が壊れない

#### Acceptance Criteria
1. While EnsembleRegressor が ensemble-on-sklearn で作り直されていない間, when `opt=True` で `fit` したとき, the EnsembleRegressor shall 各モデルを探索し、渡したモデルの探索しない引数を保ったまま、最良の引数で学習して予測できる
2. The EnsembleRegressor shall この spec では、探索した結果からモデルを作る部分だけを変える

### Requirement 10: 検証と docstring
**Objective:** As yikit の作者と licond の開発者, I want この spec の振る舞いがテストで確かめられ、英語の docstring で説明されていること, so that 変更が壊れていないことを CI で確かめられ、使い方を docs で調べられる

#### Acceptance Criteria
1. The yikit のテスト shall Requirement 1 と 2 を、次の場合で確かめる: `n_jobs=2` の LGBMRegressor と RandomForestRegressor、`random_state=7` のモデル、`random_state=None` のモデル、NGBRegressor（Base の `random_state` が None の場合と明示した場合）、`fixed_params={"n_jobs": 3, "random_state": 5}`、学習に失敗する試行を含む探索
2. The yikit のテスト shall 登録済みのすべてのモデルで、`Objective` と `ParamDistributions` の探索範囲が一致することを確かめる
3. The yikit のテスト shall 入れ子（`Pipeline`、`TransformedTargetRegressor`、その組み合わせ）の前置きと、追加したモデル（PLSRegression・LinearSVR・Ridge・Lasso・ElasticNet）を `Objective` と `OptunaSearchCV` で実際に探索できることを確かめる
4. Where optional の依存（lightgbm、ngboost など）が入っていない環境のとき, the yikit のテスト shall その依存を使うテストを飛ばし、残りのテストを実行する
5. When CI が Python 3.8〜3.14 の各版でテストを実行したとき, the yikit のテスト shall すべて通る
6. The yikit パッケージ shall `Objective`、`ParamDistributions`、`RecommendedParams` と、この spec で作る公開の関数に、英語の numpy 形式の docstring を持つ
