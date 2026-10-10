# Requirements Document

## Project Description (Input)
yikit の利用者が複数の回帰モデルをまとめて使うための EnsembleRegressor は、交差検証・Boruta・optuna による調整・まとめ方をすべて独自に実装しており、不具合が多く、テストもない。scikit-learn 1.4 以上では非公開の `_score` の引数の変更で `fit` が必ず失敗し、NumPy 1.24〜1.26 では `np.bool` で落ちる。

この spec では、EnsembleRegressor を scikit-learn の VotingRegressor・StackingRegressor で組み立て直す。予測は全データで学習し直したモデルで行い、調整は各モデルを `OptunaSearchCV(model, ParamDistributions(model))` で包んで行う。特徴量の選択は各モデルを Pipeline にして渡す使い方を案内する。詳細は `brief.md` を参照。

## Introduction
yikit で複数の回帰モデルをまとめて使う利用者のために、EnsembleRegressor を scikit-learn の部品（VotingRegressor・StackingRegressor）の上に作り直す。まとめ方（average・stacking・blending）と optuna による調整はそのまま使えるようにし、独自の交差検証・各分割のモデルの平均・`results_` はなくす。特徴量の選択は、yikit の `BorutaPy` を各モデルの Pipeline の前段に置いて行う。そのために、`BorutaPy` が Pipeline の中でも `perc="auto"` で動くように直す。この spec は 0.4.0 の破壊的変更（`boruta` 引数と `results_` の削除、予測の作り方の変更）を含む。

## Boundary Context
- **In scope**: EnsembleRegressor の作り直し（まとめ方、調整、渡したモデルの引数の扱い、入力の検査、学習後の属性）、yikit の `BorutaPy` が `fit_transform`（Pipeline の中）でも `perc="auto"` を使え、`fit` が引数 `perc` を書き換えないようにする最小限の修正、これらのテストと英語の docstring
- **Out of scope**: 分類のアンサンブル、Boruta の選択の手順そのもの（feature_selection の他の振る舞いは module-quality）、探索範囲と `ParamDistributions`（optuna-tuning）、各分割で作ったモデルの平均による予測、交差検証のスコアや OOF 予測の記録（利用者が `cross_validate` などで測る）、examples の書き直し（release-0.4.0）
- **Adjacent expectations**:
  - optuna-tuning の `ParamDistributions` と、`random_state` の規則（`Objective` と同じ規則）を使う。NGBRegressor の `Base__*` が `OptunaSearchCV` で効かないこと（`ParamDistributions` が警告する）は、この spec では直さない
  - `OptunaSearchCV`（optuna-integration）は optional の依存。入っていなくても、調整を使わなければ動く
  - scikit-learn 0.24.1 にある部品だけを使う（StackingRegressor は 0.22、`LinearRegression(positive=...)` は 0.24 から）

## Requirements

### Requirement 1: まとめ方
**Objective:** As yikit の利用者, I want 複数の回帰モデルを average・stacking・blending のどれかでまとめられること, so that 独自の実装に頼らず、scikit-learn の標準どおりのアンサンブルを使える

#### Acceptance Criteria
1. Where `method="average"` のとき, the EnsembleRegressor shall `opt=False` で各モデルの `random_state` を明示した場合、同じモデルで作った VotingRegressor と同じ予測を返す
2. Where `method="stacking"` のとき, the EnsembleRegressor shall 1 と同じ条件で、同じモデルと同じ `cv` で作った `StackingRegressor(final_estimator=LinearRegression())` と同じ予測を返す
3. Where `method="blending"` のとき, the EnsembleRegressor shall 1 と同じ条件で、同じモデルと同じ `cv` で作った `StackingRegressor(final_estimator=LinearRegression(positive=True, fit_intercept=False))` と同じ予測を返す（重みは非負で、和を 1 に揃えない）
4. When `predict` を呼んだとき, the EnsembleRegressor shall 学習データ全体で学習し直したモデルの予測をまとめて返す（各分割で作ったモデルの予測の平均は使わない）
5. When モデルが1つだけのとき, the EnsembleRegressor shall どのまとめ方でも学習と予測ができる
6. The EnsembleRegressor shall `estimators` として、モデルの並びと、`(名前, モデル)` の組の並びの両方を受け付ける。モデルだけの並びには、`make_pipeline` と同じ規則（型の名前の小文字、重なれば番号付き）で名前を付ける
7. The EnsembleRegressor shall `n_jobs` と `verbose` を、中の VotingRegressor・StackingRegressor に渡す（`n_jobs` の既定は今と同じ -1）

### Requirement 2: 調整
**Objective:** As yikit の利用者, I want 各モデルの引数を optuna で調整してからまとめられること, so that 範囲を自分で書かずに、調整したモデルのアンサンブルを作れる

#### Acceptance Criteria
1. Where `opt=True`（既定）のとき, the EnsembleRegressor shall 各モデルを、そのモデルの `ParamDistributions`（`n_features` は渡された X の列の数）を使う `OptunaSearchCV` で `n_trials` 試行だけ調整し、最良の引数で学習し直したモデルをまとめる
2. The EnsembleRegressor shall `n_trials` を引数に持ち、既定を 100 にする
3. The EnsembleRegressor shall 調整に、`cv` と `scoring`（既定は `"neg_mean_squared_error"`）を使う
4. While stacking と blending で、まとめ役の学習に使う予測（各分割の検証データに対する予測）を作る間, the EnsembleRegressor shall その分割の学習データだけで調整する（検証データを調整に使わない）
5. Where `opt=False` のとき, the EnsembleRegressor shall 調整をせず、渡したモデルをそのまま（3.2 の規則の `random_state` を除く）使う
6. If `opt=True` で optuna-integration が入っていないとき, the EnsembleRegressor shall `fit` で、インストールするか `opt=False` にするよう案内する ImportError を出す
7. If `opt=True` で探索範囲のないモデルが含まれるとき, the EnsembleRegressor shall `fit` で、そのモデルの名前と型を示すエラーを出す
8. While `verbose=0` で調整する間, the EnsembleRegressor shall optuna の試行ごとのログを出さず、`fit` の後に optuna のログの設定を元に戻す

### Requirement 3: 渡したモデルの引数を保つ
**Objective:** As yikit の利用者, I want 渡したモデルの引数がアンサンブルの中でもそのまま使われること, so that 並列数や乱数を自分で決められる

#### Acceptance Criteria
1. The EnsembleRegressor shall 渡したモデルの引数を、調整で探索した引数と 3.2 の規則の `random_state` 以外は変えない（`n_jobs` なども渡した値のまま使う）
2. Where 渡したモデル、またはその中の入れ子のモデルの `random_state` が None のとき, the EnsembleRegressor shall `Objective` と同じ規則で、EnsembleRegressor の `random_state` から引いた整数を入れる。明示した値はそのまま使う
3. When 同じ `random_state` で2回学習したとき, the EnsembleRegressor shall 同じ予測を返す（中のモデルが決定的に動く場合）
4. The EnsembleRegressor shall 渡したモデルのオブジェクトを変えない（学習するのは写し）

### Requirement 4: 特徴量の選択（Boruta）
**Objective:** As yikit の利用者, I want yikit の BorutaPy を各モデルの前段に置いて特徴量を選べること, so that `boruta` 引数がなくなっても、モデルごとに特徴量の選択を組み合わせられる

#### Acceptance Criteria
1. When `Pipeline([("boruta", BorutaPy(...)), (名前, モデル)])` を `estimators` に入れたとき, the EnsembleRegressor shall どのまとめ方でも、`opt=True` でも学習と予測ができる（調整の引数名には段の名前の前置きが付く）
2. When yikit の BorutaPy を `perc="auto"` で `fit_transform` したとき（Pipeline の中で使うときを含む）, the BorutaPy shall `fit` と同じように `perc` を自動で決めて特徴量を選ぶ
3. The BorutaPy shall `fit` で引数 `perc` を書き換えず、自動で決めた値を学習後の属性 `perc_` に持つ（`perc` に数値を渡したときは、その値を `perc_` に入れる）
4. The EnsembleRegressor shall `boruta` 引数を持たない（0.4.0 で削除し、渡すと TypeError になる）

### Requirement 5: 学習後の属性
**Objective:** As yikit の利用者, I want 学習したアンサンブルの中身を scikit-learn と同じ名前で調べられること, so that 各モデルや重みを他の sklearn のアンサンブルと同じように扱える

#### Acceptance Criteria
1. When `fit` が終わったとき, the EnsembleRegressor shall 学習した VotingRegressor か StackingRegressor を `estimator_` に、全データで学習し直した各モデルを `estimators_` と `named_estimators_` に持つ
2. Where stacking か blending のとき, the EnsembleRegressor shall まとめ役のモデルを `final_estimator_` に、その係数を `weights_` に持つ。Where average のとき, the EnsembleRegressor shall `weights_` を None にする
3. Where `opt=True` のとき, the EnsembleRegressor shall 各モデルの調整の結果（study と最良の引数）を、`estimators_` の各要素（`OptunaSearchCV`）から調べられるようにする
4. The EnsembleRegressor shall `n_features_in_` を持ち、列名のある入力で学習したときは `feature_names_in_` も持つ（その属性を持つ版の scikit-learn のとき）
5. The EnsembleRegressor shall `results_` を持たない（0.4.0 で削除）
6. If 学習する前に `predict` を呼んだとき, the EnsembleRegressor shall NotFittedError を出す

### Requirement 6: 入力の検査と sklearn の決まり
**Objective:** As yikit の利用者, I want 誤った使い方がすぐ分かり、sklearn の道具とそのまま組み合わせられること, so that 原因の分からない失敗に悩まされない

#### Acceptance Criteria
1. If `estimators` に回帰モデルでないものが含まれるとき, the EnsembleRegressor shall `fit` で、その名前と型を示す ValueError を出す
2. If `estimators` が空のとき, the EnsembleRegressor shall `fit` で ValueError を出す
3. If `method` が average・stacking・blending のどれでもないとき, the EnsembleRegressor shall `fit` で、使える値を示す ValueError を出す
4. The EnsembleRegressor shall `__init__` では引数を同じ名前の属性に入れるだけにし、`get_params`・`set_params`・`clone` で引数が保たれる

### Requirement 7: 互換性、テスト、docstring
**Objective:** As yikit の作者, I want 作り直したアンサンブルがテストで確かめられ、英語の docstring で説明されていること, so that 古い環境でも動くことを CI で確かめられ、使い方を docs で調べられる

#### Acceptance Criteria
1. When CI が Python 3.8〜3.14 の各版でテストを実行したとき, the yikit のテスト shall すべて通る
2. The yikit のテスト shall 1〜6 の受け入れ基準を確かめる。少なくとも、各まとめ方の予測が直接作った scikit-learn のアンサンブルと一致すること、少ない試行での調整、BorutaPy を前段に置いた Pipeline、渡した引数と `random_state` の規則、入力の誤り、`verbose=0` で optuna のログが出ないことを含む
3. Where optional の依存（optuna-integration、boruta、lightgbm など）が入っていない環境のとき, the yikit のテスト shall その依存を使うテストを飛ばし、残りを実行する
4. The yikit パッケージ shall EnsembleRegressor と、変えた BorutaPy の振る舞いに英語の numpy 形式の docstring を持ち、EnsembleRegressor の Examples に BorutaPy を前段に置いた Pipeline と調整の使い方を示す
