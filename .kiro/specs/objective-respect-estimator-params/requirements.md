# Requirements Document

## Project Description (Input)
yikit の optuna 用の目的関数 `Objective`（src/yikit/models/_optuna.py）を、渡したモデルの引数を勝手に書き換えないように直してください。0.4.0 の挙動の変更として出します。

### 背景
- 利用側（licond）は、モデルの並列数と乱数を自分で決めて Objective に渡したい。今の Objective は次のように上書きするため、利用側の指定が効かない。
  - LGBMRegressor・GBDTRegressor・RandomForestRegressor の固定値に "n_jobs": -1 を入れ、探索の各試行と get_best_params の結果に入れる。
  - LightGBM・ランダムフォレスト・LinearModelRegressor・MLPRegressor・NGBRegressor の固定値に "random_state": self.rng（RandomState のオブジェクト）を入れる。NGBRegressor では Base の DecisionTreeRegressor にも同じオブジェクトを入れる。get_best_params の結果にもこのオブジェクトが入り、最後のモデルと入れ子のモデルが1つの RandomState を共有する。

### 直すこと
1. n_jobs: 固定値に "n_jobs" を入れない。渡したモデル（clone(self.estimator)）の n_jobs をそのまま使う。get_best_params の結果にも n_jobs を入れない。
2. random_state: モデルで明示されている値（None 以外）はそのまま使う。None のときだけ、Objective の乱数から引いた整数を入れる。
   - 整数は __init__ で、TPESampler の種（今の `self.rng.randint(2**31 - 1)`）を引いた後に1回だけ引き、全試行と get_best_params で同じ値を使う（試行どうしを同じ乱数で比べられるように）。
   - NGBRegressor の Base（DecisionTreeRegressor）も同じ規則にする: 渡したモデルの Base の random_state が None ならその整数、明示されていればその値。
   - RandomState のオブジェクトを、最後のモデルと入れ子のモデルで共有させない。
3. get_best_params は「探索した引数」と「固定値（上の規則の random_state と、利用者が fixed_params で渡したもの）」だけを返す。get_best_estimator は clone(self.estimator).set_params(**best_params) で作り、探索の対象でない引数を渡したモデルの値のまま残す。
4. 利用者が fixed_params で渡した値は、今までどおり最優先で使う。

### 変えないこと
- 各モデルの探索の範囲、試行の評価（cross_validate とその n_jobs=self.n_jobs）、TPESampler の種の引き方。
- n_jobs・random_state 以外の固定値（例 SVR の gamma・kernel）。

### 検査を加えること
- n_jobs=2 の LGBMRegressor と RandomForestRegressor: 試行のモデル、get_best_params、get_best_estimator のどれでも n_jobs が 2 のまま。
- random_state=7 のモデル: 試行と最後のモデルで 7 のまま。
- random_state=None のモデル: 試行と get_best_params の random_state が整数で、同じ Objective の random_state なら同じ値になり、study の結果も同じになる。
- NGBRegressor（Base の random_state が None と明示の両方）: Base の random_state が上の規則に従い、最後のモデルと Base が同じ RandomState のオブジェクトを持たない。
- fixed_params={"n_jobs": 3, "random_state": 5} を渡すと、その値が使われる。

### 出すとき
- 変更履歴とリリースノートに、0.4.0 で n_jobs を上書きしなくなったこと、random_state の扱いが変わったこと（同じ種でも結果が変わる）を書く。
- 出した版の番号を licond 側に伝える（licond はその版以上を要求する）。

### 開発の進め方
- ブランチを切って開発する。ruff・mypy を通す。
- 環境の作成など、何かをインストールするときは承認が必要。

## Requirements
<!-- Will be generated in /kiro-spec-requirements phase -->
