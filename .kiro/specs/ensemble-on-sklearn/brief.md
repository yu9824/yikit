# Brief: ensemble-on-sklearn

## Problem
yikit の利用者が複数の回帰モデルをまとめて使いたいとき、今の EnsembleRegressor は交差検証、Boruta、optuna による調整、まとめ方をすべて独自に実装しています。そのため不具合が多く、テストもありません。sklearn にはアンサンブルの部品（VotingRegressor、StackingRegressor）があり、yikit の探索範囲（`ParamDistributions`）や Boruta と組み合わせれば、ほぼ同じことができます。licond はこのクラスを使っていません。

## Current State
- `src/yikit/models/_ensemble.py` の `EnsembleRegressor`。引数は `estimators`、`method`（blending・average・stacking）、`cv`、`n_jobs`、`random_state`、`scoring`、`verbose`、`boruta`、`opt`
- `fit` は外側の交差検証の分割ごとに、Boruta で特徴量を選び（`boruta=True`）、`Objective` で各モデルを 100 試行で調整し（`opt=True`）、学習・予測・スコア・重要度を `results_` に記録する。`predict` は、各分割で作ったモデルの予測を平均してから、まとめ方を通す
- blending は、学習に使っていないデータでの予測（OOF 予測）に対する重みを optuna で決める。stacking は `LinearRegression` を使う
- 不具合: `np.bool` を使っていて NumPy 1.24〜1.26 で落ちる（2.0 以降は再び存在する。`wip/0.4.0-draft` で修正済み）。非公開の `sklearn.model_selection._validation._score` を使っていて、scikit-learn 1.4 以上では `score_params` の引数が足りず、`fit` が必ず失敗する（2026-10-09 に scikit-learn 1.9.1 で確認）
- 最後のモデルを `objective.model(**fixed_params_, **best_params)` で作り直していたので、渡したモデルの引数が消え、NGBRegressor では失敗した。optuna-tuning で `objective.get_best_estimator(study)` に直す（optuna-tuning が直すのはこの部分だけ）
- テストはない

## Desired Outcome
- EnsembleRegressor が sklearn の部品で組み立てられている
  - `average` → `VotingRegressor`
  - `stacking` → `StackingRegressor(final_estimator=LinearRegression())`
  - `blending` → `StackingRegressor(final_estimator=LinearRegression(positive=True, fit_intercept=False))`
- 予測は sklearn の標準どおり、全データで学習し直したモデルで行う（各分割のモデルの平均はやめる）
- 特徴量選択は、各モデルを `Pipeline([BorutaPy(...), model])` にして渡す使い方を案内する
- 渡したモデルの引数（`n_jobs`、`random_state` など）がそのまま使われる
- 主な使い方と、まとめ方ごとの結果をテストで確かめる。docstring は英語の numpy 形式

## Approach
EnsembleRegressor を、引数から sklearn の VotingRegressor・StackingRegressor を組み立てて委ねる薄い sklearn 互換のクラスにします。調整は、各モデルを `OptunaSearchCV(model, ParamDistributions(model))` で包む形で行います（`Objective` は estimator ではないので使わない）。

## Scope
- **In**: `src/yikit/models/_ensemble.py` の作り直し、テスト、docstring、公開する名前の調整
- **Out**: 探索範囲と `ParamDistributions`（optuna-tuning）。各分割のモデルの平均による予測。examples の書き直し（release-0.4.0）

## Boundary Candidates
- 引数からアンサンブル（Voting・Stacking）を組み立てる部分
- 調整（`OptunaSearchCV` で包む）を加えるかどうかの部分
- 学習後に公開する属性（重み、各モデル、スコアなど）

## Out of Boundary
- Boruta そのものの振る舞い（feature_selection）
- 探索範囲と入れ子の解決（optuna-tuning）

## Upstream / Downstream
- **Upstream**: optuna-tuning（`ParamDistributions`、入れ子の解決）、dev-foundation（`OptunaSearchCV` の import、CI）、feature_selection の `BorutaPy`（sklearn の Pipeline に入れられる transformer として）
- **Downstream**: release-0.4.0（examples の simulate を書き直す）

## Existing Spec Touchpoints
- **Extends**: なし
- **Adjacent**: optuna-tuning（その spec の間は、EnsembleRegressor を壊さない最小限の変更だけが入る）

## Constraints
- scikit-learn 0.24.1 にある API だけを使う（StackingRegressor は 0.22、`LinearRegression(positive=...)` は 0.24 から）。Python 3.8 でも動く書き方
- `optuna-integration` は optional。入っていないときは、調整を使わない形で動く
- 要件を作るときに決めること:
  - 引数の形: `boruta`・`opt` を残すか、`n_trials` を出すか
  - blending の重み: 非負の最小二乗にし、和は1に揃えない案。今の `weights_`（和が1）との違いをどう扱うか
  - `results_` に残すもの: 各モデル、OOF 予測、交差検証のスコアなどに絞る案
  - クラスとして残すか、組み立てる関数にするか
