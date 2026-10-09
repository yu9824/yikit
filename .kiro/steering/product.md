# Product Overview

yikit は、表形式データの**回帰**を中心とした機械学習の作業を楽にするための、作者自身のツールキットです。scikit-learn の部品と組み合わせて使える estimator・transformer・関数を提供します。PyPI で公開しており（`pip install yikit`）、作者の研究コードや、それを使う別のプロジェクト（例: licond）から import されます。

## Core Capabilities

- **ハイパーパラメータ探索**: optuna で、モデルごとに決めた探索範囲を使って調整する（`Objective` で study を直接回す方法と、`ParamDistributions` を `OptunaSearchCV` に渡す方法）
- **特徴量選択**: 相関に基づくフィルタ（`FilterSelector`）と Boruta（`BorutaPy`、`perc` の自動決定つき）
- **モデル**: sklearn 互換の回帰モデルの包み（例: early stopping を自動でかける LightGBM の包み）とアンサンブル
- **可視化**: y-y プロット、permutation importance の要約、学習曲線など、回帰の結果を確かめる図
- **補助**: 評価指標、logging の設定、`tqdm` と joblib の連携など

## Target Use Cases

- 数百〜数千サンプル程度の小さな表形式データで、複数の回帰モデルを比較・調整する
- 交差検証の中で特徴量選択とハイパーパラメータ探索を行い、結果を図で確かめる
- インターネットにつながらない古い環境にも持ち込み、そのまま動かす

## Value Proposition

- **sklearn の流儀に乗る**: `fit`/`predict`/`transform`、`get_params`/`set_params`、`clone` が効くので、`Pipeline`・`cross_validate`・`StackingRegressor` などにそのまま入れられる
- **探索範囲の既定値をまとめて持つ**: よく使うモデルについて、実績のある探索範囲を1か所で管理し、`Objective` と `OptunaSearchCV` の両方から同じ範囲を使える
- **利用者の指定を尊重する**: 渡されたモデルに入っている引数（`n_jobs`、`random_state` など）を勝手に書き換えない。yikit の推奨値は、利用者が明示的に適用する形（例: `RecommendedParams`）で提供する
- **古い環境でも動く**: 1.0.0 までは依存パッケージの下限を不用意に上げない

## Product Principles

- 依存する側（licond など）は「ある版以上」を要求する。振る舞いを変えるときは minor を上げ（0.x では minor が破壊的変更の単位）、変更履歴とリリースノートに「同じ種でも結果が変わる」などの影響を書く
- sklearn や optuna に同じ機能があるなら、独自実装を持たずにそれを組み合わせる。独自に持つのは、標準の部品では表せないもの（例: 学習データから検証用を切り出す early stopping）に限る

---
_Focus on patterns and purpose, not exhaustive feature lists_
_updated_at: 2026-10-09_
