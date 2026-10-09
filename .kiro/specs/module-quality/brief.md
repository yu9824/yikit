# Brief: module-quality

## Problem
yikit を使う人（作者、licond など）は、モデル以外のモジュール（helpers・logging・metrics・feature_selection・visualize）が今の依存の版で正しく動くのかを、テストで確かめられません。docstring も日本語と英語が混ざっていたり、なかったりします。docs は Sphinx が docstring から生成しているので、docstring が欠けると API リファレンスも欠けます。

## Current State
- テストは `test_boruta.py`、`test_filtermethod.py`、`test_visualize.py`、`test_optuna.py` の4つで、9件だけ。helpers・logging・metrics にはテストがない
- `test_visualize.py` は図を `tests/imgs/` の参照画像と比べる。新しい環境（matplotlib などの版が新しい）では3件が許容値（RMS 20）を超えて失敗する（RMS 30・22・22）
- 比べた結果、差分の画像を `tests/imgs/` に書き出して、リポジトリを汚す
- docstring は、英語の numpy 形式のもの、日本語のもの、ないものが混ざっている。テストのコメントや fixture の docstring も日本語

## Desired Outcome
- helpers・logging・metrics・feature_selection・visualize の公開 API に、英語の numpy 形式の docstring がある
- それぞれの公開 API に、主な使い方と境界の条件を確かめるテストがある
- テストは Python 3.8〜3.14 と、依存の下限に近い版でも通る

## Approach
公開 API ごとに、振る舞いのテスト（入出力、例外、乱数の再現性）を先に書き、docstring を英語の numpy 形式に揃えます。図のテストは、dev-foundation で安定させた画像の比較の仕組みに乗せます。

## Scope
- **In**: `src/yikit/helpers/`、`src/yikit/logging/`、`src/yikit/metrics/`、`src/yikit/feature_selection/`、`src/yikit/visualize/` の docstring とテスト。`tests/conftest.py` の整理
- **Out**: `src/yikit/models/` のテストと docstring（optuna-tuning、ensemble-on-sklearn、gbdt-fix）。機能の追加や振る舞いの変更（不具合が見つかったら、小さなものはここで直し、大きなものは別に切り出す）

## Boundary Candidates
- 基盤（helpers・logging）
- 指標（metrics）
- 特徴量選択（feature_selection: FilterSelector、BorutaPy）
- 可視化（visualize: yyplot、SummarizePI、学習曲線、分布の図、フォントの設定）

## Out of Boundary
- models の各クラスと `Objective`・`ParamDistributions`
- docs のサイトの構成（Sphinx の設定、手で書くページ）

## Upstream / Downstream
- **Upstream**: dev-foundation（CI、型の方針、pytest の設定、画像を比べるテストの仕組みの安定化）
- **Downstream**: release-0.4.0（examples の更新のときに、テストで確かめた使い方を参照する）

## Existing Spec Touchpoints
- **Extends**: なし
- **Adjacent**: optuna-tuning（`visualize` の optuna の学習曲線は `Objective` の study を使う。テストの fixture も共有する）

## Constraints
- 依存の下限に近い版（scikit-learn 0.24.1、古い matplotlib など）でも動くテストにする
- ngboost は Python 3.14 では入らない（pyproject の marker）。ngboost を使うテストはそれを前提に skip する
- 乱数の種は `SEED = 334` に揃える
