# Brief: module-quality

## Problem
yikit を使う人（作者、licond など）は、モデル以外のモジュール（helpers・logging・metrics・feature_selection・visualize）が今の依存の版で正しく動くのかを、テストで確かめられません。docstring も日本語と英語が混ざっていたり、なかったりします。docs は Sphinx が docstring から生成しているので、docstring が欠けると API リファレンスも欠けます。

## Current State
（2026-10-12 時点。dev-foundation・optuna-tuning・ensemble-on-sklearn・gbdt-fix の後）
- 画像を比べるテストは dev-foundation で安定した（差分の画像も一時ディレクトリに出る）。BorutaPy の `perc="auto"` の経路と DataFrame の扱いは ensemble-on-sklearn で直し、テストもある
- helpers・logging・metrics にはテストがない。yyplot、set_font、with_custom_matplotlib_settings にも直接のテストがない。FilterSelector は iris での1件だけ
- docstring は、英語の numpy 形式のもの、形式が崩れたもの、ないもの（FilterSelector・SummarizePI・BorutaPy のクラス docstring など）が混ざっている。visualize の一部と `tests/conftest.py` に日本語のコメントや docstring が残る
- 読んで見つかった不具合（調査の結果）:
  - visualize: `get_dist_figure` がサンプル 1〜2 個で落ちる。`get_learning_curve_optuna` が失敗した試行や方向のない study で落ちる。`get_learning_curve_gb` が LightGBM の既定の評価名で落ち、フォントの設定を戻さない。`SummarizePI.get_figure` もフォントの設定を戻さず、`get_data` は None を返すだけ。`yyplot` は渡された ax の図全体に `tight_layout` をかける
  - feature_selection: FilterSelector は符号つきの相関で判定するので強い負の相関を消さず、`n_features_in_` を持たないので `get_feature_names_out` が使えず、削除が多いと再帰の上限で落ちる。BorutaPy は `__init__` で `random_state` を RandomState に変える（同じインスタンスの学習し直しが再現しない）
  - helpers: `tqdm_joblib` の進捗バーは、joblib 1.3 以降の逐次実行（`n_jobs=1`）では進まない
  - logging: `catch_default_handler` は抜けるときに、入る前の状態に関係なくハンドラを付け直す
  - `tests/test_visualize.py` は optuna と lightgbm を無条件に import するので、それらがない環境では収集で失敗する

## Desired Outcome
- helpers・logging・metrics・feature_selection・visualize の公開 API に、英語の numpy 形式の docstring がある
- それぞれの公開 API に、主な使い方と境界の条件を確かめるテストがある
- テストは Python 3.8〜3.14 で通る

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
- 下限の版での動作は CI で保証しない。ただし、下限の版にない API をテストで使わないように気をつける
- ngboost は Python 3.14 では入らない（pyproject の marker）。ngboost を使うテストはそれを前提に skip する
- 乱数の種は `SEED = 334` に揃える
