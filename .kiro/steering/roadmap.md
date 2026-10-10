# Roadmap

## Overview
yikit 0.4.0 を出すための残りの作業です。0.4.0-rc.0 はすでに PyPI に公開しているので、それを消さずに 0.4.0-rc.1、0.4.0 と進めます。
中心になるのは optuna による調整の作り直しです。下流の licond は、モデルの `n_jobs` と `random_state` を自分で決めて `Objective` に渡したいと考えていますが、今の `Objective` はそれを上書きしてしまいます。あわせて、探索範囲が2か所に書かれていてずれ始めていること、sklearn の Pipeline などを探索できないことも直します。
周辺の作業として、開発環境と CI を整え、テストと英語の docstring を全体にそろえ、EnsembleRegressor を sklearn の部品で作り直し、GBDTRegressor の不具合を直します。

## Approach Decision
- **Chosen**: 案A。optuna まわり（探索範囲の共通化と、`Objective` の引数の扱い）を1つの spec `optuna-tuning` にまとめ、残りは責務ごとに spec か直接実装に分ける
- **Why**: 探索範囲の共通化と `Objective` の引数の扱いは、どちらも `Objective.__call__` と `get_best_params` の同じ範囲を書き換える。1つの spec にすれば設計もレビューも1回で済む。0.4.0 はまとめて出すので、licond 向けの修正だけを急いで先に出す理由はない
- **Rejected alternatives**:
  - 案B（探索範囲と `Objective` の修正を別の spec にする）: `Objective` の引数の扱いを2回設計することになり、境界で食い違いやすい
  - 0.3.x への backport: 同じ種で結果が変わる変更はパッチ版にふさわしくない。また、v0.3.19 からの保守ブランチに、中身の違う2系統の修正が必要になる
  - v0.4.0-rc.0 の削除: PyPI では版番号を使い直せないので、消しても rc.1 以降になる
  - EnsembleRegressor の「各分割で作ったモデルの平均で予測する」方式の維持: sklearn にない機能で、独自の実装が残る。全データで学習し直す sklearn の標準の方式に合わせる
  - LinearModelRegressor・SupportVectorRegressor を非推奨にして残す: 入れ子を解く仕組みと、Ridge・Lasso・ElasticNet・SVR の探索範囲で置き換えられるので、0.4.0 で削除する
  - 固定値の暗黙の上書きを続ける: モデルの値から、利用者が明示したのか既定値のままなのかは判定できない。推奨値は `RecommendedParams` で明示的に適用してもらう

## Scope
- **In**: 開発環境と CI、Python 3.8 への対応、依存の下限の見直し、探索範囲の共通化と追加、`Objective` の引数の扱い、LinearModelRegressor・SupportVectorRegressor の削除、EnsembleRegressor の作り直し、GBDTRegressor の修正、全体のテストと英語の docstring、examples、CHANGELOG、0.4.0-rc.1 と 0.4.0 のリリース、licond への版番号の連絡
- **Out**: 新しい種類の機能（新しい特徴量選択の手法、新しい図など）、分類タスクへの対応、1.0.0 に向けた依存の下限の引き上げ

## Constraints
- **古い環境**: 1.0.0 までは依存の下限を上げない。例外は Boruta>=0.4.3（0.3 は NumPy 1.24 以上で動かない）と optuna>=3.0（`IntDistribution` などを使うため）
- **Python**: 3.8 でも動くことを目指す。CI で通せなければ `>=3.9` に戻し、その理由を記録する
- **scikit-learn**: 下限は 0.24.1。使う API はその版にあるものに限る
- **lightgbm**: 3.x と 4.x の両方で動かす
- **振る舞いの変更**: 0.4.0 は破壊的変更を含む minor。同じ種でも結果が変わることを CHANGELOG とリリースノートに書く
- **進め方**: spec ごとに main からブランチを切る。ひとまとまりの作業が終わったら PR を出し、作者がレビューしてマージする。タグとリリースは作者の確認を取ってから行う

## Boundary Strategy
- **Why this split**:
  - `dev-foundation` は、他のすべての spec が乗る土台（CI、型の方針、依存）なので先に入れる。main の CI は画像を比べるテストで失敗しているので、CI を通すための画像テストの安定化もここで持つ
  - `module-quality` は、モデル以外のモジュール（helpers・logging・metrics・feature_selection・visualize）だけを持つ。optuna まわりと並行して進められる
  - `optuna-tuning` は「何を探索するか（探索範囲）」と「探索の結果をモデルにどう入れるか（`Objective` の引数の扱い）」を両方持つ
  - `ensemble-on-sklearn` は、探索範囲と `ParamDistributions` を使う側なので最後に置く
- **Shared seams to watch**:
  - `src/yikit/models/__init__.py`: 複数の spec が公開する名前を足したり消したりする
  - `src/yikit/models/_ensemble.py` は今の `Objective`（`objective.model`、`fixed_params_`）に依存している。`optuna-tuning` で `Objective` を変えるときは、`ensemble-on-sklearn` で作り直すまで EnsembleRegressor が壊れないようにする（例: `get_best_estimator` を使う形にだけ直す）
  - `Python 3.8` と型の方針: `dev-foundation` で全体を書き換えた後は、各 spec が steering（tech.md）の型の決まりを守る
  - テストの共通 fixture（`tests/conftest.py`）: `module-quality` と `optuna-tuning` の両方が使う。変えるときは互いの影響を確かめる

## Existing Spec Updates
- なし。初期化だけしていた `objective-respect-estimator-params` は `optuna-tuning` に吸収し、ディレクトリを削除した（背景と、加えるテストの一覧は `optuna-tuning` の brief に引き継いだ）

## Direct Implementation Candidates
- [x] gbdt-fix -- GBDTRegressor の修正。LightGBM 3 と 4 の両方で動く callback 方式の early stopping、`X_train` で学習する、`**kwargs` をやめる、LightGBM 4.7 で非推奨になった `eval_set` への対応、テストと英語の docstring。optuna-tuning の探索範囲の表の GBDTRegressor の行の引数で学習できることもテストする（optuna-tuning のテストは GBDTRegressor を学習させない）。`**kwargs` をやめた後も、表の5つの名前（`n_estimators`・`min_child_weight`・`colsample_bytree`・`subsample`・`num_leaves`）はコンストラクタの引数に残す（`apply_params` が名前を検査するため）。修正の範囲が1つのクラスに収まるので spec にしない。Dependencies: dev-foundation
- [ ] release-0.4.0 -- examples の更新（作り直した EnsembleRegressor と探索範囲に合わせる。`wip/0.4.0-draft` の simulate の例を参考にする）、CHANGELOG.md の作成、版番号を 0.4.0-rc.1 にしてタグ、確認の後に 0.4.0、licond への版番号の連絡。リリースの作業なので spec にしない。Dependencies: すべての spec と gbdt-fix

## Specs (dependency order)
- [x] dev-foundation -- 開発環境と CI（pytest を Python 3.8〜3.14 で。ruff・mypy は手元で実行）、依存の下限と extra の見直し、dependabot、Python 3.8 でも動く型の書き方への全体の書き換え、画像を比べるテストの安定化。Dependencies: none
- [ ] module-quality -- モデル以外のモジュール（helpers・logging・metrics・feature_selection・visualize）の英語の docstring とテスト。Dependencies: dev-foundation
- [x] optuna-tuning -- 探索範囲の共通ファイル、入れ子の解決、RecommendedParams、モデルの追加と削除、`Objective` が利用者の引数を上書きしない扱い。Dependencies: dev-foundation
- [ ] ensemble-on-sklearn -- EnsembleRegressor を sklearn の Voting・Stacking で作り直す。Dependencies: optuna-tuning

---
_updated_at: 2026-10-09_
