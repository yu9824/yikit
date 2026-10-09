# Brief: dev-foundation

## Problem
yikit の作者（と将来の貢献者）は、変更が型や書式、古い Python・古い依存で壊れていないかを、CI で自動的に確かめられません。古いオフライン環境でも動かしたいのに、Python 3.8 の対応は rc.0 で外してしまいました。依存の下限が壊れた版を許していたり（Boruta 0.3）、使っている API の版が下限より新しかったり（optuna 3.0 の API）と、実際に動く範囲と宣言している範囲もずれています。

## Current State
- CI（`.github/workflows/CI.yml`）は、Python 3.9〜3.14 で `pip install ".[test,optional]"` と `pytest -xs .` を回すだけ。ruff と mypy は走らない
- `ruff.toml`（行の長さ 79、numpy の docstring）と `mypy.ini`（`ignore_missing_imports` だけ）はあるが、dev の extra に ruff・mypy がない。pytest の設定（`testpaths` など）もない
- `requires-python = ">= 3.9"`。`_filter.py` の入れ子の関数が注釈に `tuple[int, int]` を使っていて、3.8 では関数を定義した時点で失敗する。`helpers` には 3.8 向けの分岐が残っている
- 依存: `Boruta>=0.3`（0.3 は NumPy 1.24 以上で動かない）、`optuna` は下限なし（`ParamDistributions` が 3.0 の `IntDistribution` などを使う）、`optuna-integration` は test の extra にしかない
- `optuna.integration.sklearn` は optuna 4.9 で非推奨になった（テストが警告を出す）
- dependabot が下限を上げるだけの PR（#25 setuptools、#26 boruta）を出してくる
- 手元の古い環境 py312 は壊れていた。新しく `yikit-dev`（Python 3.12）を作った

## Desired Outcome
- CI で pytest が Python 3.8〜3.14 で通る。ruff（check と format）と mypy は手元で実行して通す（CI では走らせない。2026-10-09 に作者が決定）
- `requires-python = ">=3.8"`。3.8 で通せなければ `>=3.9` にし、通せなかった理由を記録する
- パッケージ全体が `from __future__ import annotations` と新しい注釈の書き方（`X | None`、`dict[str, Any]`）に揃い、実行時に評価される場所は 3.8 でも動く
- mypy がパッケージ全体で通る
- 依存: `Boruta>=0.4.3`、`optuna>=3.0`。`optuna-integration` を optional の extra に入れ、Python の版によって入れられる版の違いを marker で扱う
- `OptunaSearchCV` は `optuna_integration` から import し、なければ `optuna.integration` を使う
- dependabot は `versioning-strategy: increase-if-necessary`
- dev の extra に ruff・mypy を入れ、pytest の設定を `pyproject.toml` に置く
- 画像を比べるテストが CI のすべての組み合わせで通り、差分の画像をリポジトリに残さない（main の CI は 2026-10 時点でこのテストにより失敗している）
- ruff の対象は `src/` と `tests/`（`examples/` は外す）。mypy の対象は `src/yikit`

## Approach
設定ファイルと CI を先に直し、続けてパッケージ全体を機械的に書き換えます（型の書き方、3.8 で動かない構文）。振る舞いは変えません。Python 3.8 で依存が解決できるか（scikit-learn 1.3.2、optuna-integration 4.1〜4.5 など）を CI で確かめます。

## Scope
- **In**: `pyproject.toml`（requires-python、依存、extra、pytest・ruff・mypy の設定）、`.github/workflows/CI.yml`（lint と型のジョブ、3.8 を加えた版の一覧）、画像を比べるテストの安定化、`.github/dependabot.yml`、パッケージ全体の注釈の書き換えと 3.8 対策、mypy のエラーの解消、テストでの `OptunaSearchCV` の import の書き換え、classifiers の更新、dependabot の #25・#26 の後始末（#26 の中身は取り込み、PR は閉じる）
- **Out**: docstring の英語化とテストの追加（module-quality と各 spec）、モデルの振る舞いの変更（optuna-tuning、ensemble-on-sklearn、gbdt-fix）

## Boundary Candidates
- プロジェクトの設定と CI（`pyproject.toml`、`.github/`）
- パッケージ全体の型の書き方の書き換え（振る舞いを変えない機械的な変更）

## Out of Boundary
- 公開 API の追加や変更
- 新しいテストの追加（既存のテストを 3.8・新しい import に合わせて動かし、画像を比べるテストを安定させるところまで）
- docs の仕組み（Sphinx の設定や docs のワークフロー）

## Upstream / Downstream
- **Upstream**: steering の tech.md（型の決まり、依存の下限の方針）
- **Downstream**: module-quality、optuna-tuning、ensemble-on-sklearn、gbdt-fix は、この spec で整えた CI と型の方針の上で進める

## Existing Spec Touchpoints
- **Extends**: なし
- **Adjacent**: module-quality（同じモジュールの docstring を書く。こちらは注釈だけを変える）、optuna-tuning（`_optuna.py` を大きく書き換えるので、こちらでは機械的な変更にとどめる）

## Constraints
- 依存の下限は、上の2つ（Boruta、optuna）の他は上げない。下限の組み合わせでの動作は CI で保証しない（個人開発のライブラリなので、Python 3.8〜3.14 の各版で入る依存で通れば足りる。2026-10-09 に作者が決定）
- 新しい mypy は、検査の対象を Python 3.8 にできないことがある。3.8 で動くことは CI の pytest で保証し、mypy は使える版を対象にする
- GitHub Actions の ubuntu-24.04 には Python 3.8.18 がある
- 3.8 では、pip が古い依存を選ぶ（scikit-learn 1.3.2 など）。その版の組み合わせでテストが通るか確かめる
- インストールを伴う手元での作業は、作者の承認を得る
