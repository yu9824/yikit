# Brief: packaging-and-ci

## Problem
yikit 0.4.x は、Python 3.8 のオフライン環境のために古いビルドの書き方を続けています。

- `setuptools>=61` を使い、`license` を表の形で書いている（3.8 で入る setuptools が PEP 639 の書き方を受け付けないため）。新しい setuptools は表の形を非推奨にしていて、2027-02-18 に期限が切れる
- 版番号は `src/yikit/__init__.py` に手で書いている。そのため、リリースのたびに版を上げる PR が要る（0.4.0 では #35 と #36）。手順を誤ると、tag と版がずれる
- Python 3.8 と 3.9 はサポートが終わっている。それでも CI は 3.8〜3.14 の7つの版を回し、3.8・3.9 のために runner を `ubuntu-24.04` に固定している

作者は 0.5.0 で Python の下限を 3.10 に上げ、ビルドと CI を python-template（https://github.com/yu9824/python-template）にそろえたいと考えています。

## Current State
- **`pyproject.toml`**
  - `requires = ["setuptools>=61"]`、`license = { text = "Apache-2.0" }`、`requires-python = ">= 3.8"`
  - classifiers は 3.8〜3.14
  - 依存に `typing_extensions >= 4.5; python_version < '3.10'` がある
  - 版は `version = { attr = "yikit.__version__" }` から読む
  - `[project.urls]` はすでにある
- **CI（`.github/workflows/CI.yml`）**
  - テストを飛ばす判定を `on.push.paths` と `on.pull_request.paths` で行っている
  - Python 3.8〜3.14 で `pip install ".[test,optional]"` の後に `pytest -ra` を回す
  - `actions/checkout@v4` と `actions/setup-python@v5` を使い、`permissions` を設定していない
- **release（`.github/workflows/release-pypi.yml`）**
  - tag `v*` の push で走る。build の後、PyPI、TestPyPI、Sigstore で署名した GitHub Release に出す（tag に `-` を含むときは pre-release）
  - `concurrency` がなく、checkout は浅い clone のまま
- **docs（`.github/workflows/docs.yml`）**: 正式版の tag で走る。`fetch-depth: 0` で clone する。`docs_src/conf.py` は `yikit.__version__` を読む
- **dependabot**: pip だけで、`versioning-strategy: increase-if-necessary`。GitHub Actions の更新は来ない
- **ruff と mypy**: `ruff.toml` の `target-version` は `py38`。`mypy.ini` は `python_version` を指定していない
- **3.8・3.9 向けの書き分け**: `helpers/_helpers.py` と `visualize/_yyplot.py` に `sys.version_info >= (3, 9)` の分岐がある
- **steering**: `tech.md` の Core Technologies・型・Release の節が、Python 3.8 を前提に書かれている

## Desired Outcome
- **Python 3.10 以上を求める**: `requires-python = ">=3.10"`、classifiers は 3.10〜3.14 にする。`typing_extensions` の依存はなくす
- **PEP 639**: `license = "Apache-2.0"` と `license-files` を使い、ビルドに `setuptools>=77.0.3` を求める
- **版は git の tag から決める（setuptools-scm）**
  - tag `vX.Y.Z` は `X.Y.Z` として、tag `vX.Y.Z-rc.N` は `X.Y.ZrcN` として PyPI に出る
  - `yikit.__version__` はこれまでどおり使え、git の情報がない場所でも import が失敗しない
  - 生成されるファイルは git に入れない
  - リリースは tag の push だけで済み、版を上げる PR が要らない
- **CI**
  - Python 3.10〜3.14 で pytest を回す
  - テストを飛ばす判定は job の中で行う。飛ばしたときも成功として報告され、必須のチェックが待ちのまま残らない
  - `permissions` を最小にし、actions を新しい版に上げる
- **release**
  - 実行中のリリースが取り消されない（`concurrency`）
  - tag から版が正しく決まる checkout にする
  - 手動実行（`workflow_dispatch`）で作った版で TestPyPI への公開が失敗しない
- **docs**: 正式版の tag で、正しい版番号の付いた docs ができる
- **dependabot**: GitHub Actions も対象にし、更新を1つの PR にまとめる
- **ruff と mypy**: 対象を Python 3.10 にし、両方が通る。3.8・3.9 向けの書き分けは消す。振る舞いは変えない
- **steering と README**: steering（`tech.md`）の Python・ビルド・型・Release の節を 0.5.0 に合わせる。Python の版の記述が README にあれば直す

## Approach
python-template の書き方に合わせる。

- `pyproject.toml`（build-system・license・setuptools-scm の設定）
- `__init__.py` で版を読む方法と、取れないときの代わりの値
- CI の `dorny/paths-filter`
- release の `concurrency` と、公開するかどうかの判定
- dependabot の github-actions の group

ただし yikit の事情を確かめてから入れます。python-template の build job は `fetch-depth: 0` を指定していないので、その状態で yikit の tag から正しい版が出るかを確かめます。rc の tag の形が `-rc.N` であることや、TestPyPI がローカル版（`+g<hash>`）を受け付けないことも確かめます。確かめるには、手元で sdist と wheel を作って版を見ます。

## Scope
- **In**
  - `pyproject.toml` の build-system、`requires-python`、classifiers、license、版の決め方。`typing_extensions` の依存を消す
  - `src/yikit/__init__.py` の版の読み方、`.gitignore`
  - `.github/workflows/`（CI・release・docs）と `.github/dependabot.yml`
  - `ruff.toml` と `mypy.ini` の対象の版。それに合わせた、Python 3.8・3.9 向けの書き分けの削除（ruff の `UP` で出るもの。振る舞いは変えない）
  - steering の `tech.md`（Python、ビルド、型の決まり、Release の手順）と README
- **Out**
  - ライブラリの依存の下限と上限、ngboost の marker（drop-legacy-compat が持つ）
  - ライブラリの古い版のための分岐の削除（drop-legacy-compat が持つ）
  - CHANGELOG、tag の作成とリリース（直接実装の release-0.5.0 が持つ）
  - 下限の版を入れてテストする CI の job（作らないと 2026-10-14 に決めた）
  - 0.4.x の保守用のブランチ（必要になったときに作る）

## Boundary Candidates
- **ビルドとメタデータ**: `[build-system]`、`[project]` の Python・license・classifiers、`[tool.setuptools_scm]`
- **版の出どころ**: `__init__.py` と、生成される版のファイル、`.gitignore`、docs の `conf.py` が読む版
- **workflow**: CI、release、docs、dependabot
- **lint と型の対象の版**: `ruff.toml`、`mypy.ini` と、それによるコードの機械的な書き換え
- **steering の更新**: `tech.md` と README

## Out of Boundary
- `[project.dependencies]` と `[project.optional-dependencies]` にあるライブラリの版の指定（`typing_extensions` の行を消すことだけはこちらが持つ）
- `is_installed()` や、版を読んで分ける分岐など、ライブラリの版に依存するコード
- 公開 API と計算の結果

## Upstream / Downstream
- **Upstream**
  - dev-foundation（完了）で作った CI、ruff・mypy の設定、pytest の設定
  - python-template の書き方
  - PyPI と TestPyPI の trusted publishing の設定（環境 `pypi`・`testpypi`）
- **Downstream**
  - drop-legacy-compat: CI の 3.10〜3.14 と、ruff・mypy の 3.10 の設定の上で作業する
  - release-0.5.0: tag の push で 0.5.0 を出す
  - licond: `yikit>=0.5.0` を求める場合は Python 3.10 以上が要る

## Existing Spec Touchpoints
- **Extends**: なし。dev-foundation が決めた「Python 3.8 に対応する」「`setuptools>=61` と表の形の license」を、0.5.0 で置き換える。dev-foundation の spec 自体は 0.4.0 の記録として残す
- **Adjacent**
  - drop-legacy-compat: `pyproject.toml` の依存の節を共有する。この spec は Python の版に関わる行だけを触る
  - release-0.5.0: リリースの手順を `tech.md` に書くのはこの spec、実際の tag とリリースは release-0.5.0

## Constraints
- **振る舞いを変えない**: 公開 API と計算の結果は変えない。変わるのは、入れられる Python の版と、版番号の決まり方だけ
- **インストールには作者の承認が要る**: build や setuptools-scm などを手元の環境に入れるときは、作者の承認を得る。scratchpad への `pip download` や `pip install --target` も含む。手元の環境は conda の `yikit-dev`（Python 3.12）
- **ruff と mypy は手元で実行する**: CI では pytest だけを回す
- **公開は tag の push で起きる**: 確かめるには、手元のビルドか、TestPyPI への手動実行を使う。tag の作成とリリースは作者の確認を取ってから行う
- **python-template との違いは記録する**: python-template は `requires-python >= 3.9`、yikit は 3.10。違えた点はその理由を design に書く
