# Technology Stack

## Architecture

sklearn 互換の部品を集めた Python ライブラリです。サーバーや CLI はありません。

- **sklearn の estimator API に合わせる**: 独自のモデルや選択器は `BaseEstimator` と `RegressorMixin`・`SelectorMixin` などを継承し、`Pipeline` や `cross_validate` の中で使えることを前提にする
- **重い依存は optional**: optuna・lightgbm・ngboost・Boruta・tqdm などは `[optional]` extra に置き、各サブパッケージの `__init__.py` で `is_installed()` を確かめてから import する。入っていなくても `import yikit` は失敗しない

## Core Technologies

- **Language**: Python 3.10 以上（`requires-python = ">=3.10"`）。CI で 3.10〜3.14 を確かめている。Python 3.8・3.9 では pip が 0.4.x の系列を選ぶ
- **Framework**: scikit-learn（下限 0.24.1）
- **Core deps**: numpy、pandas、scipy、joblib、matplotlib、seaborn

## Key Libraries

- **optuna**（>=3.0）: `IntDistribution`・`FloatDistribution` を使うため
- **optuna-integration**: `OptunaSearchCV`。新しい optuna では `optuna.integration.sklearn` が非推奨なので、`optuna_integration` から import し、古い optuna には元の場所を使う
- **lightgbm**: 3.x と 4.x の両方で動かす。4.0 で fit の `early_stopping_rounds`・`verbose` と、`silent` が削除されているので、callback（`lightgbm.early_stopping`）を使う
- **Boruta**（>=0.4.3）: 0.3 は `np.int` などを使っていて NumPy 1.24 以上で動かないため

## Development Standards

### 互換性
- **依存の下限は 1.0.0 まで上げない**。例外は、古い下限が実際に壊れていて、新しい下限が他の依存の下限を上げない場合だけ（例: Boruta 0.4.3）
- 使う API は下限の版にあるものに限る。版で分ける必要があるときは `sys.version_info` や `is_installed()` で分岐する
- dependabot は `versioning-strategy: increase-if-necessary` にし、下限を上げるだけの PR が来ないようにする

### 型
- `from __future__ import annotations` を使い、注釈には `X | None`、`dict[str, Any]` などを書く。`typing.Optional`・`Union`・`List` などの古い書き方は使わない
- 新しい版の numpy などにしかない型は `TYPE_CHECKING` の中で import する。その型を `cast` に使うときは、型を文字列で書く（`cast("NDArray[Any]", value)`）
- mypy で検査する（`mypy.ini`: 対象は `src/yikit`、`python_version = 3.10`、`ignore_missing_imports = True`）

### sklearn の estimator の決まり
- `__init__` では引数を同じ名前の属性に入れるだけにする。検証や加工は `fit` で行う
- `**kwargs` は使わない（`get_params` に出ず、`clone` で消える）
- 学習で決まる属性は末尾に `_` を付ける（`estimator_`、`n_features_in_` など）。`check_X_y`・`check_array`・`check_is_fitted` を使う
- 乱数は `fit` の中で `check_random_state(self.random_state)` から作る
- **利用者が渡したモデルの引数を勝手に上書きしない**。探索で変えるのは探索する引数だけにし、それ以外は渡されたモデルの値のまま使う

### コードの品質
- ruff（`ruff.toml`）: 対象は Python 3.10（`target-version = "py310"`）、規則は `E4・E7・E9・F・W・I・UP・FA` に固定（ruff の版で既定の規則が変わるため）、すべてのモジュールに `from __future__ import annotations` を必須にする、行の長さ 79、`examples/`・`docs_src/` は除外。ruff 0.16 の `ruff format .` は Markdown も書き換えるので、パスを `src tests` に限って実行する
- docstring は英語・numpy 形式で書く（Sphinx の napoleon で API リファレンスを生成するため）
- エラーを握りつぶさず、意味のあるメッセージで送出する

### テスト
- pytest。`tests/conftest.py` に共通のデータ（`make_regression`、442 サンプル × 10 特徴量、seed 334）を置く
- 乱数の種は `SEED = 334` に揃える
- 図のテストは、モデルの学習に頼らない固定の入力で図を作り、文字・要素の数・描いたデータ・軸の範囲を値で確かめてから、`tests/conftest.py` の `assert_figure_matches_reference` で画像を比べる。比べる前に文字と凡例を消し、`font.size` を固定して layout をやり直すので、フォントや依存の版に左右されない（許容値 RMS 1.0、CI の全版で揺れは最大 0.037）。参照画像は `pytest tests/test_visualize.py --save-reference-figures` で作り直す。差分の画像は一時ディレクトリに出る
- CI（GitHub Actions の `CI.yml`、`ubuntu-latest`、権限は読み取りだけ）は 2 種類のテストの job を回す
  - `test`: Python 3.10〜3.14 の各版で `pip install ".[test,optional]"` の後に `pytest -ra` を回す。ある版が失敗しても他の版は止めない
  - `test-no-optional`: matrix を持たない別の job。Python 3.14 で `pip install ".[test]"` だけを入れて `pytest -ra` を回し、optional の依存がないときの動作と pandas 3 を確かめる
  - optional の依存は、テストの関数の中で `pytest.importorskip` してから使う。ファイル全体が optional の依存に頼るときは、module の先頭で `importorskip` し、その依存とそれに頼る yikit の import をその後に置く（`# noqa: E402`）。module の先頭で直接 import すると、`test-no-optional` が collection で失敗する
  - 変更の判定は `changes` job（`dorny/paths-filter`）で行い、コードに関わるファイルが変わっていなければテストの job を `if` で飛ばす。`on.<event>.paths` で workflow ごと飛ばすと必須の checks が待ちのまま残るが、`if` で飛ばした job は成功として扱われる。手動実行では判定せずに必ず回す
  - 必須の checks を設定するときは、matrix でない job（`changes`・`test-no-optional`）を選ぶか、結果をまとめる job を足す。matrix の `test` を飛ばすと、`test (3.12)` のような版ごとの check ではなく、`test` という名前の check だけが出る
  - GitHub Actions の部品の更新は、dependabot が 1 つの PR にまとめて出す
- ruff と mypy は手元で実行して通す。CI では走らせない

## Development Environment

### Required Tools
- conda の環境 `yikit-dev`（Python 3.12）に editable で入れる
- ruff、mypy、pytest、build、jupytext（examples の同期）、Sphinx（docs）

### 版番号と editable
- editable で入れると、setuptools-scm が git の tag から版を決めて `src/yikit/_version.py` に書く（git では追跡しない）。入れるときは build isolation を使う。setuptools-scm は pip の一時的なビルド用の環境に入り、`yikit-dev` には入らない
- `_version.py` は、setuptools がこのツリーで版を決めるとき（editable の入れ直し、`python -m build`、`pip install .`）に書き直され、それまでは前の版のまま。tag を付けた後や、版を確かめたいときは `pip install -e . --no-deps` で入れ直す
- `_version.py` がない（一度も入れていない・ビルドしていない）ソースでは `yikit.__version__` が `"unknown"` になる。追跡しているファイルにコミットしていない変更があると、版の末尾に `.d<日付>` が付く

### Common Commands
```bash
# Setup: pip install -e ".[test,optional,dev]"   # dev に ruff・mypy・build が入っている
# Reinstall: pip install -e . --no-deps          # _version.py の版を今の tag とコミットに合わせる
# Test:  python -m pytest
# Lint:  ruff check src tests && ruff format --check src tests
# Type:  mypy
# Build: python -m build                          # dist/ に sdist と wheel を作る
# Examples: jupytext --sync examples/**/*.ipynb
# Docs:  sphinx-apidoc -f -o ./docs_src ./src/yikit --module-first && sphinx-build -b html ./docs_src ./docs
```

## Release

- 版番号は setuptools-scm が git の tag だけから決める（`vX.Y.Z` → `X.Y.Z`、`vX.Y.Z-rc.N` → `X.Y.ZrcN`）。ソースに版番号を書かないので、版を上げる PR は要らない。tag のないコミットは、最後の tag の次の開発版になる（`v0.5.0` の後なら `0.5.1.devN+g<hash>`）。CI のテストの job は tag のない浅い clone なので `0.1.dev1+g<hash>` のような版になるが、テストは版の値に依存しない
- `vX.Y.Z` か `vX.Y.Z-rc.N` の tag を push すると、release workflow が PyPI と TestPyPI に公開し、GitHub Release（自動生成のノート）を作る。tag に `-` があれば pre-release にする。docs は正式版の tag で公開される。release と docs の checkout は、tag の ref ではその tag だけを深さ 1 で取るので、同じコミットに他の tag があってもその tag の版になる
- branch で release を手動実行すると、全履歴から開発版をローカル版（`+g<hash>`）なしで作り、TestPyPI にだけ出す。同じ版がすでにあれば飛ばす（`skip-existing`）。追跡しているファイルが変わったままだとビルドは失敗する
- rc と正式版の tag を同じコミットに付けるときは、正式版も annotated の tag にし、rc の tag より後に作る。annotated の rc と lightweight の正式版が同じコミットにあると、全履歴のビルド（branch での手動実行や手元の editable）で rc の tag が選ばれ、`v0.5.0` のコミットが `0.5.0rc1`、その後のコミットが `0.5.0rc2.devN` になる。docs を手動実行するときは、branch ではなく tag の ref を選ぶ
- PyPI では一度公開した版番号を、削除しても使い直せない。失敗した rc は消さずに次の rc を出す
- ビルドの要件は `setuptools>=77.0.3` と `setuptools-scm>=10.1`。ライセンスは PEP 639 の形（`license = "Apache-2.0"` と `license-files`）で書く
- sdist には setuptools-scm が git で追跡しているファイルをすべて入れ、`MANIFEST.in` が開発のためだけのファイル（`.kiro/`・`.claude/`・`.github/`・`.vscode/`、`CLAUDE.md`、最上位の dotfile、`ruff.toml` などの設定）を外す。最上位にファイルやディレクトリを足したときは `MANIFEST.in` を見直す。`global-exclude` は使わない（`src/yikit` の中のファイルを外すと wheel からも消える）
- release の workflow のファイル名 `release-pypi.yml` と環境の名前 `pypi`・`testpypi` は、PyPI と TestPyPI の trusted publishing の登録に使われているので変えない（ファイル名は README のバッジも使う）

## Key Technical Decisions

- **src レイアウト**（`src/yikit/`）: テストが、インストールしたパッケージに対して走るようにするため
- **optional extra に重い依存を寄せる**: 古い環境や軽い用途で、必要なものだけ入れられるようにするため
- **探索範囲は1か所で持つ**: `Objective` と `ParamDistributions` に同じ範囲を2回書くと、ずれが生じるため
- **固定値を黙って入れない**: モデルの値から、利用者が明示したのか既定値のままなのかは判定できない。推奨値は `RecommendedParams` で明示的に適用してもらう

---
_Document standards and patterns, not every dependency_
_updated_at: 2026-10-15_
