# Technology Stack

## Architecture

sklearn 互換の部品を集めた Python ライブラリです。サーバーや CLI はありません。

- **sklearn の estimator API に合わせる**: 独自のモデルや選択器は `BaseEstimator` と `RegressorMixin`・`SelectorMixin` などを継承し、`Pipeline` や `cross_validate` の中で使えることを前提にする
- **重い依存は optional**: optuna・lightgbm・ngboost・Boruta・tqdm などは `[optional]` extra に置き、各サブパッケージの `__init__.py` で `is_installed()` を確かめてから import する。入っていなくても `import yikit` は失敗しない

## Core Technologies

- **Language**: Python 3.8 以上（`requires-python = ">= 3.8"`）。CI で 3.8〜3.14 を確かめている
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
- 実行時に評価される場所（`isinstance`、型の別名、関数の外の式、`cast`）では PEP 585/604 の書き方を使わない（3.8 対策）
- `cast` の型は文字列で書く（`cast("X | Y", value)`）。新しい版の numpy などにしかない型は `TYPE_CHECKING` の中で import する
- mypy で検査する（`mypy.ini`: 対象は `src/yikit`、`ignore_missing_imports = True`）。mypy 2.4 は 3.10 未満を対象にできないので、3.8 で動くことは ruff の `target-version = "py38"` と CI の 3.8 のテストで保証する

### sklearn の estimator の決まり
- `__init__` では引数を同じ名前の属性に入れるだけにする。検証や加工は `fit` で行う
- `**kwargs` は使わない（`get_params` に出ず、`clone` で消える）
- 学習で決まる属性は末尾に `_` を付ける（`estimator_`、`n_features_in_` など）。`check_X_y`・`check_array`・`check_is_fitted` を使う
- 乱数は `fit` の中で `check_random_state(self.random_state)` から作る
- **利用者が渡したモデルの引数を勝手に上書きしない**。探索で変えるのは探索する引数だけにし、それ以外は渡されたモデルの値のまま使う

### コードの品質
- ruff（`ruff.toml`）: 対象は Python 3.8、規則は `E4・E7・E9・F・W・I・UP・FA` に固定（ruff の版で既定の規則が変わるため）、すべてのモジュールに `from __future__ import annotations` を必須にする、行の長さ 79、`examples/`・`docs_src/` は除外。ruff 0.16 の `ruff format .` は Markdown も書き換えるので、パスを `src tests` に限って実行する
- docstring は英語・numpy 形式で書く（Sphinx の napoleon で API リファレンスを生成するため）
- エラーを握りつぶさず、意味のあるメッセージで送出する

### テスト
- pytest。`tests/conftest.py` に共通のデータ（`make_regression`、442 サンプル × 10 特徴量、seed 334）を置く
- 乱数の種は `SEED = 334` に揃える
- 図のテストは、モデルの学習に頼らない固定の入力で図を作り、文字・要素の数・描いたデータ・軸の範囲を値で確かめてから、`tests/conftest.py` の `assert_figure_matches_reference` で画像を比べる。比べる前に文字と凡例を消し、`font.size` を固定して layout をやり直すので、フォントや依存の版に左右されない（許容値 RMS 1.0、CI の全版で揺れは最大 0.037）。参照画像は `pytest tests/test_visualize.py --save-reference-figures` で作り直す。差分の画像は一時ディレクトリに出る
- CI（GitHub Actions、`ubuntu-24.04` に固定）は Python 3.8〜3.14 の各版で `pip install ".[test,optional]"` の後に `pytest -ra` を回す。ある版が失敗しても他の版は止めない。setup-python は 3.8・3.9 を ubuntu-26.04 向けに配っていないので、runner を固定している
- ruff と mypy は手元で実行して通す。CI では走らせない

## Development Environment

### Required Tools
- conda の環境 `yikit-dev`（Python 3.12）に editable で入れる
- ruff、mypy、pytest、jupytext（examples の同期）、Sphinx（docs）

### Common Commands
```bash
# Setup: pip install -e ".[test,optional,dev]"   # dev に ruff・mypy が入っている
# Test:  python -m pytest
# Lint:  ruff check src tests && ruff format --check src tests
# Type:  mypy
# Examples: jupytext --sync examples/**/*.ipynb
# Docs:  sphinx-apidoc -f -o ./docs_src ./src/yikit --module-first && sphinx-build -b html ./docs_src ./docs
```

## Release

- 版番号は `src/yikit/__init__.py` の `__version__` だけに書く（pyproject は `dynamic`）。pre-release は `0.4.0-rc.1` の形
- タグ `vX.Y.Z` または `vX.Y.Z-rc.N` を push すると、PyPI への公開と GitHub Release（自動生成のノート）が走る。docs は正式版のタグで公開される
- PyPI では一度公開した版番号を、削除しても使い直せない。失敗した rc は消さずに次の rc を出す
- ビルドは `setuptools>=61` と表の形の `license` を使う（Python 3.8 でソースからビルドするため。3.8 で入る setuptools は PEP 639 の書き方を受け付けない）。新しい setuptools は表の形を 2027-02-18 の期限つきで非推奨にしているので、期限の前に（遅くとも 1.0.0 で 3.8 を外すときに）PEP 639 の書き方へ戻す

## Key Technical Decisions

- **src レイアウト**（`src/yikit/`）: テストが、インストールしたパッケージに対して走るようにするため
- **optional extra に重い依存を寄せる**: 古い環境や軽い用途で、必要なものだけ入れられるようにするため
- **探索範囲は1か所で持つ**: `Objective` と `ParamDistributions` に同じ範囲を2回書くと、ずれが生じるため
- **固定値を黙って入れない**: モデルの値から、利用者が明示したのか既定値のままなのかは判定できない。推奨値は `RecommendedParams` で明示的に適用してもらう

---
_Document standards and patterns, not every dependency_
_updated_at: 2026-10-09_
