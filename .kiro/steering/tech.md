# Technology Stack

## Architecture

sklearn 互換の部品を集めた Python ライブラリです。サーバーや CLI はありません。

- **sklearn の estimator API に合わせる**: 独自のモデルや選択器は `BaseEstimator` と `RegressorMixin`・`SelectorMixin` などを継承し、`Pipeline` や `cross_validate` の中で使えることを前提にする
- **重い依存は optional**: optuna・lightgbm・ngboost・Boruta・tqdm などは `[optional]` extra に置き、各サブパッケージの `__init__.py` で `is_installed()` を確かめてから import する。入っていなくても `import yikit` は失敗しない

## Core Technologies

- **Language**: Python。現在の `requires-python` は `>=3.9`。3.8 でも動く書き方をし、CI で確かめられれば 3.8 に戻す
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
- mypy で検査する（`mypy.ini`: `ignore_missing_imports = True`）

### sklearn の estimator の決まり
- `__init__` では引数を同じ名前の属性に入れるだけにする。検証や加工は `fit` で行う
- `**kwargs` は使わない（`get_params` に出ず、`clone` で消える）
- 学習で決まる属性は末尾に `_` を付ける（`estimator_`、`n_features_in_` など）。`check_X_y`・`check_array`・`check_is_fitted` を使う
- 乱数は `fit` の中で `check_random_state(self.random_state)` から作る
- **利用者が渡したモデルの引数を勝手に上書きしない**。探索で変えるのは探索する引数だけにし、それ以外は渡されたモデルの値のまま使う

### コードの品質
- ruff（`ruff.toml`: 行の長さ 79、docstring は numpy 形式）
- docstring は英語・numpy 形式で書く（Sphinx の napoleon で API リファレンスを生成するため）
- エラーを握りつぶさず、意味のあるメッセージで送出する

### テスト
- pytest。`tests/conftest.py` に共通のデータ（`make_regression`、442 サンプル × 10 特徴量、seed 334）を置く
- 乱数の種は `SEED = 334` に揃える
- 図は `matplotlib.testing.compare.compare_images` で `tests/imgs/` の参照画像と比べる（許容値あり）。差分の画像は一時ディレクトリに出し、リポジトリを汚さない
- CI（GitHub Actions）は Python の各版で `pip install ".[test,optional]"` の後に pytest を回す
- ruff と mypy は手元で実行して通す。CI では走らせない

## Development Environment

### Required Tools
- conda の環境 `yikit-dev`（Python 3.12）に editable で入れる
- ruff、mypy、pytest、jupytext（examples の同期）、Sphinx（docs）

### Common Commands
```bash
# Setup: pip install -e ".[test,optional,dev]" mypy ruff
# Test:  python -m pytest tests
# Lint:  ruff check . && ruff format --check .
# Type:  mypy src
# Examples: jupytext --sync examples/**/*.ipynb
# Docs:  sphinx-apidoc -f -o ./docs_src ./src/yikit --module-first && sphinx-build -b html ./docs_src ./docs
```

## Release

- 版番号は `src/yikit/__init__.py` の `__version__` だけに書く（pyproject は `dynamic`）。pre-release は `0.4.0-rc.1` の形
- タグ `vX.Y.Z` または `vX.Y.Z-rc.N` を push すると、PyPI への公開と GitHub Release（自動生成のノート）が走る。docs は正式版のタグで公開される
- PyPI では一度公開した版番号を、削除しても使い直せない。失敗した rc は消さずに次の rc を出す

## Key Technical Decisions

- **src レイアウト**（`src/yikit/`）: テストが、インストールしたパッケージに対して走るようにするため
- **optional extra に重い依存を寄せる**: 古い環境や軽い用途で、必要なものだけ入れられるようにするため
- **探索範囲は1か所で持つ**: `Objective` と `ParamDistributions` に同じ範囲を2回書くと、ずれが生じるため
- **固定値を黙って入れない**: モデルの値から、利用者が明示したのか既定値のままなのかは判定できない。推奨値は `RecommendedParams` で明示的に適用してもらう

---
_Document standards and patterns, not every dependency_
_updated_at: 2026-10-09_
