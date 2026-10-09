# Project Structure

## Organization Philosophy

機能の分野ごとにサブパッケージを分けます（`models`、`feature_selection`、`visualize`、`metrics` など）。各サブパッケージは、公開する名前を `__init__.py` に集め、実装は先頭に `_` を付けた非公開のモジュールに置きます。利用者は `from yikit.models import Objective` のように、サブパッケージから import します。

## Directory Patterns

### 機能のサブパッケージ
**Location**: `src/yikit/<domain>/`
**Purpose**: 1つの分野の公開 API。`__init__.py` で `__all__` を定義し、実装のモジュールから名前を取り込む
**Example**: `src/yikit/models/__init__.py` が `from ._ensemble import EnsembleRegressor` で取り込み、`__all__` に足す

### 非公開の実装モジュール
**Location**: `src/yikit/<domain>/_<topic>.py`
**Purpose**: 1つの話題の実装（1つのクラスとその補助、または関連する関数群）
**Example**: `models/_gbdt.py`（LightGBM の包み）、`visualize/_yyplot.py`（y-y プロット）

### optional な依存を使う機能
**Location**: 実装は通常どおり `_<topic>.py`。`__init__.py` 側で条件付きで公開する
**Purpose**: 依存が入っていない環境でも `import yikit` を壊さない
**Example**:
```python
if is_installed("optuna"):
    from ._optuna import Objective, ParamDistributions

    __all__ += ["Objective", "ParamDistributions"]
```
実装のモジュールの中で別の optional な依存を使うときも、`is_installed()` で分岐し、なければ `NoneType` などの代わりを入れて `isinstance` が失敗しないようにする

### 基盤のサブパッケージ
**Location**: `src/yikit/helpers/`、`src/yikit/logging/`
**Purpose**: 他のサブパッケージから使われる共通の道具（`is_installed`、`tqdm_joblib`、ロガーの取得など）。これらは他の機能のサブパッケージに依存しない

### テスト
**Location**: `tests/test_<topic>.py`、共通の fixture は `tests/conftest.py`、参照画像は `tests/imgs/`
**Purpose**: 公開 API の振る舞いの検査。ファイルは機能の単位で分ける
**Example**: `tests/test_optuna.py`、`tests/test_boruta.py`

### examples
**Location**: `examples/`（分野ごとにサブディレクトリ。例: `examples/feature_selection/`）
**Purpose**: 使い方を示す notebook。jupytext で `.ipynb` と `.py`（percent 形式）を対にして管理する（`jupytext.toml`）
**Example**: `examples/feature_selection/filter_method.ipynb` と `filter_method.py`

### docs
**Location**: `docs_src/`
**Purpose**: Sphinx の設定と、手で書くページ。API リファレンスは CI で `sphinx-apidoc` が docstring から生成するので、モジュールごとのページは手で書かない

## Naming Conventions

- **Files**: snake_case。サブパッケージの中の実装は `_` で始める（`_filter.py`、`_optuna.py`）
- **Classes**: PascalCase。sklearn に合わせて役割を末尾に付ける（`...Regressor`、`...Selector`）
- **Functions**: snake_case（`get_learning_curve_optuna`、`root_mean_squared_error`）
- **Fitted attributes**: 末尾に `_`（sklearn の決まり）
- **Constants**: UPPER_SNAKE_CASE（テストの `SEED` など）

## Import Organization

```python
# 標準ライブラリ → サードパーティ → yikit の順に、空行で分ける
import sys

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from yikit.helpers import is_installed          # 別のサブパッケージや実装モジュールは絶対 import
from yikit.models._gbdt import GBDTRegressor
```
- `__init__.py` で同じサブパッケージの実装を取り込むときは相対 import（`from ._ensemble import ...`）
- それ以外の場所では `yikit.` から始まる絶対 import を使う
- パスの別名はない

## Code Organization Principles

- サブパッケージ間の依存は一方向にし、循環させない。今の向きは次のとおり
  - `helpers`・`logging`: 他に依存しない（基盤）
  - `metrics`・`feature_selection`: 基盤だけに依存する
  - `visualize`: `metrics` と基盤に依存する
  - `models`: `feature_selection`・`metrics` と基盤に依存する（例: アンサンブルが Boruta を使う）
- 公開する名前は `__all__` に載せたものだけ。`_` で始まるモジュールの中身は内部用で、版をまたいで変わりうる
- 複数のモデルにまたがる設定（探索範囲、推奨値など）は1つのモジュールにまとめ、それを使う側（`Objective`、`ParamDistributions` など）は同じ定義を参照する

---
_Document patterns, not file trees. New files following patterns shouldn't require updates_
_updated_at: 2026-10-09_
