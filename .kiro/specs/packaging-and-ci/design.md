# Design Document: packaging-and-ci

## Overview
**Purpose**: yikit の作者が、0.5.0 から Python 3.10 以上だけを対象にし、tag を push するだけで正しい版番号の配布物を PyPI・TestPyPI・GitHub Release に出せるようにする。ビルドと workflow は作者の python-template にそろえる。
**Users**: yikit の作者（リリースとレビュー）、後に続く drop-legacy-compat と release-0.5.0、yikit を入れる利用者（licond など）。
**Impact**: 対応する Python は 3.8 以上から 3.10 以上になる。ライセンスは PEP 639 の式で宣言する。版番号は `src/yikit/__init__.py` に手で書くのをやめ、setuptools-scm が git の tag から決めて `src/yikit/_version.py` に書く。CI は Python 3.10〜3.14 を `ubuntu-latest` で回し、加えて Python 3.14 で optional の依存を入れない job を 1 つ回す。テストを飛ばす判定は job の中で行う。release は `concurrency` を持ち、手動実行で TestPyPI に開発版を出せる。sdist は `MANIFEST.in` で開発のためだけのファイルを外す。dependabot は GitHub Actions の更新を 1 つの PR にまとめ、ngboost の 0.5.11 以上を提案しない。ruff と mypy は 3.10 を対象にする。公開 API と計算の結果は変えない。

### Goals
- `requires-python = ">=3.10"`、classifiers は 3.10〜3.14、`typing_extensions` の依存なし
- `license = "Apache-2.0"`・`license-files` と `setuptools>=77.0.3` で、ライセンスの非推奨の警告なしにビルドできる
- tag `vX.Y.Z` から `X.Y.Z`、`vX.Y.Z-rc.N` から `X.Y.ZrcN` の配布物ができ、`yikit.__version__` がそれと同じ。版のファイルがなくても import できる
- CI・release・docs・dependabot が python-template の形になり、違う点はすべて理由がある
- CI に、Python 3.14 で `.[test]` だけを入れてテストする job（`test-no-optional`）があり、optional の依存がなくてもテストが読み込めて通る（作者の決定 D1、2026-10-14）
- dependabot が ngboost の 0.5.11 以上を提案しない（作者の決定 D2）
- sdist に `.kiro/`・`.claude/`・`CLAUDE.md`・`.github/`・`.vscode/` と開発のための dotfile・設定が入らず、wheel は変わらない（作者の決定 D3）
- 手元の ruff と mypy が 3.10 を対象にエラー 0 件で通る

### Non-Goals
- ライブラリの依存の下限・上限と、ライブラリの古い版のための分岐の削除（drop-legacy-compat）
- ngboost 0.5.11 の `set_params`・`get_params` の変更への対応（作者の決定で後の版に送る。この spec は dependabot の提案を止めるだけ）
- CHANGELOG、tag の作成、0.5.0 のリリース（release-0.5.0）
- 下限の版で試す CI の job、CI での ruff・mypy の実行
- PyPI・TestPyPI の trusted publishing の登録の変更、ブランチの保護や必須の checks の設定

## Boundary Commitments

### This Spec Owns
- `pyproject.toml` のうち、`[build-system]`、`[project]` の `dynamic`・`requires-python`・`license`・`license-files`・`classifiers`、`dependencies` の `typing_extensions` の行の削除、`[tool.setuptools.dynamic]` の削除、`[tool.setuptools_scm]`
- 版番号の出どころ: `src/yikit/__init__.py` の `__version__` の読み方、生成される `src/yikit/_version.py`（追跡しない）、`.gitignore` のその行
- 版番号のテスト（`tests/test_version.py`）
- sdist の中身: 新しい `MANIFEST.in`
- workflow: `.github/workflows/CI.yml`（optional の依存なしの `test-no-optional` job を含む）、`.github/workflows/release-pypi.yml`、`.github/workflows/docs.yml` と、`.github/dependabot.yml`（ngboost の `ignore` を含む）
- optional の依存がない環境でテストの module が読み込めること: `tests/test_gbdt.py`、`tests/test_optuna.py`、`tests/test_search_space.py` の module の先頭の optuna の import だけ（`pytest.importorskip` にする）
- 手元の検査の対象の版: `ruff.toml` の `target-version`、`mypy.ini` の `python_version`
- Python 3.8・3.9 向けの書き分けの削除: `src/yikit/helpers/_helpers.py`、`src/yikit/visualize/_yyplot.py`、`src/yikit/visualize/_utils.py` の import と注記
- steering の `tech.md` のうち、Core Technologies の Language の行、「型」、「コードの品質」、「テスト」の CI の項目、「Development Environment」、「Release」の節。README の Python の版の記述の確認

### Out of Boundary
- `pyproject.toml` の `[project] dependencies` と `[project.optional-dependencies] optional` のライブラリの行（`typing_extensions` の行を除く）と、ngboost の上限と marker（drop-legacy-compat）。`test`・`dev`・`docs` の extra は、この spec も drop-legacy-compat も変えない
- ngboost 0.5.11 の `set_params`・`get_params` の変更への対応（作者の決定で後の版に送る。上限 `<0.5.11` は drop-legacy-compat が決める）
- ライブラリの版による分岐と、その注記・テスト（`NDArray` の `TYPE_CHECKING`、LightGBM・ngboost・seaborn・scikit-learn・optuna の分岐、`tests/test_visualize.py` の LightGBM 4.6 の注記など。drop-legacy-compat）。`tests/test_gbdt.py`・`tests/test_optuna.py`・`tests/test_search_space.py` でこの spec が触るのは module の先頭の optuna の import だけで、他の行は drop-legacy-compat が触る
- `tech.md` の「互換性」「Key Libraries」「Key Technical Decisions」の節、Core Technologies の Framework・Core deps の行、`product.md`（drop-legacy-compat）。Architecture と sklearn の estimator の決まりは、どちらの spec も触らない
- CHANGELOG、tag、PyPI への公開、licond への連絡（release-0.5.0）
- `docs_src/conf.py`（`yikit.__version__` を読むので変えない）、`.github/release.yml`（リリースノートの分類。変えない）
- `dummy_tqdm` などの公開のクラスの定義（基底のクラスを含む）

### Allowed Dependencies
- ビルド: setuptools>=77.0.3、setuptools-scm>=10.1（依存の vcs-versioning を含む）。pip の build isolation で取得する
- GitHub Actions の部品: `actions/checkout@v7`、`actions/setup-python@v7`、`dorny/paths-filter@v4`、`actions/upload-artifact@v7`、`actions/download-artifact@v8`、`sigstore/gh-action-sigstore-python@v3.5.0`、`pypa/gh-action-pypi-publish@release/v1`、`peaceiris/actions-gh-pages@v4`
- 環境 `pypi`・`testpypi` の trusted publishing（今の登録のまま）
- 手元: yikit-dev（Python 3.12）の ruff 0.16 以上、mypy 2.4 以上、pytest、build 1.6.1、actionlint。setuptools-scm は作者の承認の後に入れる
- テストは標準ライブラリと pytest、`packaging`（pytest の依存）だけを使う。optional の依存がないときは `pytest.importorskip` で skip する（今のテストと同じ形）
- sdist の中身は setuptools の `MANIFEST.in` の `prune`・`exclude` だけで絞る（setuptools-scm の file finder が追跡しているファイルを足した後に適用される）

### Revalidation Triggers
- `requires-python` や CI の行列を変えたとき: drop-legacy-compat の「分岐の一覧」の「残す」の行（下限以上の版の違いのための分岐）がまだ範囲の中にあるかと、`tech.md` の互換性の「CI の 3.10 の job が入れる版」の記述を見直す。下限そのものは使う API で決めていて CI では確かめないので、行列を変えても下限は変わらない
- CI で入れる extra（`.[test,optional]`・`.[test]`）や 3.14 の job（行列の 3.14 と `test-no-optional`）を変えたとき: drop-legacy-compat の 3.14 での ngboost のテスト（その spec の Requirement 2.3）と、optional の依存なしの job でのテスト（同じく Requirement 4.6）が、まだ意図どおりに走るかを確かめ直す（どちらもその spec のタスク 4.3 で確かめる）
- drop-legacy-compat が ngboost の上限を変えたとき（0.5.11 に対応したとき）: `.github/dependabot.yml` の ngboost の `ignore` を外すか、新しい上限に合わせる
- 追跡するファイルの最上位に、開発のためだけのディレクトリや dotfile・設定を足したとき: `MANIFEST.in` に足す。`src/yikit` にデータのファイルを足すとき: `MANIFEST.in` がそれを外さないこと（wheel に効く）を確かめる
- `[tool.setuptools_scm]` や `__init__.py` の版の読み方を変えたとき: docs の `conf.py`、`tests/test_version.py`、release-0.5.0 の手順を確かめる
- release workflow の起動の条件や公開の条件を変えたとき: release-0.5.0 の手順と、PyPI・TestPyPI の trusted publishing の登録を確かめる
- workflow のファイル名を変えたとき: trusted publishing の登録と README のバッジを直す
- ruff の `target-version` や mypy の `python_version` を変えたとき: すべてのコードが新しい対象で通るか確かめる
- ブランチの保護で必須の checks を設定するとき: matrix の job を飛ばしたときの check の名前（Risks を参照）。`test-no-optional` は matrix でないので、飛ばしても名前が変わらない

## Architecture

### Existing Architecture Analysis
- 版番号は `src/yikit/__init__.py` の `__version__ = "0.4.0"` にあり、`[tool.setuptools.dynamic]` がそれを読む。docs の `conf.py` も `yikit.__version__` を読む
- CI は `on.*.paths` で起動を絞り、`ubuntu-24.04` で 3.8〜3.14 を回す。release は tag と手動実行で起動し、TestPyPI の job は手動実行でも走る（同じ版を 2 回出すと失敗する）
- dev-foundation が決めた点は保つ: `fail-fast: false`、`pip install ".[test,optional]"`、ruff の規則の固定、mypy の `files`、dependabot の `increase-if-necessary`
- テストの多くは optional の依存を `pytest.importorskip` で扱うが、`tests/test_gbdt.py`・`tests/test_optuna.py`・`tests/test_search_space.py` は module の先頭で optuna を import するので、optional の依存がない環境では集める段階で失敗する。他のテストは optional の依存がなくても通る（research.md の「optional の依存なしの job」）
- sdist は、setuptools-scm の file finder が git で追跡しているすべてのファイルを入れる。今は `MANIFEST.in` がない
- 詳細は `research.md` の「今のファイルの状態」

### Architecture Pattern & Boundary Map

```mermaid
graph TB
    GitTag[Git tag] --> ScmBuild[Build with setuptools scm]
    BuildMetadata --> ScmBuild
    ScmBuild --> VersionFile[Generated version file]
    VersionFile --> VersionSource
    VersionSource --> VersionTests
    VersionSource --> DocsConf[docs conf]
    SdistManifest --> ScmBuild
    CIWorkflow --> ScmBuild
    CIWorkflow --> Pytest
    Pytest --> OptionalTestGuard
    ReleaseWorkflow --> ScmBuild
    ReleaseWorkflow --> PyPI
    ReleaseWorkflow --> TestPyPI
    ReleaseWorkflow --> GitHubRelease
    DocsWorkflow --> ScmBuild
    DocsWorkflow --> DocsConf
    DocsWorkflow --> GitHubPages
    DependabotConfig --> CIWorkflow
    DependabotConfig --> ReleaseWorkflow
    DependabotConfig --> DocsWorkflow
    LintTargetConfig --> PyVersionSweep
    PackagingSteering --> BuildMetadata
```

**Architecture Integration**:
- Selected pattern: 版番号の真実の出どころは git の tag だけにし、ビルドのときに 1 回だけ決めて `_version.py` と配布物のメタデータの両方に書く。workflow は「どの ref でビルドするか」と「どこに出すか」だけを決める
- Domain/feature boundaries: ビルドの宣言（BuildMetadata・SdistManifest）、版の読み方（VersionSource・VersionTests）、workflow（CI・Release・Docs・Dependabot）、テストの読み込み（OptionalTestGuard）、lint の対象（LintTargetConfig・PyVersionSweep）、steering（PackagingSteering）の 6 つに分かれ、互いのファイルを共有しない
- 依存の向き: BuildMetadata → VersionSource → VersionTests。SdistManifest は sdist の中身だけを決め、wheel と版には関わらない。workflow は BuildMetadata のビルドの仕方に依存し、逆はない。CIWorkflow の `test-no-optional` は OptionalTestGuard があって初めて通る。PackagingSteering は他のすべてを記述するだけで、何からも参照されない
- Existing patterns preserved: src レイアウト、`[test,optional]` の extra で CI に入れる方式、`pytest -ra`、ruff の規則の固定、`concurrency`（CI は cancel-in-progress: true）
- Steering compliance: 公開 API を変えない、ruff・mypy は手元、依存の下限は drop-legacy-compat に任せる

### Technology Stack

| Layer | Choice / Version | Role in Feature | Notes |
|-------|------------------|-----------------|-------|
| Build | setuptools>=77.0.3、setuptools-scm>=10.1（最新は 10.3.4、vcs-versioning 2.6.0） | PEP 639 のライセンス、tag からの版、`_version.py` の生成 | python-template は `setuptools-scm[toml]>=7`。違いは後述 |
| Runtime | Python 3.10〜3.14 | 対応する版 | 3.10 のサポートは 2026-10-31 に終わるが残す（作者の決定） |
| CI | GitHub Actions、`ubuntu-latest`（今は 24.04）、`dorny/paths-filter@v4` | テストと判定。3.10〜3.14 の行列（`.[test,optional]`）と、3.14 の `test-no-optional`（`.[test]`） | setup-python は 3.10〜3.14 を 22.04・24.04・26.04 に配っている |
| sdist | `MANIFEST.in`（`prune`・`exclude`） | 開発のためだけのファイルを sdist から外す | setuptools は file finder の後に `MANIFEST.in` を当てる（setuptools 84 のソースで確認） |
| Release | `pypa/gh-action-pypi-publish@release/v1`、`sigstore/gh-action-sigstore-python@v3.5.0`、`gh release` | 公開と署名 | trusted publishing（`pypi`・`testpypi`） |
| Lint / Type（手元） | ruff 0.16 以上（`target-version = "py310"`）、mypy 2.4 以上（`python_version = 3.10`） | 3.10 を対象にした検査 | CI では走らせない |
| Workflow の検査（手元） | actionlint 1.7.12 | workflow の構文と shellcheck | 今の release workflow には SC2086 の指摘が 2 件ある |

## File Structure Plan

### Modified Files
- `pyproject.toml`（BuildMetadata） — `[build-system] requires = ["setuptools>=77.0.3", "setuptools-scm>=10.1"]`。`requires-python = ">=3.10"`。`license = "Apache-2.0"` と `license-files = ["LICEN[CS]E*"]`（3.8 のためのコメントを消す）。classifiers を 3.10〜3.14 に。`dependencies` の `typing_extensions` の行を消す（他の行は触らない）。`[tool.setuptools.dynamic]` を消す。`[tool.setuptools_scm]` を足す
- `src/yikit/__init__.py`（VersionSource） — `__version__ = "0.4.0"` を、`_version.py` があれば読み、なければ `"unknown"` にする形に置き換える。docstring と `__license__`・`__author__`・`__copyright__` は残す
- `.gitignore`（VersionSource） — `src/yikit/_version.py` を足す
- `.github/workflows/CI.yml`（CIWorkflow） — `permissions`、`changes` job、`test` job（`needs`・`if`、`ubuntu-latest`、3.10〜3.14、`fail-fast: false`、`pytest -ra`）、`test-no-optional` job（`needs`・`if` は `test` と同じ、Python 3.14、`pip install ".[test]"`、`pytest -ra`）、actions を新しい版に。`on.*.paths` と runner の固定のコメントを消す
- `.github/workflows/release-pypi.yml`（ReleaseWorkflow） — python-template の `release.yml` の形に（`concurrency`、`permissions`、`PUBLISH_PYPI`・`PYPI_PACKAGE_NAME`、outputs、公開の条件）。build の checkout の `fetch-depth` の切り替えと、ローカル版を外す `env`。TestPyPI は手動実行でも出し、手動実行では `skip-existing`。`$GITHUB_OUTPUT` を引用符で囲む。ファイル名は変えない
- `.github/workflows/docs.yml`（DocsWorkflow） — actions を新しい版に。checkout の `fetch-depth` を release と同じ式にし、`fetch-tags` を消す
- `.github/dependabot.yml`（DependabotConfig） — github-actions の項目（週ごと、`chore: `、上限 5、`groups` で 1 つの PR）を足す。pip の項目に ngboost の `ignore`（`>=0.5.11`）と理由の注記を足す。pip の項目の他の設定は変えない
- `tests/test_gbdt.py`・`tests/test_optuna.py`・`tests/test_search_space.py`（OptionalTestGuard） — module の先頭の `import optuna`・`from optuna... import ...` を、`optuna = pytest.importorskip("optuna")` の後に置く。他の行は変えない
- `ruff.toml`（LintTargetConfig） — `target-version = "py310"`
- `mypy.ini`（LintTargetConfig） — `python_version = 3.10`
- `src/yikit/helpers/_helpers.py`（PyVersionSweep） — `sys.version_info` の分岐を `from collections.abc import Callable, Iterable, Iterator` にし、使われなくなる `import sys` と `# HACK: when drop python3.8 ...` の注記を消す
- `src/yikit/visualize/_yyplot.py`（PyVersionSweep） — `sys.version_info` の分岐を `from collections.abc import Sequence` にし、`import sys` を消す。`NDArray` の注記には触らない
- `src/yikit/visualize/_utils.py`（PyVersionSweep） — `Callable` を `collections.abc` から import する
- `.kiro/steering/tech.md`（PackagingSteering） — PackagingSteering に書いた節だけを直す（実装のタスクで行う）。`README.md` は確かめるだけ

### New Files
- `tests/test_version.py`（VersionTests） — `yikit.__version__` が生成された版のファイルと一致すること、版のファイルがないと `"unknown"` になること
- `MANIFEST.in`（SdistManifest） — 開発のためだけのファイルを sdist から外す `prune`・`exclude` の行（中身は SdistManifest の節）

### Generated Files（git で追跡しない）
- `src/yikit/_version.py` — setuptools-scm がビルドとインストールのときに書く。`.gitignore` で外す。ruff は gitignore を尊重するので検査しない。mypy は `files = src/yikit` で読む（生成される中身は注釈つきで 3.10 で通る）

### Unchanged Files（確かめた）
- `docs_src/conf.py` — インストールした yikit の `__version__` を読むので変えない（`docs_src/` は sdist に残す。SdistManifest）
- `README.md` — Python の版を書いた文がない。PyPI の pyversions のバッジは classifiers から作られる
- `.github/release.yml`、`.python-version`（3.12.8。release と docs の Python に使われる）

## System Flows

### 版番号が決まる流れ

```mermaid
flowchart TD
    Start[Build starts] --> OverrideSet{Override env set}
    OverrideSet -->|release build job| Strict[local scheme no local version strict]
    OverrideSet -->|other builds| Default[local scheme node and date]
    Strict --> Ref{Checkout ref}
    Default --> Ref
    Ref -->|tag push, shallow, only that tag| Exact[Version equals tag]
    Ref -->|branch, full history and tags| Dev[Next dev version with distance]
    Ref -->|branch, shallow, no tags| Placeholder[0.1.dev1 placeholder]
    Exact --> Write[Write src yikit _version.py and metadata]
    Dev --> Write
    Placeholder --> Write
```

- tag の push（release・docs）では浅い clone なので、同じコミットに他の tag があっても push された tag の版になる（5.4、6.2）
- tag ではない ref の手動実行（release・docs）では全履歴と tag を取るので、最後の tag からの距離で開発版になる（3.3）。release では上書きの環境変数でローカル版を外す（5.5）
- CI の test job は浅い clone なので仮の版（`0.1.dev1+g<hash>`）になる。テストは版の値に依存しない

### release workflow の job

```mermaid
sequenceDiagram
    participant Trigger as Tag push or dispatch
    participant Build as build
    participant PyPI as publish to PyPI
    participant Rel as github release
    participant Test as publish to TestPyPI
    Trigger->>Build: checkout and build sdist and wheel
    Build-->>PyPI: tag without dash or with rc
    Build-->>Rel: any tag
    Build-->>Test: any tag or dispatch
```

- `concurrency` は `${{ github.workflow }}-${{ github.ref }}` で、`cancel-in-progress: false`（5.7）
- 手動実行（tag ではない ref）では PyPI と GitHub Release の job は飛ばされ、TestPyPI だけが走る（5.5）

## Requirements Traceability

| Requirement | Summary | Components | Interfaces | Flows |
|-------------|---------|------------|------------|-------|
| 1.1 | 3.10 以上を宣言 | BuildMetadata | `requires-python` | — |
| 1.2 | classifiers 3.10〜3.14 | BuildMetadata | classifiers | — |
| 1.3 | 3.9 以下では選ばれない | BuildMetadata | `Requires-Python` のメタデータ | — |
| 1.4 | `typing_extensions` を宣言しない | BuildMetadata | `dependencies` | — |
| 1.5 | 3.8・3.9 の書き分けを持たない | PyVersionSweep | import の書き換え、注記の削除 | — |
| 2.1 | SPDX の式で宣言 | BuildMetadata | `license` | — |
| 2.2 | LICENSE を含み、メタデータに示す | BuildMetadata | `license-files`、`License-Expression`・`License-File` | — |
| 2.3 | 式を受け付ける版のビルドの道具を求める | BuildMetadata | `[build-system] requires` | — |
| 2.4 | ライセンスの非推奨の警告なし | BuildMetadata | 手元のビルドの確認 | — |
| 3.1 | `vX.Y.Z` から `X.Y.Z` | BuildMetadata、ReleaseWorkflow | `[tool.setuptools_scm]`、checkout | 版番号が決まる流れ |
| 3.2 | `vX.Y.Z-rc.N` から `X.Y.ZrcN` | BuildMetadata、ReleaseWorkflow | 既定の tag の正規表現と PEP 440 の正規化 | 版番号が決まる流れ |
| 3.3 | tag の後は開発版 | BuildMetadata | `guess-next-dev` | 版番号が決まる流れ |
| 3.4 | `__version__` が入れた版と同じ | VersionSource、VersionTests | `_version.py`、`__version__` | — |
| 3.5 | 版のファイルがなくても import できる | VersionSource、VersionTests | `"unknown"` | — |
| 3.6 | 版のファイルを追跡しない | VersionSource | `.gitignore` | — |
| 3.7 | 版を手で書かない | VersionSource、BuildMetadata | `[tool.setuptools.dynamic]` の削除 | — |
| 4.1 | 変更の判定 | CIWorkflow | `changes` job、paths-filter | — |
| 4.2 | 3.10〜3.14 でテスト | CIWorkflow | `test` job の行列 | — |
| 4.3 | 手動実行では必ずテスト | CIWorkflow | `if` の `workflow_dispatch` | — |
| 4.4 | 飛ばしても成功で完了 | CIWorkflow | job の `if`（`on.paths` を使わない） | — |
| 4.5 | 他の版を止めず全体は失敗 | CIWorkflow | `fail-fast: false` | — |
| 4.6 | 読む権限だけ | CIWorkflow | `permissions` | — |
| 4.7 | runner を固定しない | CIWorkflow | `runs-on: ubuntu-latest` | — |
| 4.8 | ruff・mypy を実行しない | CIWorkflow | job を置かない | — |
| 4.9 | 3.14 で optional なしの job を別の名前で | CIWorkflow | `test-no-optional` job、`pip install ".[test]"` | — |
| 4.10 | optional なしでもテストが読み込めて通る | OptionalTestGuard | module の先頭の `pytest.importorskip("optuna")` | — |
| 5.1 | 正式版の公開 | ReleaseWorkflow | 公開の条件 | release workflow の job |
| 5.2 | rc の公開 | ReleaseWorkflow | 公開の条件、pre-release の判定 | release workflow の job |
| 5.3 | rc 以外の `-` は PyPI に出さない | ReleaseWorkflow | 公開の条件 | release workflow の job |
| 5.4 | 同じコミットの tag でも push された版 | ReleaseWorkflow | build の checkout（tag は浅く） | 版番号が決まる流れ |
| 5.5 | 手動実行は開発版を TestPyPI だけに | ReleaseWorkflow | checkout（全履歴）、上書きの `env`、`if` | 版番号が決まる流れ |
| 5.6 | 同じ版は飛ばす | ReleaseWorkflow | `skip-existing` | — |
| 5.7 | 実行中を取り消さない | ReleaseWorkflow | `concurrency` | release workflow の job |
| 5.8 | 権限は job ごと | ReleaseWorkflow | `permissions` | — |
| 5.9 | trusted publishing の登録のまま | ReleaseWorkflow | ファイル名、`environment.name` | — |
| 6.1 | 正式版の tag で docs | DocsWorkflow | 起動の条件、`conf.py` の `release` | 版番号が決まる流れ |
| 6.2 | 同じコミットの rc でも正式版の版 | DocsWorkflow | checkout（tag は浅く、`fetch-tags` なし） | 版番号が決まる流れ |
| 7.1 | 最新の major の部品 | CIWorkflow、ReleaseWorkflow、DocsWorkflow | `uses:` の版 | — |
| 7.2 | GitHub Actions の更新を 1 つの PR に | DependabotConfig | `groups` | — |
| 7.3 | pip の方針を変えない | DependabotConfig | `versioning-strategy` | — |
| 7.4 | ngboost の 0.5.11 以上を提案しない | DependabotConfig | pip の項目の `ignore` と注記 | — |
| 8.1 | ruff が 3.10 で 0 件 | LintTargetConfig、PyVersionSweep | `target-version = "py310"` | — |
| 8.2 | mypy が 3.10 で 0 件 | LintTargetConfig | `python_version = 3.10` | — |
| 8.3 | ライブラリの書き分けは変えない | PyVersionSweep | 書き換えの範囲 | — |
| 9.1 | 公開 API を変えない | PyVersionSweep、VersionSource | 書き換えの範囲 | — |
| 9.2 | 同じ結果 | PyVersionSweep | 既存のテスト | — |
| 9.3 | wheel のモジュールが同じ | BuildMetadata | 手元のビルドの確認 | — |
| 10.1 | tech.md を 0.5.0 に | PackagingSteering | 節の書き換え | — |
| 10.2 | 持つ節だけを変え、drop-legacy-compat の節と product.md を変えない | PackagingSteering | 書き換えの範囲 | — |
| 10.3 | README の Python の版 | PackagingSteering | README の確認（記述なし） | — |
| 11.1 | 入れるときは承認 | — | 手元のビルドの確認の手順 | — |
| 11.2 | TestPyPI の手動実行は確認 | ReleaseWorkflow | 手動実行の確認の手順 | — |
| 11.3 | tag・リリース・docs の手動実行をしない | DocsWorkflow、ReleaseWorkflow | 確認の手順 | — |
| 12.1 | 開発のためだけのファイルを sdist に入れない | SdistManifest | `prune`・`exclude` | — |
| 12.2 | ソース・tests・examples・docs_src などを sdist に残す | SdistManifest | 外さないものの一覧 | — |
| 12.3 | sdist から wheel をビルドでき、wheel は変わらない | SdistManifest、BuildMetadata | `global-exclude` を使わない、手元のビルドの確認 | — |

## Components and Interfaces

| Component | Domain/Layer | Intent | Req Coverage | Key Dependencies (P0/P1) | Contracts |
|-----------|--------------|--------|--------------|--------------------------|-----------|
| BuildMetadata | ビルドの宣言 | Python の版、ライセンス、ビルドの要件、版の決め方を宣言する | 1.1–1.4, 2.1–2.4, 3.1–3.3, 3.7, 9.3, 12.3 | setuptools（P0）、setuptools-scm（P0） | State |
| SdistManifest | ビルドの宣言 | 開発のためだけのファイルを sdist から外す | 12.1–12.3 | setuptools の `MANIFEST.in`（P0）、setuptools-scm の file finder（P0） | State |
| VersionSource | 版の読み方 | `yikit.__version__` を生成された版のファイルから読む | 3.4–3.7, 9.1 | BuildMetadata（P0） | Service |
| VersionTests | テスト | 版の読み方と、版のファイルがないときの値を確かめる | 3.4, 3.5 | VersionSource（P0） | — |
| CIWorkflow | workflow | 変更を判定し、3.10〜3.14 と、3.14 の optional なしでテストする | 4.1–4.9, 7.1 | GitHub Actions（P0）、paths-filter（P1）、OptionalTestGuard（P0） | Batch |
| OptionalTestGuard | テスト | optional の依存がない環境でも、テストの module の読み込みで失敗しないようにする | 4.10 | pytest の `importorskip`（P0） | — |
| ReleaseWorkflow | workflow | tag と手動実行でビルドして公開する | 3.1, 3.2, 5.1–5.9, 7.1, 11.2, 11.3 | BuildMetadata（P0）、trusted publishing（P0） | Batch |
| DocsWorkflow | workflow | 正式版の tag で docs を作って公開する | 6.1, 6.2, 7.1, 11.3 | BuildMetadata（P0）、`docs_src/conf.py`（P1） | Batch |
| DependabotConfig | workflow | pip と GitHub Actions の更新の提案。ngboost の 0.5.11 以上は提案させない | 7.2, 7.3, 7.4 | dependabot（P1）、drop-legacy-compat の ngboost の上限（P1） | State |
| LintTargetConfig | 手元の検査 | ruff と mypy の対象を 3.10 にする | 8.1, 8.2 | ruff（P0）、mypy（P0） | State |
| PyVersionSweep | コード | 3.8・3.9 のための import と注記を消す | 1.5, 8.1, 8.3, 9.1, 9.2 | LintTargetConfig（P0） | — |
| PackagingSteering | steering | tech.md の持つ節を 0.5.0 に合わせ、README を確かめる | 10.1–10.3 | 他のすべて（P1） | — |

### ビルドの宣言

#### BuildMetadata

| Field | Detail |
|-------|--------|
| Intent | `pyproject.toml` のビルドと Python の版とライセンスと版の決め方の宣言 |
| Requirements | 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.7, 9.3, 12.3 |

**Responsibilities & Constraints**
- `[build-system]`: `requires = ["setuptools>=77.0.3", "setuptools-scm>=10.1"]`、`build-backend = "setuptools.build_meta"`
- `[project]`
  - `dynamic = ["version"]`（今のまま）
  - `requires-python = ">=3.10"`
  - `license = "Apache-2.0"`、`license-files = ["LICEN[CS]E*"]`。表の形と 3.8 のためのコメントを消す
  - classifiers: `"Programming Language :: Python :: 3"` と 3.10・3.11・3.12・3.13・3.14
  - `dependencies` から `"typing_extensions >= 4.5; python_version < '3.10'"` を消す。他の行（scikit-learn など）は drop-legacy-compat が触るので、この spec では変えない
- `[tool.setuptools.dynamic]`（`version = { attr = "yikit.__version__" }`）を消す。残すと setuptools-scm と版の出どころが 2 つになる
- `[tool.setuptools_scm]`:
  ```toml
  [tool.setuptools_scm]
  version_file = "src/yikit/_version.py"
  write_to_source = true
  ```
  tag の設定（`tag.prefix`・`tag.regex`）は置かない（既定の正規表現で `v0.5.0-rc.1` が `0.5.0rc1` になる）。`fallback_version` も置かない

**Contracts**: State [x]
- tag と版の対応: `vX.Y.Z` → `X.Y.Z`、`vX.Y.Z-rc.N` → `X.Y.ZrcN`、`vX.Y.Z-beta.N` → `X.Y.ZbN`。tag の後の N コミット目は `X.Y.(Z+1).devN+g<hash>`（rc の後は `X.Y.Zrc(N+1).devM+g<hash>`）
- 配布物のメタデータ: `Metadata-Version: 2.4`、`License-Expression: Apache-2.0`、`License-File: LICENSE`、`Requires-Python: >=3.10`
- wheel の中身: 今の `.py` のモジュールに `yikit/_version.py` と `.dist-info/licenses/LICENSE` が加わるだけ（`src/yikit` で追跡しているのは `.py` だけ）
- sdist の中身: git で追跡しているファイルから SdistManifest が外すものを除いたものと、`PKG-INFO`・`src/yikit/_version.py`・`src/yikit.egg-info/`・`setup.cfg`（setuptools が足す）・`MANIFEST.in`
- `python -m build` は sdist を作ってから、その sdist から wheel を作る。wheel のビルドが通ることが、sdist に要るものが残っていることの確認になる（12.3）

**Implementation Notes**
- Integration: `[tool.setuptools.dynamic]` の削除と VersionSource の変更は同じコミットに入れる（片方だけだと版の出どころが食い違う）
- Validation: 手元のビルドの確認（Testing Strategy）で、tag ごとの版、メタデータ、警告、wheel の中身を確かめる
- Risks: git の情報も PKG-INFO もない場所ではビルドが失敗する（`fallback_version` を置かない。誤った版の配布物を作らないため）

#### SdistManifest

| Field | Detail |
|-------|--------|
| Intent | `MANIFEST.in` で、開発のためだけのファイルを sdist から外す（作者の決定 D3、2026-10-14） |
| Requirements | 12.1, 12.2, 12.3 |

**Responsibilities & Constraints**
- `MANIFEST.in` の中身（コメントは英語）:
  ```
  # setuptools-scm puts every git-tracked file into the sdist first;
  # these lines then drop the files that only matter for developing yikit.
  # Kept: src, tests, examples, docs_src, LICENSE, README.md, CHANGELOG.md,
  # pyproject.toml. The wheel only takes files inside src/yikit, so do not
  # add global-exclude here.
  prune .claude
  prune .github
  prune .kiro
  prune .vscode
  exclude CLAUDE.md
  exclude .gitattributes .gitignore .python-version
  exclude ruff.toml mypy.ini jupytext.toml
  ```
- 外すもの（12.1）: AI のエージェントの設定と仕様（`.claude/`・`.kiro/`・`CLAUDE.md`）、GitHub の設定（`.github/`）、エディタの設定（`.vscode/`）、最上位の dotfile（`.gitattributes`・`.gitignore`・`.python-version`）、手元の検査と examples の同期の設定（`ruff.toml`・`mypy.ini`・`jupytext.toml`）。どれもパッケージのビルド、インストール、テストに使わない（pytest の設定は `pyproject.toml` にある）
- 残すもの（12.2）: `src/`、`tests/`（`tests/imgs/` の参照画像を含む）、`examples/`、`docs_src/`、`LICENSE`、`README.md`、`CHANGELOG.md`、`pyproject.toml`。`CHANGELOG.md` は利用者向けの変更の記録なので残す
- `docs_src/` は残す: Sphinx の設定と目次（5 ファイル、約 16 KB）で、開発の道具ではなく docs のソースである。sdist と `sphinx-apidoc` だけで、ネットにつながらない環境でも docs を作れる。`examples/` を残すのと同じ考え。`docs_src/_static/.gitkeep` も残す（なければ Sphinx が `_static` がないと警告する）。そのため dotfile は `global-exclude .*` ではなく、最上位のものを名前で外す
- `prune` と `exclude` は最上位からのパスに当たる（`exclude` の `*` は `/` をまたがない）。`global-exclude` は使わない: setuptools は `MANIFEST.in` を当てた後のファイルの一覧から、パッケージの中の `.py` 以外のファイルを package data として wheel に入れる（`include-package-data` は pyproject では既定で true）ので、`global-exclude` は `src/yikit` のデータのファイルにも効きうる。今は `src/yikit` で追跡しているのは `.py` だけで、`.py` のモジュールは `MANIFEST.in` と関係なく wheel に入る
- 順番の前提: setuptools は sdist のファイルの一覧を、既定のファイルと file finder（setuptools-scm。git の index で追跡しているファイル）→ `MANIFEST.in` → ライセンスのファイル・`readme` で指したファイル → build の出力などを除く、の順に作る（setuptools 84 の `manifest_maker.run`）。そのため `MANIFEST.in` で setuptools-scm が足したファイルを外せる。`LICENSE` と `README.md` は後から必ず足される

**Contracts**: State [x]
- sdist の最上位は `CHANGELOG.md`・`LICENSE`・`MANIFEST.in`・`PKG-INFO`・`README.md`・`docs_src/`・`examples/`・`pyproject.toml`・`setup.cfg`・`src/`・`tests/` だけ（2026-10-14 の追跡しているファイルに、この `MANIFEST.in` を setuptools の `FileList` で当てると、168 ファイルが 66 ファイルになり、最上位は `MANIFEST.in`・`PKG-INFO`・`setup.cfg` を除いて上のとおり。research.md）
- wheel の中身は `MANIFEST.in` で変わらない（9.3 の比べ方で、0.4.0 の wheel との違いが版のファイルとライセンスのファイルだけ）

**Implementation Notes**
- Validation: 実装のときに、setuptools の `FileList` で `git ls-files` の一覧に `MANIFEST.in` を当てて、外すもの・残すものを確かめる（setuptools-scm を入れなくてもできる）。手元のビルドの確認（作者の承認の後）で、本物の sdist の中身の一覧と、その sdist から作った wheel を確かめる（12.1〜12.3）
- Risks: 開発のためのディレクトリや dotfile を最上位に足したとき、`MANIFEST.in` に足し忘れると sdist に入る（Revalidation Triggers）。`export-ignore`（`.gitattributes`）でも外せるが、GitHub の Source code のアーカイブからも消えるので使わない（research.md）

### 版の読み方

#### VersionSource

| Field | Detail |
|-------|--------|
| Intent | `yikit.__version__` を生成された `_version.py` から読み、なければ `"unknown"` にする |
| Requirements | 3.4, 3.5, 3.6, 3.7, 9.1 |

**Responsibilities & Constraints**
- `src/yikit/__init__.py` は python-template と同じ形にする:
  ```python
  from __future__ import annotations

  from pathlib import Path

  # setuptools-scm writes ``_version.py`` from the git tag when the package
  # is built or installed; a source tree that was never installed lacks it.
  _version_file = Path(__file__).parent / "_version.py"

  if _version_file.is_file():
      from ._version import __version__
  else:
      __version__ = "unknown"
  ```
- 例外を握りつぶさない: `try/except ImportError` ではなく、ファイルの有無で分ける（`_version.py` の中の誤りを隠さない）
- `.gitignore` に `src/yikit/_version.py` を足す（setuptools-scm の docs の勧めどおり、場所を限って書く）
- `yikit` の最上位には `__all__` がなく、公開 API はサブパッケージにある。`Path` と `_version_file` が最上位の名前に加わるが、公開 API ではない（python-template と同じ）

**Contracts**: Service [x]

##### Service Interface
```python
# yikit/__init__.py
__version__: str  # yikit._version.__version__ when the file exists, else "unknown"

# yikit/_version.py (generated by setuptools-scm, not tracked)
version: str
__version__: str
version_tuple: tuple[int | str, ...]
__version_tuple__: tuple[int | str, ...]
commit_id: str | None
__commit_id__: str | None
```
- Preconditions: なし（import に git も PKG-INFO も要らない）
- Postconditions: インストールした yikit では `__version__ == yikit._version.__version__` で、PEP 440 の版。版のファイルがないときは `"unknown"`
- Invariants: 版番号はソースのどこにも手で書かない

**Implementation Notes**
- Validation: VersionTests。手元の editable の環境は、setuptools-scm を入れて入れ直すまで `"unknown"` になる（Migration Strategy）
- Risks: 手元の最上位に古い `yikit.egg-info`（0.3.15）があり、`importlib.metadata` が誤った版を返すので、版の比較に `importlib.metadata` を使わない

#### VersionTests

| Field | Detail |
|-------|--------|
| Intent | `tests/test_version.py` で、版の読み方の 2 つの道筋を確かめる |
| Requirements | 3.4, 3.5 |

- `test_version_comes_from_generated_file`（名前は目安）: インストールした `yikit` の隣に `_version.py` があり、`yikit.__version__ == yikit._version.version` で、`packaging.version.Version(yikit.__version__)` が通る。`_version.py` がないときは、入れ直し（`pip install -e .`）を案内するメッセージで失敗する
- `test_version_falls_back_without_version_file`: インストールした `yikit/__init__.py` を `tmp_path` の別名のパッケージに写し、`importlib.util.spec_from_file_location` で読み込むと、`__version__ == "unknown"` で import に失敗しない
- 英語の docstring とコメントで書く（`test_public_api.py` の決まりに合わせる）

### workflow

#### CIWorkflow

| Field | Detail |
|-------|--------|
| Intent | 変更を判定し、テストに関わる変更があれば Python 3.10〜3.14 と、3.14 の optional なしでテストする |
| Requirements | 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 7.1 |

**Contracts**: Batch [x]

##### Batch / Job Contract
- Trigger: `workflow_dispatch`、main への `push`、`pull_request`（opened・synchronize・reopened・ready_for_review）。`on` に `paths` を書かない（4.4）
- Workflow: `name: CI`、`run-name` と `concurrency`（cancel-in-progress: true）は今のまま。`permissions: contents: read`（4.6）
- `changes` job: `runs-on: ubuntu-latest`、`permissions: contents: read, pull-requests: read`、`outputs.code`。`actions/checkout@v7`（`fetch-depth: 0`。push のときに直前のコミットと比べるため）。`dorny/paths-filter@v4`（`id: filter`、`if: github.event_name != 'workflow_dispatch'`）。filter `code` は今の `on.paths` と同じ一覧（`src/**`、`**/*.py`、`tests/**`、`packages/**`、`.github/workflows/CI.yml`、`pyproject.toml`、`setup.py`、`setup.cfg`、`requirements*.txt`、`tox.ini`）（4.1）
- `test` job: `needs: changes`、`if: needs.changes.outputs.code == 'true' || github.event_name == 'workflow_dispatch'`（4.3、4.4）。`runs-on: ubuntu-latest`（4.7）。`strategy.fail-fast: false`、`matrix.python-version: ["3.10", "3.11", "3.12", "3.13", "3.14"]`（4.2、4.5）。`actions/checkout@v7`（既定の浅い clone）、`actions/setup-python@v7`、`pip install ".[test,optional]"`、`pytest -ra`
- `test-no-optional` job（作者の決定 D1、2026-10-14）: `needs: changes`、`if` は `test` と同じ（4.3、4.4）。`runs-on: ubuntu-latest`。matrix を使わず、`name` も書かない。`actions/checkout@v7`、`actions/setup-python@v7`（`python-version: "3.14"`）、`pip install ".[test]"`、`pytest -ra`（4.9）
  - 目的: optional の依存（optuna・lightgbm・ngboost・Boruta・tqdm・colorlog）がないときに yikit とテストが動くことを確かめる。あわせて pandas 3 を試し続ける（drop-legacy-compat が 3.14 でも ngboost を入れると、ngboost の依存の lifelines が pandas<3.0 を求め、`test (3.14)` では pandas 3 が入らなくなる）
  - check の名前は job の ID の `test-no-optional` になる（`name` のない job は ID が名前になる。今の docs の workflow の `docs` job と同じ）。行列の job は `test (3.10)` のように組み合わせの値が付くので、名前はぶつからない。matrix でないので、`if` で飛ばしても名前は変わらない
  - 行列の `include` にしない理由: 今の行列にない鍵だけを足す `include`（`{python-version: "3.14", extras: "test"}`）は、GitHub の決まりでは元の 3.14 の組み合わせに足され、6 つ目の job にならない（元の値を上書きしない限り、既存の組み合わせに足される）。新しい鍵をすべての組み合わせに持たせると、`test (3.x)` の名前がすべて変わる
- ruff・mypy の job は置かない（4.8）
- Output: job ごとの成否。飛ばしたときは `test` と `test-no-optional` が skipped として報告され、実行は成功で終わる

**Implementation Notes**
- Validation: actionlint の指摘 0 件。`on` に `paths` がなく、`test` と `test-no-optional` の job に `if` がある（4.1、4.4 の静的な確認）。作業ブランチでの手動実行で、5 つの版と `test-no-optional` の 6 つの job が通り、`test-no-optional` の log で optional の依存が入らず pandas 3 が入っている。テストを飛ばす判定は PR の差分全体を比べるのでマージの前には観察できず、main に入った後、コード以外だけを変える push で `test` が飛ばされ実行が成功になることを、作者か release-0.5.0 が確かめる（この spec のタスクにはしない）
- Risks: test job の setuptools-scm は浅い clone の警告を出し、仮の版になる（テストは版の値に依存しない）。`test-no-optional` は OptionalTestGuard がないと、テストを集める段階で 3 つの module が失敗する。CI で入れる extra や 3.14 の job を変えるときは、drop-legacy-compat の 3.14 での ngboost のテスト（その spec のタスク 4.3）も確かめ直す（Revalidation Triggers）

#### OptionalTestGuard

| Field | Detail |
|-------|--------|
| Intent | optional の依存がない環境でも、テストの module の読み込みで失敗せず、optional の依存を使うテストだけが skip されるようにする |
| Requirements | 4.10 |

**Responsibilities & Constraints**
- 対象は module の先頭で optuna を import している 3 つのファイルだけ（yikit-dev で optional の依存を塞いで確かめた。research.md）
  - `tests/test_gbdt.py`: `import optuna` を消し、`lightgbm = pytest.importorskip("lightgbm")` の前に `optuna = pytest.importorskip("optuna")` を置く
  - `tests/test_optuna.py`: 標準ライブラリ・numpy・pytest・scikit-learn の import の後に `optuna = pytest.importorskip("optuna")` を置き、`import optuna` を消して、`from optuna.distributions import ...`・`from optuna.trial import TrialState` と yikit の import をその後に移す
  - `tests/test_search_space.py`: 標準ライブラリ・numpy・pytest・scikit-learn の import の後に `optuna = pytest.importorskip("optuna")` を置き、`from optuna.distributions import (...)` と yikit の import をその後に移す
- yikit の import も `importorskip` の後に置く: `yikit.models` は optuna がないと `Objective` などを出さず、`yikit.models._search_space` は module の先頭で optuna を import するため
- `importorskip` の後の import には `# noqa: E402` を付ける（`tests/test_search_cv.py`・`tests/test_gbdt.py` の今の書き方と同じ）。テストの関数、補助の関数、`TYPE_CHECKING` の中の import、`test_optuna.py` の `OptunaSearchCV` の import の切り替えは変えない（drop-legacy-compat がこれらのファイルの他の行を後で直す）
- optional の依存がすべて入った環境では、集まるテストと結果は変わらない
- 他のテストは、optional の依存を使うところで `pytest.importorskip` か `is_installed` をすでに使っていて、optional の依存がなくても通る

**Implementation Notes**
- Validation: yikit-dev で、optional の依存（`colorlog`・`tqdm`・`ngboost`・`lightgbm`・`boruta`・`optuna`・`optuna_integration`）を `sys.modules[name] = None` で塞ぐ `sitecustomize.py` を scratchpad に置き、`PYTHONPATH` で読ませて `python -m pytest -ra -p no:cacheprovider` を実行する。何も入れずに、`.[test]` だけの環境を真似られる（子プロセスにも効く）。collection の error と failed が 0 件。塞がずに実行した結果は 1.1 の基準と同じ
- Risks: 後で module の先頭に optional の依存の import を足すと、`test-no-optional` が落ちる。drop-legacy-compat は、この 3 つのファイルを直すときに `importorskip` を残す

#### ReleaseWorkflow

| Field | Detail |
|-------|--------|
| Intent | tag の push で PyPI・TestPyPI・GitHub Release に出し、手動実行で TestPyPI に開発版を出す |
| Requirements | 3.1, 3.2, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 7.1, 11.2, 11.3 |

**Contracts**: Batch [x]

##### Batch / Job Contract
- File: `.github/workflows/release-pypi.yml`（名前は変えない。5.9）
- Trigger: `workflow_dispatch`、tag の push（`v[0-9]+.[0-9]+.[0-9]+`、`v[0-9]+.[0-9]+.[0-9]+-*`）
- Workflow: `concurrency: { group: ${{ github.workflow }}-${{ github.ref }}, cancel-in-progress: false }`（5.7）。`env: { PUBLISH_PYPI: "true", PYPI_PACKAGE_NAME: "yikit" }`。`permissions: contents: read`（5.8）
- `build` job:
  - `outputs`: `publish_pypi`、`package_name`（`check` の step で `env` を `"$GITHUB_OUTPUT"` に書く）
  - `env`: `SETUPTOOLS_SCM_OVERRIDES_FOR_YIKIT: '{local_scheme = "no-local-version-strict"}'`（5.5。追跡しているファイルに変更があればビルドを失敗させる）
  - `actions/checkout@v7` に `fetch-depth: ${{ startsWith(github.ref, 'refs/tags/') && 1 || 0 }}`（tag は push された tag だけを浅く。それ以外は全履歴と全 tag。5.4、5.5）
  - `actions/setup-python@v7`（版は `.python-version`）、`pip install ".[dev]"`、`python3 -m build`、`actions/upload-artifact@v7`（`python-package-distributions`）
- `publish-to-pypi` job: `if: startsWith(github.ref, 'refs/tags/') && (!contains(github.ref_name, '-') || contains(github.ref_name, 'rc')) && needs.build.outputs.publish_pypi == 'true'`。`environment: { name: pypi, url: https://pypi.org/p/${{ needs.build.outputs.package_name }} }`。`permissions: id-token: write`。`actions/download-artifact@v8`、`pypa/gh-action-pypi-publish@release/v1`（5.1、5.2、5.3）
- `github-release` job: `if: startsWith(github.ref, 'refs/tags/')`。`permissions: contents: write, id-token: write`。`actions/download-artifact@v8`、`sigstore/gh-action-sigstore-python@v3.5.0`、tag に `-` を含めば pre-release（`"$GITHUB_OUTPUT"`）、`gh release create --generate-notes`、`gh release upload`（5.1、5.2、5.3）
- `publish-to-testpypi` job: `if: needs.build.outputs.publish_pypi == 'true' && (startsWith(github.ref, 'refs/tags/') || github.event_name == 'workflow_dispatch')`。`environment: { name: testpypi, url: https://test.pypi.org/p/${{ needs.build.outputs.package_name }} }`。`permissions: id-token: write`。`pypa/gh-action-pypi-publish@release/v1` に `repository-url: https://test.pypi.org/legacy/` と `skip-existing: ${{ github.event_name == 'workflow_dispatch' }}`（5.6）
- Idempotency & recovery: tag の push の 2 回目（同じ版）は PyPI・TestPyPI で失敗する（版を使い直せないため、意図どおり）。手動実行の 2 回目は TestPyPI で飛ばされる。失敗した job は GitHub の「Re-run failed jobs」でやり直す

**Implementation Notes**
- Validation: actionlint の指摘 0 件。作者の確認を得てから、作業ブランチで手動実行し、build の版がローカル版を含まない開発版であること、TestPyPI に出たこと、PyPI と GitHub Release の job が飛ばされたことを確かめる（11.2）。tag の push での確認は release-0.5.0 が行う（11.3）
- Risks: TestPyPI の trusted publishing が手動実行で通るかは、手動実行したことがないので未確認。失敗したら作者が TestPyPI の設定を見直す

#### DocsWorkflow

| Field | Detail |
|-------|--------|
| Intent | 正式版の tag で、その版番号の docs を作って GitHub Pages に出す |
| Requirements | 6.1, 6.2, 7.1, 11.3 |

**Contracts**: Batch [x]

##### Batch / Job Contract
- Trigger と `permissions: contents: write` は今のまま（正式版の tag と手動実行）
- `actions/checkout@v7` に `ref: ${{ github.ref }}` と `fetch-depth: ${{ startsWith(github.ref, 'refs/tags/') && 1 || 0 }}`。`fetch-tags` は書かない（浅い clone でもすべての tag を取ってしまうため。6.2）
- `actions/setup-python@v7`、`pip install ".[docs,optional]"`、`sphinx-apidoc`、`sphinx-build`、`peaceiris/actions-gh-pages@v4` は今のまま。`docs_src/conf.py` はインストールした yikit の `__version__` を `release` に使う（6.1）

**Implementation Notes**
- Validation: actionlint の指摘 0 件。docs の手動実行は公開中の docs を書き換えるので、この spec では行わない（11.3）。正式版の tag での確認は release-0.5.0 の v0.5.0 で行う
- Risks: `sphinx-apidoc` は `_` で始まるモジュールを既定で出さないので、`_version.py` は API リファレンスに出ない

#### DependabotConfig

| Field | Detail |
|-------|--------|
| Intent | GitHub Actions の更新を 1 つの PR にまとめて提案させ、ngboost の 0.5.11 以上を提案させない |
| Requirements | 7.2, 7.3, 7.4 |

**Contracts**: State [x]
- pip の項目の今の設定は変えない（`versioning-strategy: increase-if-necessary`、週ごと、`chore: `、上限 5）（7.3）
- pip の項目に `ignore` を足す（作者の決定 D2、2026-10-14）（7.4）:
  ```yaml
      ignore:
        # ngboost 0.5.11 passes nested names in set_params/get_params, which
        # yikit does not support yet. drop-legacy-compat caps ngboost<0.5.11
        # (decided 2026-10-14); remove this entry when yikit supports it.
        - dependency-name: "ngboost"
          versions: [">=0.5.11"]
  ```
  `versions` の範囲の書き方（`">=1.0.0"` の形）は GitHub Docs の「Controlling which dependencies are updated by Dependabot」の例にある。`ignore` は version updates と security updates の両方に効く
- 追加: `package-ecosystem: "github-actions"`、`directory: "/"`、`schedule.interval: "weekly"`、`commit-message.prefix: "chore: "`、`open-pull-requests-limit: 5`、`groups: { github-actions: { patterns: ["*"] } }`（7.2）

**Implementation Notes**
- この spec が main に入った時点の ngboost の行は、まだ `>=0.3.6,<0.5.9; python_version < '3.14'` で、`ignore` は 0.5.9・0.5.10 を止めない。drop-legacy-compat が入る前に、dependabot が `<0.5.9` を 0.5.9・0.5.10 まで広げる PR を出すことがある。drop-legacy-compat はタスク 1 で今の ngboost の行を読み直すので、その PR を入れても入れなくてもよい
- ngboost 0.5.11 への対応は、作者の決定で後の版に送る（この spec でも drop-legacy-compat でも行わない）。対応するときに `ignore` を外す（Revalidation Triggers）
- Risks: ngboost の 0.5.11 以上に security advisory が出ても、dependabot の PR は来ない。0.5.11 への対応を後に送る間は、作者が ngboost の release を見る

### 手元の検査

#### LintTargetConfig

| Field | Detail |
|-------|--------|
| Intent | ruff と mypy の対象を Python 3.10 にする |
| Requirements | 8.1, 8.2 |

**Contracts**: State [x]
- `ruff.toml`: `target-version = "py310"`。他の設定（`select`、`required-imports`、`extend-exclude`、`force-exclude`、行の長さ）は変えない
- `mypy.ini`: `python_version = 3.10` を足す。`files = src/yikit`、`ignore_missing_imports = True` は変えない

#### PyVersionSweep

| Field | Detail |
|-------|--------|
| Intent | Python 3.8・3.9 のためだけの書き分けを、振る舞いを変えずに消す |
| Requirements | 1.5, 8.1, 8.3, 9.1, 9.2 |

**Responsibilities & Constraints**
- ruff の UP035・UP036 の修正（UP036 は unsafe の修正）で、次の 3 つを直す
  - `src/yikit/helpers/_helpers.py`: `from collections.abc import Callable, Iterable, Iterator`
  - `src/yikit/visualize/_yyplot.py`: `from collections.abc import Sequence`
  - `src/yikit/visualize/_utils.py`: `from collections.abc import Callable` と `from typing import TypeVar`
- 使われなくなった `import sys`（`_helpers.py`、`_yyplot.py`）を手で消す（F401 は unfixable）
- `_helpers.py` の `# HACK: when drop python3.8, use dummy_tqdm(Iterable[T])` の注記を消す。`class dummy_tqdm(Iterable, Generic[T])` は変えない（`Iterable[T]` だけにすると MRO から `Generic` が外れる）
- ライブラリの版による書き分けと注記（`_yyplot.py` の `NDArray` の注記など）には触らない（8.3）
- 3.10 で今使っている import の先は変わらない（`_helpers.py`・`_yyplot.py` は今も `collections.abc` の側を通る）。`_utils.py` の `TypeVar` の bound は型検査のための値

**Implementation Notes**
- Validation: `ruff check src tests`、`ruff format --check src tests`、`mypy` がエラー 0 件。`pytest` の全体が変更の前と同じ結果（9.2）。`grep` で `src` と `tests` に `sys.version_info` がない

### steering

#### PackagingSteering

| Field | Detail |
|-------|--------|
| Intent | `tech.md` のうち、この spec が持つ節を 0.5.0 に合わせ、README を確かめる |
| Requirements | 10.1, 10.2, 10.3 |

**Responsibilities & Constraints**
- 持つ節（10.2）: Core Technologies の Language の行、「型」、「コードの品質」、「テスト」の CI の項目、「Development Environment」、「Release」。他の節は drop-legacy-compat（Core Technologies の Framework・Core deps の行、「互換性」、「Key Libraries」、「Key Technical Decisions」）か、どちらの spec も触らない（Architecture、sklearn の estimator の決まり、「テスト」の CI 以外の項目）
- Core Technologies の Language: Python 3.10 以上（`requires-python = ">=3.10"`）、CI で 3.10〜3.14、3.8・3.9 の環境は 0.4.x の系列が受け持つ
- 型:
  - 「実行時に評価される場所では PEP 585/604 を使わない（3.8 対策）」を消す
  - `cast` は、`TYPE_CHECKING` の中で import した型を使うときに文字列で書く
  - 「新しい版の numpy などにしかない型は `TYPE_CHECKING` の中で import する」は残す（0.5.0 でも使う決まり。drop-legacy-compat の「互換性」は、書き直さずにこの節を参照する）
  - mypy の項目を `python_version = 3.10` に直し、3.8 の説明を消す
- コードの品質: ruff の対象を Python 3.10 に
- テスト（CI の項目）: `ubuntu-latest`、3.10〜3.14 で `pip install ".[test,optional]"`、Python 3.14 で `.[test]` だけを入れる `test-no-optional` の job（optional の依存がないときの動作と pandas 3 を受け持つ。テストの module の先頭で optional の依存を import せず、`pytest.importorskip` を使う）、`changes` job で変更を判定してテストを飛ばす（飛ばしても成功）、`permissions` は読み取りだけ、に直す。runner の固定の説明を消す。必須の checks を設定するときは matrix でない job（`changes`・`test-no-optional`）を選ぶこと、GitHub Actions の部品は dependabot が 1 つの PR にまとめて上げることを書く
- Development Environment: editable で入れると `src/yikit/_version.py` ができること、tag を付けた後や版を確かめたいときは入れ直すこと、Build のコマンド（`python -m build`）を足す
- Release: 版番号は setuptools-scm が git の tag から決めること（`vX.Y.Z` → `X.Y.Z`、`vX.Y.Z-rc.N` → `X.Y.ZrcN`、版を上げるコミットは要らない、同じコミットに rc と正式版の tag を付けてよい）、tag の push での公開先、手動実行は TestPyPI に `X.Y.(Z+1).devN` を出すこと、PyPI では版を使い直せないこと、ビルドは `setuptools>=77.0.3` と PEP 639 の `license`、sdist は `MANIFEST.in` で開発のためだけのファイルを外すこと（最上位に開発用のファイルを足したら `MANIFEST.in` にも足す）、workflow のファイル名は trusted publishing の登録に使われていること
- `updated_at` を更新する
- 触らない: 互換性、Key Libraries、Key Technical Decisions、Core Technologies の Framework・Core deps、Architecture、sklearn の estimator の決まり、`product.md`（10.2）
- README: Python の版を書いた文がないことを確かめる（10.3。直すものはない）

## python-template との違い

python-template（2026-10-14 の main）と違えた点と、その理由。

| 箇所 | python-template | yikit | 理由 |
|------|-----------------|-------|------|
| `requires-python`・classifiers | `>= 3.9`、3.9〜3.14 | `>=3.10`、3.10〜3.14 | 作者の決定（2026-10-14） |
| `typing_extensions` の依存 | `python_version < '3.10'` で宣言 | 宣言しない | 3.10 以上では不要で、yikit は import していない |
| ビルドの要件 | `setuptools-scm[toml]>=7` | `setuptools-scm>=10.1` | `version_file` は 8.0、`write_to_source` は 10.1 から。`toml` の extra は 8.0 から中身が空 |
| 版のファイルの設定 | `write_to = ".../_version.py"` | `version_file = "src/yikit/_version.py"`、`write_to_source = true` | `write_to` は非推奨で sdist から壊れる。`write_to_source` を書かないと非推奨の警告が出て、将来 editable で版のファイルができなくなる |
| `.gitignore` | `_version.py`（場所を限らない） | `src/yikit/_version.py` | setuptools-scm の docs の勧め。他の場所の `_version.py` を隠さない |
| CI の行列 | 3.9〜3.14、`fail-fast` は既定 | 3.10〜3.14、`fail-fast: false` | 作者の決定。ある版の失敗で他の版を止めない（dev-foundation の決定） |
| CI の pytest | `pytest -xs .` | `pytest -ra` | `testpaths` を設定済み。最初の失敗で止めず、skip の理由を出す |
| CI の optional なしの job | なし | `test-no-optional`（3.14、`.[test]`） | 作者の決定 D1。yikit は重い依存を optional にしているので、ないときの動作と pandas 3 を確かめる |
| sdist の中身 | `MANIFEST.in` なし（追跡しているファイルをすべて入れる） | `MANIFEST.in` で開発のためだけのファイルを外す | 作者の決定 D3。yikit は `.kiro/`・`.claude/` など AI のエージェントの仕様と設定を追跡している |
| release のファイル名 | `release.yml` | `release-pypi.yml` | PyPI・TestPyPI の trusted publishing の登録と README のバッジがこの名前を使う |
| `PUBLISH_PYPI` | `"false"` | `"true"` | yikit は公開する |
| release の build の checkout | 既定（浅い clone） | tag は浅く、それ以外は全履歴（`fetch-depth` の式） | 手動実行で正しい開発版を作るため。tag では push された tag だけにする |
| release の build の `env` | なし | `SETUPTOOLS_SCM_OVERRIDES_FOR_YIKIT` に `no-local-version-strict` | TestPyPI はローカル版を受け付けない。汚れたツリーからの公開を止める |
| TestPyPI への公開 | tag の push だけ | tag の push と手動実行。手動実行では `skip-existing` | brief: 手動実行で TestPyPI への公開を確かめられるようにし、同じ版の 2 回目で失敗しない |
| `$GITHUB_OUTPUT` | 引用符なし | `"$GITHUB_OUTPUT"` | actionlint（shellcheck SC2086）の指摘をなくす |
| docs の checkout | `fetch-depth: 0`、`fetch-tags: true` | release と同じ `fetch-depth` の式、`fetch-tags` なし | 同じコミットの rc の tag が版に使われないようにする |
| dependabot | pip・docker・github-actions | pip（`increase-if-necessary`、ngboost の `>=0.5.11` の `ignore`）・github-actions | yikit に docker はない。pip の方針は dev-foundation の決定。`ignore` は作者の決定 D2 |
| `ruff.toml`・`mypy.ini` | 最小の設定 | `target-version = "py310"` と規則の固定、mypy の `files` と `python_version` | dev-foundation の決定（規則の固定）に、この spec の 3.10 を足す |

同じにした点: `__init__.py` の版の読み方（`"unknown"` の代わりの値を含む）、PEP 639 の `license-files`、CI の `changes` job と `permissions`、release の `concurrency` と公開の条件と job の権限、actions の版、dependabot の github-actions の `groups`。

## Error Handling

### Error Strategy
- git の情報も PKG-INFO もない場所でのビルド: setuptools-scm が版を決められずにビルドが失敗する。`fallback_version` で誤った版の配布物を作るより、失敗を選ぶ
- release のビルドで追跡しているファイルに変更がある: `no-local-version-strict` でビルドを失敗させ、公開しない
- 手動実行で同じ版がすでに TestPyPI にある: `skip-existing` で公開を飛ばし、job は成功する。tag の push での重複は失敗のまま（PyPI では版を使い直せない）
- `_version.py` がない: `__version__` を `"unknown"` にし、import は失敗させない。VersionTests はインストールした環境でこれを失敗として扱い、入れ直しを案内する
- CI の test job の浅い clone: setuptools-scm の警告だけで、失敗にはしない
- optional の依存がない環境: optional の依存を使うテストは `pytest.importorskip` で skip し、テストを集める段階では失敗させない（OptionalTestGuard）。yikit 自身は `is_installed()` で optional の機能を出さないだけで、import は失敗しない（今のまま）
- sdist に要るファイルを `MANIFEST.in` で外してしまった: sdist から作る wheel のビルドか、展開した sdist でのテストが失敗する。手元のビルドの確認で見つける

### Monitoring
- 手動実行の確認では、build の log の版と、TestPyPI の JSON API（`https://test.pypi.org/pypi/yikit/<版>/json`）の `requires_python`・`license_expression` を見る
- CI の手動実行の確認では、`test-no-optional` の `pip install` の log で、optional の依存が入らず pandas 3 が入ったことを見る

## Testing Strategy

### 単体テスト（`tests/test_version.py`）
- インストールした yikit の `__version__` が `yikit._version.version` と同じで、PEP 440 の版として読める（3.4）
- `_version.py` のないパッケージとして `__init__.py` を読み込むと、import は通り `__version__ == "unknown"`（3.5）

### 既存のテストと手元の検査
- 手元（3.12）で `python -m pytest` の全体が変更の前と同じ結果で通る（9.2）。`_version.py` を作るには editable の入れ直しが要る（下の手元のビルドの確認）
- `ruff check src tests`、`ruff format --check src tests`、`mypy` がエラー 0 件（8.1、8.2）
- optional の依存を `sys.modules[name] = None` で塞いだ `sitecustomize.py`（scratchpad）を `PYTHONPATH` で読ませて `python -m pytest -ra -p no:cacheprovider` を実行し、collection の error と failed が 0 件（4.10）。変更の前は `tests/test_gbdt.py`・`tests/test_optuna.py`・`tests/test_search_space.py` の 3 つが collection で失敗し、残りは 296 passed・102 skipped（2026-10-14）
- `actionlint .github/workflows/*.yml` の指摘 0 件、`.github/dependabot.yml` と workflow が YAML として読める。dependabot の pip の項目に ngboost の `ignore` がある（4.x、5.x、6.x、7.x）
- setuptools の `FileList` で `git ls-files` の一覧に `MANIFEST.in` を当て、外すものがなく、残すものがある（12.1、12.2 の前もっての確認。setuptools-scm は要らない）

### 手元のビルドの確認（作者の承認の後）
- 作者の承認を得て、yikit-dev に setuptools-scm（10.1 以上。依存の vcs-versioning を含む）を入れる。build 1.6.1 と setuptools 84.0.0 はすでにある。以後は `pip install --no-build-isolation --no-deps -e .` と `python -m build --no-isolation` で、他のダウンロードをしない（11.1）
- 作業ブランチのコミットを scratchpad に clone し、その clone だけに tag を付ける（本物のリポジトリには tag を付けない。11.3）
  - 同じコミットに `v0.5.0-rc.1`（annotated）と `v0.5.0`（lightweight）を付け、actions/checkout と同じく push した tag だけを浅く取った clone（`git fetch --no-tags --depth=1 file://<clone> +refs/tags/<tag>:refs/tags/<tag>`）でビルドすると、`yikit-0.5.0rc1` と `yikit-0.5.0` の sdist と wheel ができる（3.1、3.2、5.4、6.2）
  - tag の後にコミットを足すと、上書きの環境変数なしでは `0.5.1.dev1+g<hash>`、ありでは `0.5.1.dev1` になる。追跡しているファイルを変えたままだと、上書きありのビルドは失敗する（3.3、5.5）
  - wheel の METADATA に `Metadata-Version: 2.4`、`License-Expression: Apache-2.0`、`License-File: LICENSE`、`Requires-Python: >=3.10` と 3.10〜3.14 の classifiers があり、`typing_extensions` がない。ビルドの log にライセンスの非推奨の警告がない（1.1〜1.4、2.1〜2.4）
  - 同じ clone の `v0.4.0` から作った wheel と、ファイルの一覧を比べ、違いが `yikit/_version.py` とライセンスのファイルだけであること（9.3、12.3。`v0.4.0` には `MANIFEST.in` がないので、`MANIFEST.in` が wheel を変えないことの確認にもなる）
  - `v0.5.0` の sdist の中身の一覧（`tar -tzf`）に、SdistManifest で外すもの（`.kiro/`・`.claude/`・`CLAUDE.md`・`.github/`・`.vscode/`・`.gitattributes`・`.gitignore`・`.python-version`・`ruff.toml`・`mypy.ini`・`jupytext.toml`）がなく、残すもの（`src/`・`src/yikit/_version.py`・`tests/`・`tests/imgs/`・`examples/`・`docs_src/`・`LICENSE`・`README.md`・`CHANGELOG.md`・`pyproject.toml`）がある（12.1、12.2）。`python -m build` が sdist から wheel を作れた（12.3）
  - その sdist を scratchpad に展開し、展開した `tests/` で `python -m pytest -ra -p no:cacheprovider` を実行して、手元の全体と同じ数が通る（参照画像など、テストに要るものが sdist にある。12.2）
- editable を入れ直した後、`src/yikit/_version.py` ができ、`git status` に出ないこと（3.6）

### CI と公開の確認
- 作業ブランチで CI を手動実行し、3.10〜3.14 の 5 つの job と `test-no-optional` の 6 つが通る。`test-no-optional` の check の名前が `test (3.x)` とぶつからず、log で optional の依存が入らず pandas 3 が入っている（4.2、4.3、4.5、4.9、4.10）
- 作者の確認を得て、作業ブランチで release workflow を手動実行する。build の版が `0.4.1.devN`（ローカル版なし）で、TestPyPI に出て、PyPI と GitHub Release の job が飛ばされる（5.5、5.9、11.2）
- マージの後の確認（この spec のタスクにはしない。作者か release-0.5.0 に渡す。tasks.md の Implementation Notes）: main に入った後、コード以外だけを変える push で `test` と `test-no-optional` が飛ばされ、実行が成功になる（4.1、4.4）。tag の push での PyPI・TestPyPI・GitHub Release への公開（rc を含む。5.1〜5.4）、正式版の tag での docs の版（6.1、6.2）。GitHub Actions の部品の新しい版が出たら、dependabot の PR が 1 つにまとまる（7.2）。作者が手動実行を断ったときの TestPyPI の手動実行（5.5、5.6）

## Risks and Open Questions
- **必須の checks**: matrix の `test` job を `if` で飛ばすと、版ごとの名前（`test (3.12)` など）ではなく job の名前だけの check になる（コミュニティで知られた振る舞い）。今は main に保護も ruleset もない（2026-10-14）。将来、必須の checks を設定するときは `changes`・`test-no-optional` のような matrix でない job を選ぶか、結果をまとめる job を足す。tech.md に書く
- **pandas 3 を試すのは `test-no-optional` だけになる**: drop-legacy-compat の後は、optional を入れる job では lifelines が pandas<3.0 を求める。pandas 3 と optional の依存の組み合わせ（例: ngboost と pandas 3）は CI で試さない（作者の決定 D1 の範囲）
- **optional の依存を module の先頭で import するテスト**: 後で足すと `test-no-optional` が collection で落ちる。tech.md の「テスト」の CI の項目に書く
- **手動実行の版のぶつかり**: 別のブランチで最後の tag からの距離が同じだと版が同じになり、後の方の TestPyPI への公開は飛ばされる。手動実行の確認では build の log の版と TestPyPI の版を照らし合わせる
- **TestPyPI の手動実行は未経験**: release workflow を手動実行したことがない。trusted publishing が通らなければ、作者が TestPyPI の登録を見直す
- **作者の editable の環境**: setuptools-scm を入れて入れ直すまで `yikit.__version__` は `"unknown"` で、`tests/test_version.py` は失敗する
- **sdist の中身**: setuptools-scm を使うと、sdist に git で追跡しているすべてのファイル（2026-10-14 に合わせて約 3.5 MB）が入るので、作者の決定 D3 で `MANIFEST.in` を足す。外すのは約 0.9 MB（`.kiro/` 約 640 KB、`.claude/` 約 200 KB など）で、`examples/`（約 2.0 MB）と `tests/`（約 330 KB）は残る。最上位に開発用のファイルを足したときに `MANIFEST.in` へ足し忘れると sdist に入る（Revalidation Triggers、tech.md の Release）
- **dependabot の ngboost の `ignore`**: 0.5.11 以上の security advisory の PR も来なくなる。0.5.11 への対応（後の版）で `ignore` を外す
- **Python 3.10 のサポートの終わり（2026-10-31）**: 作者の決定で 3.10 を残す。setup-python が 3.10 を配らなくなったら行列から外す

## Migration Strategy
1. 作業ブランチで BuildMetadata と VersionSource を同じコミットで変える。その後、作者の editable の環境は、setuptools-scm を入れて入れ直すまで `yikit.__version__` が `"unknown"` になる
2. 版番号の手書きをやめた後、main のビルドの版は `0.4.1.devN+g<hash>` に見える。release-0.5.0 が `v0.5.0-rc.1` を付けると `0.5.0rc1` になり、確かめた後に同じコミットへ `v0.5.0` を付けてよい
3. Python 3.8・3.9 の利用者は、pip が 0.4.x を選ぶ。0.4.x の保守が要るときは `v0.4.0` からブランチを切る（この spec の外）
4. sdist を使う人（再配布する人を含む）: 0.5.0 の sdist には `.kiro/`・`.claude/`・`.github/` などが入らない。`tests/`・`examples/`・`docs_src/` は残る
5. 戻し方: この spec の PR を revert すれば、手書きの版（0.4.0）と 3.8 の設定に戻る。tag を付ける前なら、PyPI への影響はない
