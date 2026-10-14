# Research & Design Decisions

## Summary
- **Feature**: `packaging-and-ci`
- **Discovery Scope**: Extension（既存のビルドの設定・workflow・lint の設定への手入れと、版番号の決め方の置き換え。外部の道具（setuptools-scm、GitHub Actions の部品）の仕様は一次資料で確かめた）
- **Key Findings**:
  - setuptools-scm の `write_to` は 8.0 から非推奨で、`version_file` に置き換わった。10.1 で `write_to_source` が入り、設定しないと非推奨の警告が出る。python-template はまだ `write_to` と `setuptools-scm[toml]>=7` を使っている
  - actions/checkout v7 は、tag の push では既定（`fetch-depth: 1`）で「push された tag だけ」を取ってくる。`fetch-depth: 0` にするとすべての tag が入り、同じコミットに rc と正式版の tag があると `git describe` が rc を選ぶことがある（annotated と lightweight が混ざるとき）。一方、tag ではない ref の手動実行では、履歴と tag がないと版が `0.1.dev1+g<hash>` になる
  - TestPyPI と PyPI はローカル版（`+g<hash>`）を受け付けない。setuptools-scm の `SETUPTOOLS_SCM_OVERRIDES_FOR_YIKIT='{local_scheme = "no-local-version-strict"}'` で、release のビルドだけローカル版を外せる（作業ツリーが汚れていればビルドを失敗させる）
  - 0.5.0 の spec の横断のレビューの後、作者が 2026-10-14 に 3 つを決めた: D1 CI に Python 3.14 で `.[test]` だけを入れる job を 1 つ足す、D2 dependabot が ngboost の 0.5.11 以上を提案しない（0.5.11 への対応は後の版）、D3 `MANIFEST.in` で sdist から開発のためだけのファイルを外す。D1 には、module の先頭で optuna を import する 3 つのテストのファイルの手直しが要る

## Research Log

### 今のファイルの状態（2026-10-14、ブランチ `docs/spec-0.5.0`）
- **Context**: brief の Current State を、実際のファイルで確かめる
- **Sources Consulted**: `pyproject.toml`、`src/yikit/__init__.py`、`.gitignore`、`.github/workflows/{CI,release-pypi,docs}.yml`、`.github/dependabot.yml`、`.github/release.yml`、`ruff.toml`、`mypy.ini`、`docs_src/conf.py`、`README.md`、`.python-version`
- **Findings**:
  - `pyproject.toml`: `requires = ["setuptools>=61"]`、`license = { text = "Apache-2.0" }`（3.8 のためのコメント付き）、`requires-python = ">= 3.8"`、classifiers は 3.8〜3.14、依存の先頭に `typing_extensions >= 4.5; python_version < '3.10'`、`[tool.setuptools.dynamic] version = { attr = "yikit.__version__" }`
  - `typing_extensions` は `src` でも `tests` でも import されていない（`grep`）。宣言だけが残っている
  - `src/yikit/__init__.py` は `__version__ = "0.4.0"` を手で書いている
  - `.gitignore` に `_version.py` の行はない
  - CI: `on.push.paths` と `on.pull_request.paths` で起動を絞る。`runs-on: ubuntu-24.04`、3.8〜3.14、`fail-fast: false`、`actions/checkout@v4`・`actions/setup-python@v5`、`permissions` なし、`concurrency`（cancel-in-progress: true）あり
  - release: ファイル名は `release-pypi.yml`。build・PyPI・GitHub Release（Sigstore）・TestPyPI の 4 つの job。TestPyPI の job には `if` がなく、手動実行でも走る。`concurrency` と workflow の `permissions` はない。`# FIXME` のコメントが残っている
  - docs: 正式版の tag と手動実行で起動。`ref: ${{ github.ref }}`・`fetch-depth: 0`・`fetch-tags: true`。`permissions: contents: write`。`peaceiris/actions-gh-pages@v4` で `force_orphan: true` の公開
  - dependabot: pip だけ。`versioning-strategy: increase-if-necessary`
  - `ruff.toml`: `target-version = "py38"`。`mypy.ini`: `files = src/yikit`、`ignore_missing_imports = True`（版の指定なし）
  - `docs_src/conf.py` は `from yikit import __version__` を `release` に使う。変える必要はない
  - `README.md` に Python の版を書いた文はない。PyPI の pyversions のバッジ（classifiers から作られる）だけがある
  - `.python-version` は `3.12.8`。release と docs の `actions/setup-python` は版を指定していないので、このファイルの 3.12.8 を使う
  - tag はすべて lightweight（`git for-each-ref refs/tags --format='%(objecttype)'` が commit）。最新は `v0.4.0`（main のマージのコミット）
- **Implications**: 版番号は tag から決めても、`yikit.__version__` を読む側（docs）は変えずに済む。`typing_extensions` の行は消すだけでよい

### python-template の今の書き方
- **Context**: brief の Approach（python-template にそろえる）
- **Sources Consulted**: `https://api.github.com/repos/yu9824/python-template/git/trees/main?recursive=1`、raw の `pyproject.toml`、`src/python_template/__init__.py`、`.gitignore`、`.github/workflows/{CI,release,docs}.yml`、`.github/dependabot.yml`、`HOW_TO_RELEASE.md`、`ruff.toml`、`mypy.ini`
- **Findings**:
  - `pyproject.toml`: `requires = ["setuptools>=77.0.3", "setuptools-scm[toml]>=7"]`、`license = "MIT"`、`license-files = ["LICEN[CS]E*"]`、`requires-python = ">= 3.9"`、`typing_extensions` の依存あり、`[tool.setuptools_scm] write_to = "src/python_template/_version.py"`
  - `__init__.py`: `Path(__file__).parent / "_version.py"` があれば `from ._version import __version__`、なければ `__version__ = "unknown"`
  - `.gitignore`: 先頭の「custom」に `_version.py`
  - CI: workflow の `permissions: contents: read`。`changes` job（`pull-requests: read` を足す、`actions/checkout@v7` を `fetch-depth: 0`、手動実行では `dorny/paths-filter@v4` を飛ばす）と、`needs: changes` と `if` の付いた `test` job（`ubuntu-latest`、3.9〜3.14、`fail-fast` は既定、`pytest -xs .`）
  - release: `concurrency`（`cancel-in-progress: false`）、`env` の `PUBLISH_PYPI`（既定 "false"）と `PYPI_PACKAGE_NAME` を build job の outputs で後の job に渡す。checkout は既定（浅い clone）。TestPyPI は tag の push のときだけ。`actions/upload-artifact@v7`、`actions/download-artifact@v8`、`sigstore/gh-action-sigstore-python@v3.5.0`
  - docs: `actions/checkout@v7`（`fetch-depth: 0`・`fetch-tags: true`）、`actions/setup-python@v7`
  - dependabot: pip・docker・github-actions。github-actions は `groups` で 1 つの PR にまとめる
- **Implications**: 大筋はそのまま取り入れる。yikit の事情で変える点は「python-template との違い」（design.md）に理由とともに書く

### setuptools-scm の設定（10.3.4、2026-09-23）
- **Context**: Requirement 3（tag からの版）、brief の「`version_file` と `write_to`」
- **Sources Consulted**: PyPI の JSON API（`setuptools-scm`・`vcs-versioning`・`setuptools`）、`pypa/setuptools-scm` の `docs/config.md`・`docs/overrides.md`・`docs/integrations.md`・`docs/usage.md`、`setuptools-scm/CHANGELOG.md`、`vcs-versioning/CHANGELOG.md`、`vcs-versioning/src/vcs_versioning/_config.py`・`_backends/_git.py`・`_dump_version.py`
- **Findings**:
  - 最新は setuptools-scm 10.3.4（requires_python `>=3.8`、`vcs-versioning>=2.5.0,<3` に依存）、vcs-versioning 2.6.0（`packaging>=26.2` に依存。yikit-dev の packaging は 26.3）。extra の `toml` は残っているが中身は空（8.0 から tomli は通常の依存）
  - `version_file` は 8.0.0 で `write_to` の代わりとして入った。`write_to` は「sdist から使うと壊れる」ので非推奨（`docs/config.md`）
  - `write_to_source` は 10.1.0 で入った。設定しないと、版のファイルをソースのツリーに書き、同時に「将来の major で既定を変える」という `DeprecationWarning` を出す。`true` にすると警告なしでソースのツリーにも書く。10.0.2 から、版のファイルは `build_py` の出力にも登録される
  - 生成される `_version.py`（vcs-versioning の既定の型）は `from __future__ import annotations` 付きで、`__version__: str` などの注釈を持つ。Python 3.10 で読める
  - 版のファイルを git で追跡していると警告が出る（vcs-versioning 1.0.0）。`.gitignore` には `src/<pkg>/_version.py` を書くよう勧めている
  - 既定の tag の正規表現 `^(?:[\w-]+-)?(?P<version>[vV]?\d+(?:\.\d+){0,2}[^\+]*)(?:\+.*)?$`、版の組み立ては `guess-next-dev`、ローカル版は `node-and-date`
  - `git describe --dirty --tags --long --abbrev=40 --match *[0-9]*` で tag を探す（`--tags` なので lightweight の tag も使う）。tag が見つからないときは `0.0`（`fallback_version` があればそれ）に距離を足し、`0.1.devN+g<hash>` になる
  - 浅い clone では既定（`warn_on_shallow`）で警告だけを出す。HEAD がちょうど tag のときは警告しない
  - 環境変数 `SETUPTOOLS_SCM_OVERRIDES_FOR_${DIST_NAME}` で `pyproject.toml` の設定を上書きできる（yikit では `SETUPTOOLS_SCM_OVERRIDES_FOR_YIKIT`）。公式の GitHub Actions の例は、build の job に `'{local_scheme = "no-local-version-strict"}'` を付け、checkout を `fetch-depth: 0` にしている
  - `no-local-version-strict`（vcs-versioning 1.0.0、setuptools-scm 10.0.0 から）は、ローカル版を外し、追跡しているファイルに変更があればビルドを失敗させる。追跡していないファイル（gitignore した `build/`・`*.egg-info/`・`_version.py`）は汚れと見なさない
  - setuptools-scm を設定すると、sdist には git で追跡しているファイルがすべて入る（file finder）。wheel にはパッケージの中のファイルだけが入る。`src/yikit` で追跡しているのは `.py` だけ（`git ls-files src`）
  - setuptools-scm は setuptools 61 以上で動き、80 以上を勧めている。setuptools 77.0.3 は Python 3.9 以上、最新の 84.0.0 は 3.10 以上
- **Implications**: `version_file = "src/yikit/_version.py"` と `write_to_source = true` を使い、ビルドの要件は `setuptools-scm>=10.1` にする。`[toml]` の extra は付けない。release の build job にだけ上書きの環境変数を付ける

### tag と PyPI の版の対応（手元で確かめた）
- **Context**: Requirement 3.1・3.2
- **Sources Consulted**: 上の正規表現と、yikit-dev の `packaging` 26.3 の `Version`
- **Findings**:

  | tag | 正規表現の version | PyPI の版 |
  |-----|--------------------|-----------|
  | `v0.5.0` | `v0.5.0` | `0.5.0` |
  | `v0.5.0-rc.1` | `v0.5.0-rc.1` | `0.5.0rc1` |
  | `v0.4.0-rc.0` | `v0.4.0-rc.0` | `0.4.0rc0`（PyPI にある 0.4.0rc0 と一致） |
  | tag の 3 コミット後 | — | `0.5.1.dev3+g<hash>`（ローカル版を外すと `0.5.1.dev3`） |

  rc の tag の後のコミットは `0.5.0rc2.devN`（`guess-next-dev` が最後の数字を 1 つ上げる）
- **Implications**: 今の tag の付け方（`vX.Y.Z-rc.N`）を変えずに、PyPI の版が今までと同じ形になる。setuptools-scm の `tag` の設定は要らない

### 同じコミットに複数の tag があるとき
- **Context**: Requirement 5.4・6.2。setuptools-scm にすると、rc で確かめた後に同じコミットへ正式版の tag を付けられる（版を上げるコミットが要らない）
- **Sources Consulted**: scratchpad の使い捨てのリポジトリでの `git describe --dirty --tags --long --match '*[0-9]*'`
- **Findings**:

  | 同じコミットの tag | `git describe` の結果 |
  |--------------------|-----------------------|
  | `v0.5.0-rc.1` だけ | `v0.5.0-rc.1-0-g...` |
  | 両方 lightweight | `v0.5.0-0-g...`（名前の順で先の方） |
  | rc が annotated、正式版が lightweight | `v0.5.0-rc.1-0-g...`（annotated が優先され、誤り） |
  | 両方 annotated | `v0.5.0-0-g...`（新しい方） |

  - actions/checkout v7（`src/ref-helper.ts` の `getRefSpec`）は、tag の ref で `fetch-tags` が false のとき `+refs/tags/<tag>:refs/tags/<tag>` だけを取る。`fetch-depth: 0` のときは `getRefSpecForAllHistory` で、すべてのブランチと tag を取る
  - 「rc が annotated、正式版が lightweight」のリポジトリから、`git fetch --no-tags --depth=1 file://<repo> +refs/tags/v0.5.0:refs/tags/v0.5.0` で正式版の tag だけを浅く取ると `v0.5.0-0-g...`、rc の tag だけを取ると `v0.5.0-rc.1-0-g...` になった（全部を取ると `v0.5.0-rc.1`）
- **Implications**: tag の push では、checkout を浅いまま（push された tag だけ）にすると、tag の種類によらず push された tag の版になる。履歴と tag が要るのは tag ではない ref の手動実行だけなので、`fetch-depth` を ref で切り替える

### TestPyPI とローカル版
- **Context**: Requirement 5.5・5.6
- **Sources Consulted**: setuptools-scm の `docs/integrations.md`（Publishing to PyPI from CI/CD）、PyPI・TestPyPI の yikit の JSON API、`gh run list --workflow release-pypi.yml`、`gh api repos/yu9824/yikit/environments`
- **Findings**:
  - PEP 440 のローカル版（`+` 以降）は PyPI と TestPyPI に上げられない
  - 今の release workflow は手動実行されたことがない（すべて tag の push）。TestPyPI には 0.4.0 までの版がある
  - 環境 `pypi`・`testpypi` に保護の規則やブランチの制限はない（`github-pages` だけにブランチの制限がある）
  - 手動実行の版（`X.Y.(Z+1).devN`）は、同じコミット（または別のブランチの同じ距離）で同じになる。2 回目の公開は「同じファイルがある」で失敗する
- **Implications**: 手動実行のときだけ `skip-existing` を付け、同じ版の 2 回目を失敗にしない。ブランチの間で版がぶつかると、後の方は公開されずに飛ばされるので、手動実行の確認では run のログの版を見る

### CI の起動と判定
- **Context**: Requirement 4
- **Sources Consulted**: `dorny/paths-filter` v4.0.3 の README、GitHub Docs の「Troubleshooting required status checks」（検索結果）、`gh api repos/yu9824/yikit/branches/main/protection`・`rulesets`、actions/python-versions の `versions-manifest.json`、actions/runner-images の README
- **Findings**:
  - workflow の `on.paths` で起動しないと、checks は「Pending」のまま残る。job の `if` で飛ばすと「Success」として報告される
  - paths-filter は pull request では REST API で差分を取るので `pull-requests: read` が要る。push では同じブランチの直前のコミットと比べ、履歴が要る（`fetch-depth: 0`）。手動実行には比べる相手がない
  - main にはブランチの保護も ruleset もない（2026-10-14）。必須の checks は今は設定されていない
  - matrix の job を `if` で飛ばすと、版ごとの名前（`test (3.12)` など）ではなく job の名前だけの check になる。将来、版ごとの名前を必須にすると待ちのまま残る（コミュニティで知られた振る舞い。Docs には明記がない）
  - setup-python は Python 3.10.22・3.11.17・3.12.15・3.13.16・3.14.8 を ubuntu 22.04・24.04・26.04 に配っている。3.9 は 26.04 に配っていない。`ubuntu-latest` は今 24.04
- **Implications**: python-template の `changes` job をそのまま使い、runner は `ubuntu-latest` にする。必須の checks を将来設定するときの注意は tech.md と Risks に書く

### GitHub Actions の部品の最新版（2026-10-14）
- **Context**: Requirement 7.1
- **Sources Consulted**: `https://api.github.com/repos/<owner>/<repo>/releases/latest`
- **Findings**: actions/checkout v7.0.1、actions/setup-python v7.0.0、dorny/paths-filter v4.0.3、actions/upload-artifact v7.0.2、actions/download-artifact v8.0.2、sigstore/gh-action-sigstore-python v3.5.0、peaceiris/actions-gh-pages v4.1.0、pypa/gh-action-pypi-publish v1.14.2（`release/v1` の枝で使う）。python-template の指定（`@v7`・`@v4`・`@v8`・`@v3.5.0`・`@release/v1`）と一致する
- **Implications**: python-template と同じ指定を使う

### ruff と mypy を 3.10 にしたときの違反
- **Context**: Requirement 1.5・8
- **Sources Consulted**: yikit-dev の ruff 0.16.10 と mypy 2.4.0 を読み取りだけで実行（`ruff check --no-cache --target-version py310 --diff src tests`、`--unsafe-fixes --diff`、`ruff format --no-cache --target-version py310 --check src tests`、`mypy --python-version 3.10 --cache-dir=<scratchpad>`）
- **Findings**:
  - ruff の違反は 5 件: UP035（deprecated-import）3 件、UP036（outdated-version-block、unsafe な修正）2 件
    - `src/yikit/helpers/_helpers.py`: `if sys.version_info >= (3, 9): from collections.abc import Callable, Iterable, Iterator else: from typing import ...`
    - `src/yikit/visualize/_yyplot.py`: 同じ形で `Sequence`
    - `src/yikit/visualize/_utils.py`: `from typing import Callable, TypeVar`（3.10 では `collections.abc.Callable` を使う）
  - 自動修正の後、`_helpers.py` と `_yyplot.py` の `import sys` は使われなくなる（F401。`ruff.toml` で F401 は unfixable なので手で消す）
  - `ruff format` は 3.10 を対象にしても変わらない（38 files already formatted）
  - mypy は `--python-version 3.10` でもエラー 0 件（24 files）
  - `_helpers.py` に `# HACK: when drop python3.8, use dummy_tqdm(Iterable[T])` という注記がある。`class dummy_tqdm(Iterable, Generic[T])` を `Iterable[T]` だけにすると、`typing.Generic` が MRO から外れる（`collections.abc.Iterable[T]` は `types.GenericAlias` で、継承しても `Generic` にならない）。`dummy_tqdm` は `yikit.helpers` の公開のクラス
  - 3.10 で動いている今のコードは、`_helpers.py` と `_yyplot.py` で 3.9 以上の側（`collections.abc`）をすでに使っている。`_utils.py` の `F = TypeVar("F", bound=Callable)` の bound は `typing.Callable` から `collections.abc.Callable` に変わるが、型検査のための値で、実行時の振る舞いは変わらない
  - ライブラリの版による書き分け（`NDArray` を `TYPE_CHECKING` で import する numpy < 1.21 の注記、LightGBM・ngboost・seaborn・sklearn・optuna の分岐、テストの「LightGBM 4.6 on Python 3.8」の注記）は drop-legacy-compat の範囲
  - `tests/conftest.py` の「CI measured at most 0.037 across Python 3.8-3.14」は、許容値を決めたときの測定の記録なので残す
- **Implications**: 書き換えは 3 ファイルの import と、`import sys` の削除と、HACK の注記の削除に限る。`dummy_tqdm` の基底のクラスは変えない

### PEP 639 とビルドの要件
- **Context**: Requirement 2
- **Sources Consulted**: PyPI の JSON API（setuptools、yikit）、dev-foundation の research.md（`setuptools>=77.0.3` を 3.8 のために下げた経緯）
- **Findings**:
  - setuptools 77.0.3（2025-03-20、Python >=3.9）から PEP 639 の `license = "<SPDX>"` と `license-files` を受け付ける。表の形の `license` は 2027-02-18 の期限つきの非推奨
  - PyPI の yikit 0.4.0 は `license: Apache-2.0`、`license_expression: None`。PEP 639 にすると `License-Expression: Apache-2.0` と `License-File: LICENSE` が入る（Metadata-Version 2.4）
  - yikit の classifiers にライセンスの分類はない（PEP 639 では式と一緒に使えない）
- **Implications**: python-template と同じ `license-files = ["LICEN[CS]E*"]` を使う。ビルドの要件は `setuptools>=77.0.3`（作者の決定）

### 手元の環境（yikit-dev）
- **Context**: Requirement 11、手元での確かめ方
- **Sources Consulted**: `/Users/yu9824/opt/miniforge3/envs/yikit-dev/bin/python` での import と `importlib.metadata`
- **Findings**:
  - setuptools 84.0.0、build 1.6.1、packaging 26.3、PyYAML 6.0.3 は入っている。setuptools-scm と vcs-versioning は入っていない。sphinx は入っていない
  - yikit は editable（`__editable___yikit_0_4_0rc0_finder.py`）で、メタデータの版は `0.4.0rc0`
  - リポジトリの最上位に古い `yikit.egg-info`（Version 0.3.15、gitignore されている）がある。`python -m pytest` や `python -c` を最上位で実行すると、`importlib.metadata.version("yikit")` がこれを読んで `0.3.15` を返す
  - `actionlint` 1.7.12 が Homebrew で入っている。今の workflow には `$GITHUB_OUTPUT` を引用符で囲んでいないことによる shellcheck の指摘（SC2086、info）が 2 件ある
- **Implications**:
  - setuptools-scm を入れずに pyproject を変えると、editable を入れ直せず、`src/yikit/_version.py` ができない（`yikit.__version__` は `"unknown"`）。入れ直すには setuptools-scm が要るので、作者の承認を得る。入れるのは setuptools-scm と vcs-versioning の 2 つで、その後は `--no-build-isolation --no-deps` で入れ直し、`python -m build --no-isolation` でビルドすれば、他のダウンロードは起きない
  - テストで版を `importlib.metadata` と比べると、手元では古い egg-info のせいで失敗する。テストは `yikit._version` と比べる
  - workflow の検査に actionlint を使える。scratchpad に置いた workflow の下書き（この design の契約どおり）は actionlint の指摘 0 件だった

### optional の依存なしの job（作者の決定 D1、2026-10-14）
- **Context**: Requirement 4.9・4.10。drop-legacy-compat で 3.14 でも ngboost が入ると、ngboost の依存の lifelines が pandas<3.0 を求め、optional を入れる CI の job では pandas 3 が入らなくなる。作者は、3.14 で `.[test]` だけを入れる job を 1 つ足し、pandas 3 と optional の依存がないときの動作をそこで確かめると決めた
- **Sources Consulted**: yikit-dev での読み取りだけの試し（optional の依存を塞いだ pytest）、`gh api repos/yu9824/yikit/commits/f65c6ca.../check-runs`、GitHub Docs の「Running variations of jobs in a workflow」「Using jobs in a workflow」「Troubleshooting required status checks」
- **Findings**:
  - scratchpad の `sitecustomize.py` で `colorlog`・`tqdm`・`ngboost`・`lightgbm`・`boruta`・`optuna`・`optuna_integration` を `sys.modules[name] = None` にし、`PYTHONPATH` で読ませた（何も入れない。`importlib.util.find_spec` も None を返すので、yikit の `is_installed()` も False になる。子プロセスにも効く）
  - その状態で `import yikit.models, yikit.visualize, yikit.metrics, yikit.feature_selection` は通る
  - `pytest --co` は 3 つのファイルで集める段階で失敗する: `tests/test_gbdt.py`（module の先頭の `import optuna`。`lightgbm` の `importorskip` より前）、`tests/test_optuna.py`（先頭の `import optuna`）、`tests/test_search_space.py`（先頭の `from optuna.distributions import ...`）。この 3 つでは yikit の import も optuna に頼る（`yikit.models` の `Objective` などは optuna があるときだけ出る。`yikit.models._search_space` は先頭で optuna を import する）
  - 3 つを除くと 296 passed・102 skipped（失敗なし）。他のテストは optional の依存を `pytest.importorskip` か `is_installed` で扱っている
  - `.[test]` の依存（pytest と、scikit-learn・numpy・pandas・joblib・scipy・matplotlib・seaborn とその依存）には optional の依存が入らない
  - check の名前: 今の main のコミットの check runs は、matrix の `test` job が `test (3.8)`〜`test (3.14)`、`name` のない docs の job が `docs`（job の ID）。GitHub Docs は `jobs.<job_id>.name` を「GitHub の UI に出る job の名前」とだけ書き、`name` がないときの名前は書いていないので、実際の check runs で確かめた
  - matrix の `include`: GitHub Docs の決まりでは、`include` の項目は元の組み合わせの値を上書きせずに足せるなら、その組み合わせに足される。`{python-version: "3.14", extras: "test"}` は元の 3.14 の組み合わせに足され、新しい job にならない
  - job の `if` で飛ばした job は「Success」と報告される（GitHub Docs の「Troubleshooting required status checks」）
- **Implications**: `test-no-optional` は matrix でない別の job にし、`name` を書かない（check の名前は `test-no-optional`）。3 つのテストのファイルは、module の先頭の optuna の import を `pytest.importorskip("optuna")` の後に置く（OptionalTestGuard）。確かめは同じ塞ぎ方の pytest で、何も入れずにできる

### sdist の中身と `MANIFEST.in`（作者の決定 D3、2026-10-14）
- **Context**: Requirement 12。setuptools-scm の file finder は、git で追跡しているすべてのファイルを sdist に入れる
- **Sources Consulted**: setuptools の docs（「Controlling files in the distribution」、「Data Files Support」）、setuptools-scm の docs（Usage の file finder）、yikit-dev の setuptools 84.0.0 のソース（`setuptools/command/egg_info.py` の `manifest_maker`、`setuptools/command/build_py.py` の `analyze_manifest`、`setuptools/command/sdist.py`）、`git ls-files` と各ファイルの大きさ、setuptools の `FileList` での試し
- **Findings**:
  - `MANIFEST.in` の命令: `include`・`exclude`・`recursive-include`・`recursive-exclude`・`global-include`・`global-exclude`・`graft`・`prune`（`prune` は「`dir-pattern` に当たるディレクトリの下のすべてのファイルを外す」、`global-exclude` は「ソースのツリーのどこでも、パターンに当たるファイルを外す」）
  - setuptools 84 の `manifest_maker.run` は、`add_defaults()`（既定のファイルと、`walk_revctrl()` で setuptools-scm などの file finder が返すファイル）→ `read_template()`（`MANIFEST.in`）→ `add_license_files()` → `_add_referenced_files()`（`readme` で指したファイルなど）→ `prune_file_list()` の順に一覧を作る。`MANIFEST.in` の `prune`・`exclude` は、setuptools-scm が足したファイルにも効く。`LICENSE` と `README.md` は後から必ず足される
  - setuptools-scm の docs: git の file finder は index をもとに一覧を作り、`git archive` と同じく `export-ignore` に従う。追跡しているが配りたくないファイルは、`MANIFEST.in` の `exclude` などか、`.gitattributes` の `export-ignore` で外せる
  - wheel への影響: pyproject で設定すると `include-package-data` は既定で true。setuptools は `MANIFEST.in` を当てた後の一覧（egg-info の SOURCES.txt）から、パッケージのディレクトリの中の `.py` 以外のファイルを package data として wheel に入れる（`build_py.analyze_manifest` は `.py` のモジュールを飛ばす）。`.py` のモジュールは `MANIFEST.in` と関係なく入る。そのため、パッケージの外を `prune`・`exclude` しても wheel は変わらない。`global-exclude` はパッケージの中のデータのファイルにも当たりうる。今 `src/yikit` で追跡しているのは `.py` だけ
  - `exclude` のパターンは最上位からのパスに当たり、`*` は `/` をまたがない（distutils の `FileList`）
  - 追跡しているファイル（2026-10-14、168 個、約 3.5 MB）: `examples/` 約 2.0 MB、`.kiro/` 約 640 KB、`tests/` 約 330 KB（参照画像を含む）、`src/` 約 240 KB、`.claude/` 約 200 KB、他は小さい。最上位は `.claude/`・`.gitattributes`・`.github/`・`.gitignore`・`.kiro/`・`.python-version`・`.vscode/`・`CHANGELOG.md`・`CLAUDE.md`・`LICENSE`・`README.md`・`docs_src/`・`examples/`・`jupytext.toml`・`mypy.ini`・`pyproject.toml`・`ruff.toml`・`src/`・`tests/`
  - `docs_src/` は `conf.py`・`index.md`・`Makefile`・`make.bat`・`_static/.gitkeep` の 5 つ（約 4 KB）
  - design の `MANIFEST.in` を setuptools の `FileList` で `git ls-files` の一覧に当てると、168 個が 66 個になり、最上位は `CHANGELOG.md`・`LICENSE`・`README.md`・`docs_src/`・`examples/`・`pyproject.toml`・`src/`・`tests/` だけ。残る dotfile は `docs_src/_static/.gitkeep` だけ
- **Implications**: `MANIFEST.in` は最上位のディレクトリの `prune` と、最上位のファイルの `exclude` だけで書き、`global-exclude` は使わない。`docs_src/` は残す（docs のソースで、小さく、ネットにつながらない環境でも docs を作れる。`examples/` を残すのと同じ考え）。確かめは、実装のときの `FileList` の試しと、作者の承認の後の本物の sdist の一覧

### dependabot の ngboost の `ignore`（作者の決定 D2、2026-10-14）
- **Context**: Requirement 7.4。drop-legacy-compat は ngboost の上限を `<0.5.11` にする（0.5.11 で `set_params`・`get_params` が入れ子の名前を扱うようになり、yikit はまだ合わせていない）。0.5.11 への対応は後の版に送る
- **Sources Consulted**: GitHub Docs の「Dependabot options reference」（`ignore`）、「Controlling which dependencies are updated by Dependabot」
- **Findings**:
  - `ignore` の項目は `dependency-name`・`versions`・`update-types`。`versions` は版か範囲で、例に `versions: [ ">=1.0.0" ]` がある
  - `ignore` は version updates と security updates の両方に効く
  - `increase-if-necessary` は、新しい版が宣言の範囲の外に出たときに範囲を広げる PR を出す。今の ngboost の行（`<0.5.9; python_version < '3.14'`）には、`>=0.5.11` の `ignore` があっても、0.5.9・0.5.10 まで広げる PR が来うる
- **Implications**: pip の項目に `dependency-name: "ngboost"`、`versions: [">=0.5.11"]` と、理由の注記（drop-legacy-compat の上限の決定）を足す。0.5.9・0.5.10 への PR は止めない（drop-legacy-compat がタスク 1 で今の行を読み直す）

## Architecture Pattern Evaluation

| Option | Description | Strengths | Risks / Limitations | Notes |
|--------|-------------|-----------|---------------------|-------|
| `_version.py` を作り `__init__.py` で読む（python-template の形） | setuptools-scm が `version_file` を書き、無ければ `"unknown"` | git もメタデータもない場所でも import が速く、docs の `conf.py` を変えずに済む。python-template と同じ | 一度もインストールしていないソースでは `"unknown"` | 採用 |
| `importlib.metadata.version("yikit")` を読む（setuptools-scm の推奨） | インストールの情報から読む | 生成するファイルがない | 古い egg-info などのメタデータに引きずられる（手元で 0.3.15 を返した）。python-template と違う | 不採用 |
| `fetch-depth: 0` をいつも使う（公式の例） | release と docs で常に全履歴を取る | 単純 | 同じコミットの rc の tag が選ばれることがある | 不採用 |
| `fetch-depth` を ref で切り替える | tag の push は浅く、他は全履歴 | tag の種類によらず push された tag の版になる。手動実行も正しい開発版になる | 式が 1 行増える | 採用 |
| 必須の checks のための集約の job を足す | `test` の結果をまとめる job を 1 つ足す | 版ごとの名前を必須にしなくてよい | python-template にない。今は必須の checks がない | 不採用（Risks に記録） |
| optional なしを別の job にする | `test-no-optional`（matrix なし、3.14、`.[test]`） | 名前が `test (3.x)` とぶつからず、飛ばしても変わらない | job の手順が `test` と少し重なる | 採用（D1） |
| optional なしを行列の `include` にする | `{python-version: "3.14", extras: "test"}` | 1 つの job の定義で済む | 元の 3.14 の組み合わせに足されて 6 つ目の job にならない。避けるには全組み合わせに鍵を持たせ、`test (3.x)` の名前が変わる | 不採用 |
| sdist を `MANIFEST.in` で絞る | `prune`・`exclude` | sdist だけに効き、wheel と GitHub の Source code のアーカイブは変わらない | 最上位に開発用のファイルを足すたびに見直す | 採用（D3） |
| sdist を `.gitattributes` の `export-ignore` で絞る | setuptools-scm の file finder が従う | 1 か所で済む | GitHub の Source code のアーカイブ（`git archive`）からも消える | 不採用 |

## Design Decisions

### Decision: 版のファイルの設定は `version_file` と `write_to_source = true`
- **Context**: Requirement 3.4〜3.6
- **Alternatives Considered**:
  1. python-template と同じ `write_to`
  2. `version_file` だけ（`write_to_source` を書かない）
  3. `version_file` と `write_to_source = true`
- **Selected Approach**: 3。`[tool.setuptools_scm] version_file = "src/yikit/_version.py"`、`write_to_source = true`。ビルドの要件は `setuptools-scm>=10.1`
- **Rationale**: `write_to` は非推奨で sdist から使うと壊れる。`write_to_source` を書かないと 10.1 以降で毎回非推奨の警告が出て、将来の既定の変更で editable のときに `src/yikit/_version.py` ができなくなる（`yikit.__version__` が `"unknown"` になる）
- **Trade-offs**: setuptools-scm の下限が 10.1 になる（python-template は 7）。ビルドは build isolation で最新を取るので、利用者には影響しない
- **Follow-up**: 手元のビルドで、警告が出ないことと、`src/yikit/_version.py` ができることを確かめる

### Decision: release の build は、tag の push では浅く、手動実行では全履歴で checkout し、ローカル版を外す
- **Context**: Requirement 5.4・5.5
- **Selected Approach**: build job の checkout に `fetch-depth: ${{ startsWith(github.ref, 'refs/tags/') && 1 || 0 }}`。job の `env` に `SETUPTOOLS_SCM_OVERRIDES_FOR_YIKIT: '{local_scheme = "no-local-version-strict"}'`
- **Rationale**: tag の push では push された tag だけがあればよく、他の tag が入らない方が確実。手動実行では最後の tag からの距離を数えるために履歴と tag が要る。ローカル版は TestPyPI が受け付けないので外し、作業ツリーが汚れていればビルドを止める
- **Trade-offs**: 手動実行の版は同じ距離でぶつかりうる（skip-existing で失敗にはしない）

### Decision: docs も release と同じ checkout にする
- **Context**: Requirement 6
- **Selected Approach**: `fetch-depth` を release と同じ式にし、`fetch-tags: true` を外す
- **Rationale**: `fetch-tags: true` は浅い clone でもすべての tag を取るので、同じコミットの rc の tag が選ばれうる

### Decision: CI の test job は浅い clone のまま
- **Context**: Requirement 4
- **Selected Approach**: python-template と同じく、test job の checkout は既定のまま。版は `0.1.dev1+g<hash>` のような仮の値になり、setuptools-scm が浅い clone の警告を出す
- **Rationale**: テストは版の値に依存しない。版のファイルができていることと、`__version__` がそれと同じことだけを確かめる
- **Trade-offs**: CI のログに浅い clone の警告が出る

### Decision: release workflow のファイル名は `release-pypi.yml` のまま
- **Context**: Requirement 5.9
- **Selected Approach**: 中身は python-template の `release.yml` にそろえ、ファイル名は変えない
- **Rationale**: PyPI と TestPyPI の trusted publishing は workflow のファイル名で登録されている。README のバッジもこの名前を使う

### Decision: `PUBLISH_PYPI` の仕組みを取り入れる
- **Context**: brief の「公開するかどうかの判定」を python-template にそろえる
- **Selected Approach**: workflow の `env` に `PUBLISH_PYPI: "true"` と `PYPI_PACKAGE_NAME: "yikit"` を置き、build job の outputs で後の job の `if` と `environment.url` に渡す（python-template と同じ）。今の `# FIXME` のコメントは消える
- **Rationale**: python-template と差分が小さくなり、公開を止めたいときに 1 行で止められる
- **Trade-offs**: yikit では常に `"true"` なので、間接の手順が 1 つ増える

### Decision: 3.8・3.9 向けの書き換えは import に限り、`dummy_tqdm` の基底は変えない
- **Context**: Requirement 1.5・8.3・9.1
- **Selected Approach**: ruff の UP035・UP036 の修正（3 ファイルの import）と、使われなくなった `import sys` の削除、`_helpers.py` の HACK の注記の削除だけを行う
- **Rationale**: HACK の注記どおり `Iterable[T]` にすると、公開のクラスの MRO から `Generic` が外れる。振る舞いを変えないことを優先する
- **Follow-up**: なし

### Decision: `MANIFEST.in` で開発のためだけのファイルを sdist から外す（作者の決定 D3）
- **Context**: Requirement 12。setuptools-scm の file finder は git で追跡しているすべてのファイルを sdist に入れる。前の版の design は python-template と同じく受け入れていたが、2026-10-14 の横断のレビューの後、作者が外すと決めた
- **Alternatives Considered**:
  1. 受け入れる（python-template と同じ）
  2. `MANIFEST.in` の `prune`・`exclude`
  3. `.gitattributes` の `export-ignore`
- **Selected Approach**: 2。`prune` は `.claude`・`.github`・`.kiro`・`.vscode`、`exclude` は `CLAUDE.md`・`.gitattributes`・`.gitignore`・`.python-version`・`ruff.toml`・`mypy.ini`・`jupytext.toml`。`global-exclude` は使わない。`docs_src/` は残す
- **Rationale**: setuptools は file finder の後に `MANIFEST.in` を当てるので、追跡しているファイルを外せる。パッケージの外だけを外すので wheel は変わらない。`export-ignore` は GitHub の Source code のアーカイブも変える。`docs_src/` は docs のソース（約 4 KB）で、sdist だけで docs を作れるので残す
- **Trade-offs**: 最上位に開発用のファイルを足すたびに `MANIFEST.in` を見直す（Revalidation Triggers、tech.md の Release）
- **Follow-up**: 実装のときに `FileList` で、作者の承認の後に本物の sdist で確かめる

### Decision: optional の依存なしのテストは、3.14 の別の job にする（作者の決定 D1）
- **Context**: Requirement 4.9・4.10
- **Selected Approach**: `test-no-optional` job（`needs: changes`、`test` と同じ `if`、`ubuntu-latest`、3.14、`pip install ".[test]"`、`pytest -ra`、`name` なし）。3 つのテストのファイルの先頭の optuna の import を `pytest.importorskip("optuna")` の後に置く
- **Rationale**: 行列の `include` では 6 つ目の job にならないか、`test (3.x)` の名前が変わる。matrix でない job は飛ばしても名前が変わらない。テストのファイルの手直しは import の場所だけで、optional の依存がそろった環境の結果を変えない
- **Trade-offs**: CI の job が 1 つ増える（3.14 の行列の job と同じくらいの時間）。drop-legacy-compat は、この 3 つのファイルを直すときに `importorskip` を残す

### Decision: dependabot で ngboost の 0.5.11 以上を無視する（作者の決定 D2）
- **Context**: Requirement 7.4
- **Selected Approach**: pip の項目に `ignore: [{dependency-name: "ngboost", versions: [">=0.5.11"]}]` と、drop-legacy-compat の上限の決定を指す注記
- **Rationale**: `increase-if-necessary` は、0.5.11 が範囲の外にある限り、範囲を広げる PR を出し続ける。0.5.11 への対応は後の版に送ったので、その PR は要らない
- **Trade-offs**: 0.5.11 以上の security updates の PR も来ない。対応するときに `ignore` を外す

## Synthesis
- **Generalization**: release の build と docs は、どちらも「tag の push では push された tag の版、それ以外では最後の tag からの開発版」という同じ問題を持つ。checkout の `fetch-depth` に同じ式を使い、別々の工夫をしない
- **Build vs Adopt**:
  - 版番号は setuptools-scm に任せる（python-template と同じ）。tag の名前から shell で版を作って `SETUPTOOLS_SCM_PRETEND_VERSION` に渡す方法は、正規化や rc の形を自前で持つことになるので採らない
  - ローカル版を外すのは setuptools-scm の上書きの環境変数（`no-local-version-strict`）を使い、自前の後処理を書かない
  - 同じ版の 2 回目は `pypa/gh-action-pypi-publish` の `skip-existing` を使う
  - 変更の判定は `dorny/paths-filter` を使う（python-template と同じ）
- **Simplification**:
  - 必須の checks のための集約の job、`fallback_version`、setuptools-scm の `tag` の設定は置かない（今の要件に要らない）
  - `MANIFEST.in` は最上位の `prune` と `exclude` だけにし、`global-exclude` や `include` を使わない
  - optional なしの job は、手順を `test` と同じにし、入れる extra と Python の版だけを変える
  - CI の test job の checkout は既定のまま（版の値をテストで使わない）
  - `__init__.py` は python-template の形をそのまま使う
- **使った skill**: kiro-spec-requirements・kiro-spec-design・kiro-spec-tasks の手順と規則だけ。他の skill は使っていない。2026-10-14 の横断のレビューの後の手直し（D1〜D3 ほか）では、kiro-spec-design の design-review-gate と kiro-spec-tasks の tasks-generation の規則で、traceability・境界・タスクの番号と `_Requirements:_` を確かめ直した（subagent は使わず inline）

## Risks & Mitigations
- 必須の checks を将来、版ごとの名前（`test (3.12)` など）で設定すると、テストを飛ばした実行で待ちのまま残る — 今は main に保護がない。設定するときは `changes` のように matrix でない job を必須にするか、結果をまとめる job を足す（tech.md に書く）
- 手動実行の版が別のブランチの版とぶつかり、TestPyPI への公開が飛ばされる — 手動実行の確認では、run のログで公開した版を確かめる
- TestPyPI の trusted publishing の登録が手動実行を想定していない（手動実行したことがない） — 最初の手動実行で確かめる。失敗したら作者が TestPyPI の設定を見直す
- 作者の editable の環境で `_version.py` がなく、`yikit.__version__` が `"unknown"` になる — 入れ直しの手順を tech.md に書き、手元の確認のタスクで入れ直す（作者の承認の後）
- main から git で入れる人には、v0.5.0-rc.1 の tag までの間、版が `0.4.1.devN` に見える（0.5.0 を求める側からは選ばれない） — release-0.5.0 で rc の tag を早めに付ける
- git の情報も PKG-INFO もない場所（GitHub の自動の Source code のアーカイブなど）ではビルドが失敗する — PyPI の sdist を使ってもらう。`fallback_version` は付けない（誤った版の配布物を作らないため）
- Python 3.10 のサポートが 2026-10-31 に終わる — 3.10 は作者の決定で残す。setup-python が 3.10 を配らなくなったら行列から外す
- 後で module の先頭に optional の依存の import を足すと `test-no-optional` が落ちる — tech.md の「テスト」の CI の項目に `pytest.importorskip` を使うと書く
- 最上位に開発用のファイルを足したとき、`MANIFEST.in` に足し忘れると sdist に入る — tech.md の Release と Revalidation Triggers に書く
- dependabot の ngboost の `ignore` で、0.5.11 以上の security updates の PR も来ない — 0.5.11 への対応（後の版）で外す

## References
- [setuptools-scm: Configuration](https://setuptools-scm.readthedocs.io/en/latest/config/) — `version_file`、`write_to_source`、`write_to` の非推奨、環境変数
- [setuptools-scm: Overrides](https://setuptools-scm.readthedocs.io/en/latest/overrides/) — `SETUPTOOLS_SCM_OVERRIDES_FOR_${DIST_NAME}` と `no-local-version-strict`
- [setuptools-scm: Integrations](https://setuptools-scm.readthedocs.io/en/latest/integrations/) — PyPI・TestPyPI への公開の例
- [setuptools-scm CHANGELOG](https://github.com/pypa/setuptools-scm/blob/main/setuptools-scm/CHANGELOG.md) — 8.0.0（`version_file`）、10.1.0（`write_to_source`）
- [actions/checkout v7.0.1 src/ref-helper.ts](https://github.com/actions/checkout/blob/v7.0.1/src/ref-helper.ts) — tag の ref の refspec
- [dorny/paths-filter v4](https://github.com/dorny/paths-filter) — push と pull request での差分の取り方、権限
- [Troubleshooting required status checks](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/collaborating-on-repositories-with-code-quality-features/troubleshooting-required-status-checks) — job の `if` で飛ばすと Success、workflow の paths で飛ばすと Pending
- [actions/python-versions versions-manifest](https://github.com/actions/python-versions) — Ubuntu の各版に配っている Python
- [yu9824/python-template](https://github.com/yu9824/python-template) — そろえる先
- [PEP 639](https://peps.python.org/pep-0639/) — ライセンスの式とファイル
- [setuptools: Controlling files in the distribution](https://setuptools.pypa.io/en/latest/userguide/miscellaneous.html) — `MANIFEST.in` の命令、既定で sdist に入るファイル
- [setuptools: Data Files Support](https://setuptools.pypa.io/en/latest/userguide/datafiles.html) — `include-package-data` の既定（pyproject では true）と、`MANIFEST.in`・file finder と package data の関係
- [setuptools-scm: Usage](https://setuptools-scm.readthedocs.io/en/latest/usage/) — git の file finder（index、`export-ignore`）と、`MANIFEST.in` で外すこと
- [Running variations of jobs in a workflow](https://docs.github.com/en/actions/how-tos/write-workflows/choose-what-workflows-do/run-job-variations) — matrix の `include` が元の組み合わせに足される決まり
- [Using jobs in a workflow](https://docs.github.com/en/actions/how-tos/write-workflows/choose-what-workflows-do/use-jobs) — `jobs.<job_id>` と `jobs.<job_id>.name`
- [Dependabot options reference](https://docs.github.com/en/code-security/dependabot/working-with-dependabot/dependabot-options-reference) — `ignore`（version updates と security updates に効く）
- [Controlling which dependencies are updated by Dependabot](https://docs.github.com/en/code-security/dependabot/dependabot-version-updates/controlling-dependencies-updated) — `versions: [ ">=1.0.0" ]` の例
