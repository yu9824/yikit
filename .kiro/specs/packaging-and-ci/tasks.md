# Implementation Plan

- [ ] 1. 基盤: ビルドの宣言と版番号の出どころ
- [x] 1.1 パッケージの宣言を、Python 3.10 以上・PEP 639 のライセンス・tag からの版に切り替える
  - 何かを変える前に、手元で `python -m pytest -ra` を実行し、passed・skipped・failed などの数を Implementation Notes に記録する（2.2・2.3・4.1 で比べる基準。yikit-dev の editable は常に作業ツリーの `src` を読むので、後からは取り直せない）
  - 対応する Python を 3.10 以上にし、分類を 3.10〜3.14 にする。Python 3.10 未満のための `typing_extensions` の依存の行を消す。他の依存の行（`dependencies` と `optional` のライブラリの下限など）は触らない（drop-legacy-compat が持つ）。`test`・`dev`・`docs` の extra も変えない
  - ライセンスを SPDX の式 `Apache-2.0` で宣言し、ライセンスのファイルを配布物に含める。3.8 のための表の形とそのコメントを消す
  - ビルドの要件を setuptools 77.0.3 以上と setuptools-scm 10.1 以上にする
  - 版番号を初期化のモジュールから読む設定を消し、setuptools-scm が版のファイルを書く設定（`version_file` と `write_to_source = true`）を足す。tag の設定と `fallback_version` は置かない
  - 1.2 と同じコミットに入れる。この時点では手元の環境に入れ直さない（setuptools-scm のダウンロードが起きるため。4.1 で作者の承認を得てから行う）
  - 完了の状態: `python -c "import tomllib; ..."` で読んだ pyproject の `requires-python`・`license`・`license-files`・classifiers・`[build-system]`・`[tool.setuptools_scm]` が design のとおりで、`[tool.setuptools.dynamic]` と `typing_extensions` の行がなく、`git diff` で他の依存の行と `test`・`dev`・`docs` の extra に変更がない
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 3.1, 3.2, 3.3, 3.7_

- [x] 1.2 `yikit.__version__` を、生成された版のファイルから読むようにする
  - 版のファイルがあればそこから読み、なければ `"unknown"` にする。ファイルの有無で分け、import の例外を握りつぶさない（python-template と同じ形）
  - 手で書いた版番号を消す。docstring とライセンス・作者などの値は残す
  - 生成される版のファイルを、場所を限って git の追跡から外す
  - 完了の状態: 初期化のモジュールに版番号の文字列がなく、`git check-ignore -v src/yikit/_version.py` が `.gitignore` の行を示し、手元の今の環境で `import yikit` が失敗せずに `__version__` が `"unknown"` になる。`ruff check` と `ruff format --check` が初期化のモジュールでエラー 0 件で、`pytest tests/test_public_api.py` が通る
  - _Requirements: 3.4, 3.5, 3.6, 3.7, 9.1_

- [ ] 1.3 版番号の読み方のテストを加える
  - インストールした yikit の隣に版のファイルがあり、`__version__` がそのファイルの版と同じで、PEP 440 の版として読めることを確かめる。版のファイルがないときは、入れ直し（editable の再インストール）を案内するメッセージで失敗させる
  - 初期化のモジュールを、版のファイルのない別名のパッケージとして一時ディレクトリから読み込むと、import に失敗せず `__version__` が `"unknown"` になることを確かめる
  - 版の比較に `importlib.metadata` を使わない（手元の最上位の古い `yikit.egg-info` が誤った版を返すため）。docstring とコメントは英語で書く
  - 完了の状態: `pytest tests/test_version.py` で、版の代わりの値のテストが通り、版のファイルのテストは「版のファイルがない、入れ直す」という案内で失敗する（4.1 の入れ直しの後に通る）。`ruff check` と `ruff format --check` がこのテストのファイルでエラー 0 件
  - _Requirements: 3.4, 3.5_

- [ ] 2. 手元の検査とテストを、Python 3.10 と optional の依存なしの CI に合わせる
- [ ] 2.1 ruff と mypy の対象を Python 3.10 にする
  - ruff の対象の版を 3.10 にする。規則・除外・必須の import・行の長さは変えない
  - mypy の対象の版を 3.10 にする。対象のファイルと `ignore_missing_imports` は変えない
  - 完了の状態: `ruff check src tests --statistics` に出る違反が UP035 の 3 件と UP036 の 2 件だけで、`mypy` がエラー 0 件で終わる
  - _Requirements: 8.1, 8.2_

- [ ] 2.2 Python 3.8・3.9 のための書き分けを消す
  - ruff の UP035・UP036 の修正（UP036 は unsafe）で、helpers と visualize の 3 つのモジュールの import を `collections.abc` に揃え、Python の版の分岐をなくす。使われなくなった `sys` の import を手で消す
  - helpers の「3.8 を外したら `Iterable[T]` にする」という注記を消す。`dummy_tqdm` の基底のクラスは変えない（MRO から `Generic` が外れるため）
  - ライブラリの版による書き分けと注記（`NDArray` の注記など）には触らない
  - 完了の状態: `ruff check src tests`、`ruff format --check src tests`、`mypy` がエラー 0 件、`grep -rn "version_info" src tests` が何も出さず、手元の `python -m pytest -ra` の結果が、`tests/test_version.py` のテストを除いて、1.1 で記録した基準と同じ数になる
  - _Requirements: 1.5, 8.1, 8.3, 9.1, 9.2_

- [ ] 2.3 optional の依存がない環境でも、テストの読み込みが失敗しないようにする
  - module の先頭で optuna を import している 3 つのテストのファイル（GBDT、optuna の探索、探索範囲）で、optuna とそれに頼る yikit の import を、optuna がなければ module ごと skip する `importorskip` の後に置く。テストの関数、補助の関数、他の行は変えない（drop-legacy-compat が後でこれらのファイルの他の行を直す）
  - optional の依存（colorlog・tqdm・ngboost・lightgbm・boruta・optuna・optuna_integration）を塞いで読み込めなくする `sitecustomize.py` を scratchpad に置き、`PYTHONPATH` で読ませて pytest を回す（何も入れずに `.[test]` だけの環境を真似る）。手順は design の OptionalTestGuard
  - 完了の状態: 塞いだ pytest で collection の error と failed が 0 件で、skip の理由が optional の依存がないことだけ。塞がない `python -m pytest -ra` の結果が、`tests/test_version.py` のテストを除いて 1.1 の基準と同じ数。`ruff check src tests` と `ruff format --check src tests` がエラー 0 件。塞いだ pytest の passed・skipped の数を Implementation Notes に書く
  - _Requirements: 4.10, 9.2_

- [ ] 3. workflow・dependabot・sdist の設定を、python-template と作者の決定（2026-10-14）に合わせる
- [ ] 3.1 (P) CI を、変更の判定の job、Python 3.10〜3.14 のテストの job、optional の依存なしの job にする
  - 変更の判定の job を足す: pull request を読む権限を足し、全履歴で checkout し、手動実行では判定を飛ばす。判定の対象のパスは今の起動の条件と同じ一覧
  - テストの job は判定の結果か手動実行のときだけ走り、`ubuntu-latest` で 3.10〜3.14 の 5 つの版を、ある版の失敗で他を止めずに、test と optional の依存を入れて `pytest -ra` で回す
  - optional の依存なしの job を、行列ではない別の job として足す: テストの job と同じ判定の条件で走り、Python 3.14 で test の依存だけを入れて `pytest -ra` を回す。job の名前は書かず、job の ID（`test-no-optional`）が check の名前になるようにする
  - workflow の既定の権限を読み取りだけにし、起動の条件の `paths` と runner の固定（とそのコメント）を消す。actions を design の版にする。ruff・mypy の job は置かない
  - 完了の状態: `actionlint .github/workflows/CI.yml` の指摘が 0 件で、`on` に `paths` がなく、テストの job と optional なしの job に変更の判定への `needs` と `if`（判定の結果か手動実行）があり、行列が 3.10〜3.14 の 5 つで、optional なしの job が matrix を持たず、Python 3.14 で `.[test]` だけを入れる
  - _Boundary: CIWorkflow_
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 7.1_

- [ ] 3.2 (P) release を python-template の形にし、tag と手動実行で正しい版を作る
  - 実行中を取り消さない `concurrency`、読み取りだけの既定の権限、公開するかどうかとパッケージの名前を build の job の outputs で渡す仕組み（公開する設定）、PyPI・GitHub Release・TestPyPI の公開の条件と job ごとの権限を、design の契約どおりにする
  - build の job は、tag の ref では push された tag だけを浅く、それ以外では全履歴で checkout し、ローカル版を外して汚れたツリーでは失敗する上書きの環境変数を付ける
  - TestPyPI へは tag の push と手動実行で出し、手動実行のときだけ同じ版を飛ばす。`$GITHUB_OUTPUT` を引用符で囲む。actions を design の版にする
  - ファイル名と環境の名前（`pypi`・`testpypi`）は変えない
  - 完了の状態: `actionlint .github/workflows/release-pypi.yml` の指摘が 0 件で、`git diff --stat` でファイル名が変わっておらず、各 job の `if`・`permissions`・`environment` が design の表と一致する
  - _Boundary: ReleaseWorkflow_
  - _Requirements: 3.1, 3.2, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 7.1_

- [ ] 3.3 (P) docs の checkout を、tag の版が正しく決まる形にする
  - checkout の深さを release と同じ式にし、すべての tag を取る指定を消す。actions を design の版にする。起動の条件、権限、Sphinx の手順、公開の手順は変えない
  - docs の workflow を手動実行しない（公開中の docs を書き換えるため）
  - 完了の状態: `actionlint .github/workflows/docs.yml` の指摘が 0 件で、checkout に `fetch-tags` がなく、`fetch-depth` が release と同じ式
  - _Boundary: DocsWorkflow_
  - _Requirements: 6.1, 6.2, 7.1, 11.3_

- [ ] 3.4 (P) dependabot に GitHub Actions の更新を加え、ngboost の 0.5.11 以上を提案から外す
  - GitHub Actions の項目（週ごと、`chore: `、上限 5、すべてを 1 つにまとめる group）を足す
  - pip の項目に、ngboost の 0.5.11 以上を無視する `ignore` を足し、理由（0.5.11 の `set_params`・`get_params` の変更に yikit がまだ合わせておらず、drop-legacy-compat が上限を `<0.5.11` にした。2026-10-14 の作者の決定）の英語の注記を付ける。pip の項目の他の設定は変えない
  - 完了の状態: dependabot の設定が YAML として読め、github-actions の項目に group があり、pip の項目の `ignore` が `dependency-name: "ngboost"` と `versions: [">=0.5.11"]` で注記があり、`git diff` で pip の項目の他の行に変更がない
  - _Boundary: DependabotConfig_
  - _Requirements: 7.2, 7.3, 7.4_

- [ ] 3.5 (P) sdist から開発のためだけのファイルを外す設定を足す
  - design の SdistManifest の中身で、リポジトリの最上位に `MANIFEST.in` を足す（AI のエージェントの設定と仕様、GitHub とエディタの設定、最上位の dotfile、手元の検査と examples の同期の設定を外す。`global-exclude` は使わない）
  - `docs_src/`・`tests/`・`examples/`・`CHANGELOG.md` などは残す
  - setuptools-scm を入れずに確かめる: yikit-dev の setuptools の `FileList` で、`git ls-files` の一覧に `MANIFEST.in` の各行を当てる
  - 完了の状態: その試しで、外すもの（design の SdistManifest の一覧）が残らず、最上位が `CHANGELOG.md`・`LICENSE`・`README.md`・`docs_src`・`examples`・`pyproject.toml`・`src`・`tests` だけで、`src/yikit` のファイルの数が `git ls-files src/yikit` と同じ。結果のファイルの数を Implementation Notes に書く
  - _Boundary: SdistManifest_
  - _Requirements: 12.1, 12.2, 12.3_

- [ ] 4. 手元でのビルドと版の確認（作者の承認が要る）
- [ ] 4.1 作者の承認を得て setuptools-scm を入れ、editable を入れ直して、テストと検査を通す
  - 作者に、yikit-dev に setuptools-scm（10.1 以上。依存の vcs-versioning を含む）を入れる承認を求める。承認がなければ 4.1〜4.3 を行わずに 5 と 6 に進む。そのときは、手元で確かめられなかった要件（2.4、3.4 の手元での確認、3.6、9.3、12.1〜12.3 の本物の sdist での確認（3.5 の試しは済んでいる）、および 4.2 の版の対応）を最後の報告に書く（6.1・6.2 の CI と手動実行で確かめられるのは一部だけ）。build isolation を使うビルドやインストール（setuptools-scm を取りに行く）は、承認がない限り実行しない。承認がないときは 4.1〜4.3 のチェックを付けずに残し、理由を Implementation Notes に書く
  - 入れ直しは build isolation を使わず、依存も入れ直さない（`--no-build-isolation --no-deps`）。他のダウンロードを起こさない
  - 完了の状態: `src/yikit/_version.py` ができて `git status` に出ず、`python -m pytest -ra` の全体が通って、1.1 の基準に `tests/test_version.py` のテストを足した数になり、`ruff check src tests`・`ruff format --check src tests`・`mypy` がエラー 0 件
  - _Requirements: 3.4, 3.6, 8.1, 8.2, 9.2, 11.1_

- [ ] 4.2 使い捨ての clone で、tag とコミットから決まる版を確かめる
  - 1〜3 の変更をコミットしてから、作業ブランチの HEAD を scratchpad に clone する（コミットしていない変更は clone に入らない）。その clone だけに、同じコミットへ `v0.5.0-rc.1`（annotated）と `v0.5.0`（lightweight）を付ける。本物のリポジトリには tag を付けない
  - actions/checkout と同じく、push した tag だけを浅く取った clone から `python -m build --no-isolation` でビルドし、`0.5.0rc1` と `0.5.0` の sdist と wheel ができることを確かめる
  - tag の後にコミットを足した clone で、上書きの環境変数なしでは `+g<hash>` の付いた開発版、ありではローカル版のない開発版になり、追跡しているファイルを変えたままだと上書きありのビルドが失敗することを確かめる
  - 完了の状態: 各ビルドの配布物のファイル名（または失敗）が Implementation Notes に書かれ、design の「tag と版の対応」と一致する
  - _Requirements: 3.1, 3.2, 3.3, 5.4, 5.5, 6.2, 11.1, 11.3_

- [ ] 4.3 配布物のメタデータと中身を、0.4.0 の配布物と比べて確かめる
  - 4.2 の `v0.5.0` の wheel の METADATA に、`Metadata-Version: 2.4`、`License-Expression: Apache-2.0`、`License-File: LICENSE`、`Requires-Python: >=3.10`、3.10〜3.14 の classifiers があり、`typing_extensions` がないことを確かめる。ビルドの log にライセンスの非推奨の警告がないことも確かめる
  - 同じ clone の `v0.4.0` から作った wheel とファイルの一覧を比べ、違いが版のファイルとライセンスのファイルだけであることを確かめる（`v0.4.0` には sdist を絞る設定がないので、その設定が wheel を変えないことの確認にもなる）
  - `v0.5.0` の sdist の中身の一覧を取り、design の SdistManifest で外すものがなく、残すもの（生成された版のファイルと、テストの参照画像を含む）があることを確かめる。sdist から wheel のビルドが通ったことも確かめる
  - その sdist を scratchpad に展開し、展開したテストで pytest を回して、4.1 と同じ数が通ることを確かめる（テストに要るものが sdist にある）
  - 完了の状態: METADATA の該当の行、wheel の一覧の差、sdist の中身の一覧（最上位と、外したもの・残したものの照らし合わせ）、展開した sdist での pytest の数が Implementation Notes に書かれ、design の BuildMetadata と SdistManifest の契約どおり
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.4, 9.3, 11.1, 12.1, 12.2, 12.3_

- [ ] 5. steering の tech.md の該当の節を 0.5.0 に合わせ、README を確かめる
  - Core Technologies の Language、型、コードの品質、テストの CI の項目、Development Environment、Release を design の PackagingSteering のとおりに書き換える（tag と版番号の対応、手動実行の扱い、editable の入れ直し、optional の依存なしの job とテストで `importorskip` を使うこと、必須の checks を設定するときの注意、sdist を絞る設定と、最上位にファイルを足したときの見直し、workflow のファイル名と trusted publishing の関係を含む）。`updated_at` を更新する
  - 型の節の「新しい版の numpy などにしかない型は `TYPE_CHECKING` の中で import する」は残す（drop-legacy-compat の互換性の節が参照する）
  - Core Technologies の Framework・Core deps、互換性、Key Libraries、Key Technical Decisions、その他の節と `product.md` には触らない（drop-legacy-compat が持つか、どちらの spec も触らない）
  - README に対応する Python の版を書いた文がないことを確かめる（あれば 3.10 以上に直す）
  - 完了の状態: tech.md の対象の節に Python 3.8・`ubuntu-24.04` の固定・`setuptools>=61`・手で書く版番号の記述がなく、テストの CI の項目に optional の依存なしの job があり、型の節に `TYPE_CHECKING` の決まりが残り、`git diff` で Framework・Core deps の行、互換性・Key Libraries・Key Technical Decisions の節と `product.md` に変更がない
  - _Requirements: 10.1, 10.2, 10.3_

- [ ] 6. 統合の確認
- [ ] 6.1 作業ブランチで CI を手動実行し、Python 3.10〜3.14 と optional の依存なしの job が通ることを確かめる
  - 作業ブランチを push し、CI を手動実行する。変更の判定の手順が飛ばされ、テストの 5 つの job と optional なしの job が走ることを確かめる
  - optional なしの job の `pip install` の log から、optional の依存（optuna・lightgbm・ngboost・Boruta・tqdm・colorlog）が入っておらず、pandas が 3 系であることを確かめる
  - 失敗したら原因を直す。コードの振る舞いを変える必要があるときは作者に相談する
  - 完了の状態: その run で、テストの 5 つの job（3.10〜3.14）と `test-no-optional` の 6 つがすべて成功し、`test-no-optional` の check の名前が `test (3.x)` とぶつからない。run の ID と、optional なしの job の pandas の版が Implementation Notes に書かれている
  - _Depends: 2.3, 3.1_
  - _Requirements: 4.2, 4.3, 4.5, 4.9, 4.10, 9.2_

- [ ] 6.2 作者の確認を得て release を手動実行し、TestPyPI に開発版が出ることを確かめる
  - 作者に、作業ブランチで release workflow を手動実行して TestPyPI に開発版を出してよいか確認する。確認が得られなければ行わず、その旨を報告し、Implementation Notes の「渡す確認」に残す
  - build の log の版がローカル版のない開発版で、TestPyPI の JSON API でその版の `requires_python` が `>=3.10`、`license_expression` が `Apache-2.0` であり、PyPI と GitHub Release の job が飛ばされたことを確かめる
  - 同じコミットでもう 1 回手動実行し、TestPyPI の job が同じ版を飛ばして成功することを確かめる
  - 完了の状態: 2 つの run の ID と、公開した版、TestPyPI の値が Implementation Notes に書かれている
  - _Depends: 3.2_
  - _Requirements: 5.5, 5.6, 5.9, 11.2_

## Implementation Notes
- 作者の決定（2026-10-14、実装の開始時）:
  - 4.1〜4.3 は build isolation を使う。一時的なビルド用の環境に setuptools・setuptools-scm をダウンロードするのは承認済み。yikit-dev には setuptools-scm を入れず、変えるのは yikit の editable の入れ直し（`pip install -e . --no-deps`、isolation あり）だけ。手元のビルドは `python -m build`（isolation あり）。タスクの本文の `--no-build-isolation`・`--no-isolation` の代わりにこれを使う
  - 6.1 は、作業の最後に開く PR の CI で確かめる（手動実行の代わり）
  - 6.2 は行わない。TestPyPI への公開は release-0.5.0 の `v0.5.0-rc.1` の tag で確かめる（手動実行の経路は確かめないまま残る）
- 共通: 何かを入れる・ダウンロードする前（scratchpad への `pip download` や `pip install --target` も含む）に作者の承認を得る。setuptools-scm を入れるまで `pip install -e .` を実行しない（build isolation で setuptools-scm を取りに行くため）
- 共通: 本物のリポジトリに tag を付けない。docs の workflow を手動実行しない（`force_orphan` で公開中の docs を書き換える）
- 共通: drop-legacy-compat はこの spec の後に、同じ `pyproject.toml`・`tech.md`・`src/yikit/visualize/_yyplot.py` と、2.3 で触る 3 つのテストのファイルを触る。この spec では design の Boundary Commitments の行と節だけを触る。drop-legacy-compat は 4.1（editable の入れ直し）の手順を参照している（番号を変えない）
- 1.1 の基準（2026-10-14、変更の前、yikit-dev の Python 3.12.15・pytest 9.1.1）: `python -m pytest -ra -p no:cacheprovider` で 700 collected、700 passed・0 skipped・0 failed・0 error・0 xfail・542 warnings（optional の依存がすべて入った環境）
- 1.3・4.1: リポジトリの最上位に古い `yikit.egg-info`（0.3.15、gitignore されている）があり、最上位で `python -m pytest` や `python -c` を実行すると `importlib.metadata.version("yikit")` がこれを読む。版の比較は `yikit._version` とする
- 2.2: `ruff.toml` では F401 が unfixable。自動修正の後に残る `import sys` は手で消す（`--fixable F401` を CLI で付けてもよい）。UP036 の修正は `--unsafe-fixes` が要る
- 2.3: 塞ぎ方は `sys.modules[name] = None`（`import` は `ModuleNotFoundError`、`importlib.util.find_spec` は None を返す）。`PYTHONDONTWRITEBYTECODE=1` と `-p no:cacheprovider` を付け、リポジトリに cache を残さない。2026-10-14 の変更の前は、3 つのファイルが集める段階で失敗し、残りは 296 passed・102 skipped だった。`importorskip` の後の import には `# noqa: E402` を付ける
- 3.x: 今の release workflow には actionlint（shellcheck SC2086、info）の指摘が 2 件ある。`"$GITHUB_OUTPUT"` で消える。scratchpad に置いた design どおりの下書きは actionlint の指摘 0 件だった
- 3.2: 上書きの環境変数は `SETUPTOOLS_SCM_OVERRIDES_FOR_YIKIT: '{local_scheme = "no-local-version-strict"}'`。`fetch-depth` の式は `${{ startsWith(github.ref, 'refs/tags/') && 1 || 0 }}`（0 は偽なので、`&& 0 || 1` の順にしない）
- 3.4: この spec が main に入った時点では、ngboost の行はまだ `<0.5.9; python_version < '3.14'` なので、dependabot が 0.5.9・0.5.10 まで広げる PR を出すことがある。`ignore` はそれを止めない（drop-legacy-compat がタスク 1 で今の行を読み直す）
- 3.5: `FileList` は `from setuptools._distutils.filelist import FileList`。`git ls-files` の一覧を `extend` し、`MANIFEST.in` のコメントでない各行を `process_template_line` に渡す。2026-10-14 の追跡しているファイルでは 168 個が 66 個になった。本物の sdist には、これに `PKG-INFO`・`setup.cfg`・`MANIFEST.in`・`src/yikit.egg-info/`・`src/yikit/_version.py` が加わる
- 4.2・4.3: 浅い clone は `git init` の後に `git fetch --no-tags --depth=1 file://<clone> +refs/tags/<tag>:refs/tags/<tag>` と `git checkout <tag>` で作る（ローカルのパスでは `--depth` が効かないので `file://` を使う）。`v0.4.0` の wheel は、その tag を checkout した clone で `python -m build --wheel --no-isolation` すれば、yikit-dev の setuptools 84 だけで作れる。展開した sdist でのテストは `python -m pytest -ra -p no:cacheprovider <展開した場所>/tests` で、yikit は yikit-dev の editable のものが読まれる
- 作者か release-0.5.0 に渡す確認（この spec のタスクにしない）:
  - テストを飛ばす判定（4.1・4.4）: PR の差分全体を比べるので、マージの前には観察できない。マージの後、コード以外のファイルだけを変える main への最初の push（例: release-0.5.0 の CHANGELOG）で、`test` と `test-no-optional` が skipped・実行が success になることを見る。この spec の中では、3.1 の静的な確認（`on` に `paths` がなく、テストの job に `if` がある）で代える
  - tag の push での公開（5.1〜5.4）: rc の tag（`v0.5.0-rc.1`）で PyPI・TestPyPI・pre-release の GitHub Release、正式版の tag で PyPI・TestPyPI・GitHub Release に出ること（release-0.5.0 が行う）
  - docs の版（6.1・6.2）: 正式版の tag で公開された docs の版が tag の版であること（release-0.5.0 の v0.5.0 で見る）
  - dependabot の group（7.2）: GitHub Actions の部品の新しい版が出たとき、PR が 1 つにまとまること
  - TestPyPI の手動実行（5.5・5.6・11.2）: 6.2 を作者が断ったときは、作者が後で行う
- release-0.5.0 の CHANGELOG の材料（利用者に見える変化）:
  - Python 3.10 以上が必要（3.8・3.9 では pip が 0.4.x を選ぶ）
  - 版番号は setuptools-scm が git の tag から決める（`vX.Y.Z` → `X.Y.Z`、`vX.Y.Z-rc.N` → `X.Y.ZrcN`）。`__version__` はインストールのときに作られる `_version.py` から読み、インストールしていないソースでは `"unknown"`。main から入れると、次の tag までは `X.Y.(Z+1).devN` の版になる
  - ライセンスを PEP 639 の式（`License-Expression: Apache-2.0`）で宣言し、`typing_extensions` の依存をなくした
  - sdist から開発のためだけのファイル（`.kiro/`・`.claude/`・`CLAUDE.md`・`.github/`・`.vscode/`、開発の設定と dotfile）を外した。`tests/`・`examples/`・`docs_src/` は残る。wheel の中身は変わらない
  - CI に、Python 3.14 で optional の依存を入れずにテストする job を足した（optional の依存がないときの動作と pandas 3 を確かめる）
