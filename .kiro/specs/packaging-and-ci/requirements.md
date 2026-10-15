# Requirements Document

## Project Description (Input)
yikit 0.4.x は、作者の古いオフライン環境（Python 3.8）のために、古いビルドの書き方を続けている。`setuptools>=61` と表の形の `license`（新しい setuptools では 2027-02-18 に期限が切れる非推奨の書き方）を使い、版番号は `src/yikit/__init__.py` に手で書くので、リリースのたびに版を上げる PR が要り、tag と版がずれるおそれがある。CI はサポートの終わった 3.8・3.9 を含む 7 つの版を回し、そのために runner を `ubuntu-24.04` に固定している。

0.5.0 では Python の下限を 3.10 に上げ、ビルドと CI を作者の python-template にそろえる。PEP 639 の `license` と `setuptools>=77.0.3` を使い、版番号は setuptools-scm で git の tag から決める（`vX.Y.Z` は `X.Y.Z`、`vX.Y.Z-rc.N` は `X.Y.ZrcN`）。CI・release・docs の workflow と dependabot を python-template にそろえ、ruff と mypy の対象を 3.10 にして、3.8・3.9 向けの書き分けを消す。steering の `tech.md` を 0.5.0 に合わせる。2026-10-14 の作者の決定で、CI に optional の依存を入れない Python 3.14 の job を 1 つ足し、dependabot が ngboost の 0.5.11 以上を提案しないようにし、`MANIFEST.in` で sdist から開発のためだけのファイルを外す。ライブラリの依存の下限と古い版のための分岐は drop-legacy-compat が持つ。詳しい背景は `brief.md` を参照。

## Introduction
yikit の作者が、0.5.0 から Python 3.10 以上だけを対象にし、tag を push するだけで正しい版番号の配布物を公開できるようにする。CI・release・docs の workflow と dependabot は python-template の書き方にそろえ、手元の ruff と mypy は Python 3.10 を対象にする。この spec は公開 API と計算の結果を変えない。変わるのは、入れられる Python の版、版番号の決まり方、sdist の中身、ビルドと workflow の設定、手元の検査の対象の版である。Python 3.8・3.9 の環境は 0.4.x の系列が受け持つ。

## Boundary Context
- **In scope**: 対応する Python の版の宣言、ライセンスとビルドの要件の宣言、版番号の決め方と `yikit.__version__`、sdist の中身、CI・release・docs の workflow（optional の依存なしでテストする job を含む）と、optional の依存がない環境でテストが読み込めること、dependabot（ngboost の 0.5.11 以上を提案から外すことを含む）、手元の ruff と mypy の対象の版と、Python 3.8・3.9 向けの書き分けの削除、steering（`tech.md`）のうち Python・型・コードの品質・CI・Release・開発環境の記述、README の Python の版の記述
- **Out of scope**: `pyproject.toml` の `[project] dependencies` と `optional` の extra にあるライブラリの行（下限・上限・ngboost の条件）と、ライブラリの古い版のための分岐とそのテスト（drop-legacy-compat）。`test`・`dev`・`docs` の extra は、この spec も drop-legacy-compat も変えない。ngboost 0.5.11 の `set_params`・`get_params` の変更への対応（2026-10-14 に作者が後の版に送ると決めた）。CHANGELOG と tag の作成とリリース（release-0.5.0）、下限の版を入れて試す CI の job（作らないと 2026-10-14 に決めた）、0.4.x の保守用ブランチ、CI での ruff・mypy の実行、PyPI・TestPyPI の trusted publishing の登録の変更
- **Adjacent expectations**:
  - drop-legacy-compat は、この spec の Python 3.10〜3.14 の CI（optional の依存なしの 3.14 の job を含む）と、3.10 を対象にした ruff・mypy の上で作業する。`pyproject.toml` の `[project] dependencies` と `optional` の extra のライブラリの行は drop-legacy-compat が触る。この spec が依存の行で触るのは、Python 3.10 未満のための `typing_extensions` の行だけ
  - drop-legacy-compat は ngboost の上限を `<0.5.11` にする（2026-10-14 の作者の決定）。この spec は、その決定に合わせて dependabot が ngboost の 0.5.11 以上を提案しないようにする
  - release-0.5.0 は、この spec の後に tag の push だけで 0.5.0 を出す。版を上げるコミットは作らない
  - PyPI と TestPyPI の trusted publishing は、今の release workflow のファイル名と環境の名前（`pypi`・`testpypi`）で登録されている。この spec はそれを前提にし、登録を変えない

## Requirements

### Requirement 1: 対応する Python の版
**Objective:** As yikit の作者, I want 0.5.0 から Python 3.10 以上だけを対象にすること, so that サポートの終わった版のための書き分けと CI の手間をなくせる

#### Acceptance Criteria
1. The yikit のパッケージ情報 shall 対応する Python の版を 3.10 以上として宣言する
2. The yikit のパッケージ情報 shall 分類（classifiers）に Python 3.10・3.11・3.12・3.13・3.14 を示し、3.8・3.9 を示さない
3. If Python 3.9 以下の環境で yikit を入れようとしたとき, the yikit のパッケージ情報 shall pip に 0.5.0 以降の版を選ばせない（その環境では 0.4.x の系列が選ばれる）
4. The yikit のパッケージ情報 shall Python 3.10 未満のためだけの依存（`typing_extensions`）を宣言しない
5. The yikit のコード shall Python 3.8・3.9 で動かすためだけの書き分け（Python の版による import の分岐と、3.8 を外すときに直すと書いた注記）を持たない

### Requirement 2: ライセンスとビルドの要件の宣言
**Objective:** As yikit の作者, I want ライセンスを今の標準の書き方で宣言すること, so that ビルドの道具の非推奨の期限（2027-02-18）に左右されずに配布物を作れる

#### Acceptance Criteria
1. The yikit のパッケージ情報 shall ライセンスを SPDX の式 `Apache-2.0` で宣言する
2. When 配布物（sdist と wheel）をビルドしたとき, the 配布物 shall ライセンスの本文のファイル（`LICENSE`）を含み、メタデータにライセンスの式とそのファイルを示す
3. The yikit のビルドの要件 shall ライセンスの式を受け付ける版のビルドの道具を求め、それより古い版ではビルドされないようにする
4. When 配布物をビルドしたとき, the ビルド shall ライセンスの書き方についての非推奨の警告を出さない

### Requirement 3: 版番号を git の tag から決める
**Objective:** As yikit の作者, I want 版番号が git の tag から決まること, so that 版を上げる PR なしで、tag と版がずれずにリリースできる

#### Acceptance Criteria
1. When `vX.Y.Z` の tag の付いたコミットから配布物をビルドしたとき, the ビルド shall 版 `X.Y.Z` の配布物を作る（例: `v0.5.0` から `0.5.0`）
2. When `vX.Y.Z-rc.N` の tag の付いたコミットから配布物をビルドしたとき, the ビルド shall 版 `X.Y.ZrcN` の配布物を作る（例: `v0.5.0-rc.1` から `0.5.0rc1`）
3. When 最後の tag より後のコミットから配布物をビルドしたとき, the ビルド shall 最後の tag の版より新しい開発版の版番号を付ける
4. When 配布物から、または手元のソースから yikit を入れたとき, the yikit shall `yikit.__version__` に、入れたときの版番号と同じ文字列を返す
5. If 版の情報が作られていない場所（一度もビルドもインストールもしていないソースのツリー）から yikit を import したとき, the yikit shall import に失敗せず、`yikit.__version__` に版が分からないことを示す文字列を入れる
6. The リポジトリ shall ビルドやインストールのときに作られる版の情報のファイルを git で追跡しない
7. The yikit のソース shall 版番号を手で書いた箇所を持たず、リリースに版番号を書き換えるコミットを必要としない

### Requirement 4: CI でのテスト
**Objective:** As yikit の作者, I want 変更のたびにテストが Python 3.10〜3.14 で自動で確かめられること, so that 壊れた変更を main に入れずに済む

#### Acceptance Criteria
1. When main への push があったとき、または pull request が作成・更新されたとき, the CI shall 変更にテストに関わるファイル（コード、テスト、パッケージの宣言、CI の設定）が含まれるかを判定する
2. When テストに関わるファイルが変わっていたとき, the CI shall Python 3.10・3.11・3.12・3.13・3.14 のそれぞれで、test と optional の依存を入れてテストを実行する
3. When CI を手動で実行したとき, the CI shall 変更の内容にかかわらず、Python 3.10〜3.14 のすべてでテストを実行する
4. If テストに関わるファイルが変わっていないとき, the CI shall テストを飛ばし、その実行を成功として完了する（workflow が起動されずに checks が待ちのまま残ることがない）
5. If ある Python の版でテストが失敗したとき, the CI shall 他の版のテストを中断せずに最後まで実行し、実行の全体を失敗として報告する
6. The CI shall リポジトリの内容と pull request を読む権限だけで動き、書き込みの権限を持たない
7. The CI shall Python の版のために runner の OS の版を固定せず、GitHub が標準とする Ubuntu の runner で動く
8. The CI shall ruff と mypy を実行しない（作者が手元で実行する）
9. When テストに関わるファイルが変わっていたとき、または CI を手動実行したとき, the CI shall Python 3.14 で、optional の依存を入れずに test の依存だけを入れてテストを実行する job を、Python の版ごとのテストの job（`test (3.x)`）とは別の、ぶつからない名前で実行する
10. If optional の依存が入っていない環境でテストを実行したとき, the yikit のテスト shall optional の依存を使うテストだけを skip し、テストの module の読み込みで失敗せずに、残りのテストを実行して通す

### Requirement 5: リリースの workflow
**Objective:** As yikit の作者, I want tag を push するだけで正しい版番号の配布物が公開されること, so that リリースの手順を減らし、版の誤りを防げる

#### Acceptance Criteria
1. When `vX.Y.Z` の tag が push されたとき, the release workflow shall 版 `X.Y.Z` の配布物を PyPI と TestPyPI に公開し、署名した配布物を添えた GitHub Release を作る
2. When `vX.Y.Z-rc.N` の tag が push されたとき, the release workflow shall 版 `X.Y.ZrcN` の配布物を PyPI と TestPyPI に公開し、pre-release の GitHub Release を作る
3. When rc 以外の `-` を含む tag（例: `vX.Y.Z-beta.N`）が push されたとき, the release workflow shall PyPI には公開せず、TestPyPI への公開と pre-release の GitHub Release だけを行う
4. When 同じコミットに rc の tag と正式版の tag の両方があるとき, the release workflow shall push された tag の版番号で配布物を作る
5. When release workflow を tag ではない ref で手動実行したとき, the release workflow shall ローカル版の部分（`+` 以降）を含まない開発版の版番号で配布物をビルドし、TestPyPI だけに公開する（PyPI と GitHub Release には出さない）
6. If 手動実行で作った版がすでに TestPyPI にあるとき, the release workflow shall 失敗せずに、その公開を飛ばす
7. While release workflow が実行中のとき, when 同じ ref で release workflow が再び起動されたとき, the release workflow shall 実行中のものを取り消さずに最後まで実行する
8. The release workflow shall 既定の権限をリポジトリの内容を読むことだけにし、公開と GitHub Release の作成に要る権限は、それを行う job だけに与える
9. The release workflow shall PyPI と TestPyPI の trusted publishing の今の登録（workflow のファイル名、環境の名前 `pypi`・`testpypi`）のままで公開できる

### Requirement 6: docs の workflow
**Objective:** As yikit の利用者, I want 公開された docs に正しい版番号が付くこと, so that 読んでいる docs がどの版のものか分かる

#### Acceptance Criteria
1. When 正式版の tag（`vX.Y.Z`）が push されたとき, the docs workflow shall その tag の版番号を付けた docs を作り、GitHub Pages に公開する
2. When 正式版の tag と同じコミットに rc の tag もあるとき, the docs workflow shall 正式版の版番号を付ける

### Requirement 7: workflow の部品と依存の更新
**Objective:** As yikit の作者, I want workflow で使う GitHub Actions の部品が新しく、更新の提案も自動で来ること, so that 古い部品のまま放置せずに済む

#### Acceptance Criteria
1. When この spec の変更が入ったとき, the workflow（CI・release・docs）shall 2026-10-14 の時点で最新の major 版の GitHub Actions の部品を使う
2. When workflow で使う GitHub Actions の部品に新しい版が出たとき, the dependabot shall それらの更新を1つの pull request にまとめて提案する
3. The dependabot shall pip の依存の更新の方針（宣言の範囲内の新しい版では pull request を作らない）を変えない
4. The dependabot shall ngboost の 0.5.11 以上の版への更新を提案せず、その設定に、外す理由（drop-legacy-compat が決めた上限 `<0.5.11`）を示す注記を持つ

### Requirement 8: 手元での静的検査と型検査の対象
**Objective:** As yikit の開発者, I want ruff と mypy が Python 3.10 を対象に検査すること, so that 3.10 以上で使える書き方に揃い、古い版向けの書き分けが残らない

#### Acceptance Criteria
1. When 開発者が手元で `src/` と `tests/` に ruff の静的検査と書式の検査を実行したとき, the 検査 shall Python 3.10 を対象にして、エラーを0件で終える
2. When 開発者が手元で mypy を実行したとき, the 型検査 shall Python 3.10 を対象にして `src/yikit` を検査し、エラーを0件で終える
3. The 3.10 を対象にした書き換え shall Python の版による書き分けだけを変え、ライブラリの版による書き分けを変えない

### Requirement 9: 振る舞いを変えないこと
**Objective:** As yikit の利用者, I want この変更で yikit の使い方や結果が変わらないこと, so that 0.5.0 に上げても同じコードが同じ結果を返す

#### Acceptance Criteria
1. The packaging-and-ci の変更 shall 公開 API の追加・削除・名前と引数の変更を含まない
2. When Python 3.10〜3.14 で、同じ入力と同じ乱数の種で公開 API を呼んだとき, the yikit shall この spec の変更の前と同じ結果を返す
3. When wheel をビルドしたとき, the wheel shall 版の情報のファイルとライセンスのファイルを除き、変更の前と同じモジュールを含む

### Requirement 10: steering と README
**Objective:** As yikit の開発者と AI のエージェント, I want steering と README が 0.5.0 の決まりを示すこと, so that 後の作業（drop-legacy-compat、release-0.5.0）が古い前提で進まない

#### Acceptance Criteria
1. When この spec の変更が入ったとき, the steering（`tech.md`）shall Python の版、型の決まり、コードの品質（ruff・mypy の対象）、CI（optional の依存なしの job を含む）、Release の手順（tag と版番号の対応、手動実行の扱い、ビルドの要件、sdist の中身）、開発環境の手順を 0.5.0 の内容で示す
2. The steering の更新 shall `tech.md` のうち Core Technologies の Language の行、型、コードの品質、テストの CI の項目、Development Environment、Release の節だけを変え、Core Technologies の Framework・Core deps の行、互換性、Key Libraries、Key Technical Decisions の節と `product.md` を変えない（drop-legacy-compat が持つ。Key Technical Decisions は「optional extra に重い依存を寄せる」の理由の句だけを drop-legacy-compat が直し、他の項目はどちらも触らない）
3. Where README に対応する Python の版の記述があるとき, the README shall 3.10 以上と示す

### Requirement 11: 環境と公開に影響する確認の扱い
**Objective:** As yikit の作者, I want 手元の環境や公開先に影響する確認が、自分の承認のもとで行われること, so that 意図しないインストールや公開が起きない

#### Acceptance Criteria
1. If 確かめるために手元の環境へパッケージを入れる（scratchpad へのダウンロードや `--target` への展開も含む）必要があるとき, the 開発の手順 shall 作者の承認を得てから行う
2. If 確かめるために release workflow を手動実行して TestPyPI に公開するとき, the 開発の手順 shall 作者の確認を得てから行う
3. The packaging-and-ci の作業 shall tag の作成とリリースを行わず、docs の公開先を書き換える手動実行もしない

### Requirement 12: sdist の中身
**Objective:** As yikit の作者, I want sdist に開発のためだけのファイルが入らないこと, so that AI のエージェントの指示や CI・エディタの設定を配らず、利用者と再配布する人に要るものだけを出せる

#### Acceptance Criteria
1. When sdist をビルドしたとき, the sdist shall 開発のためだけのファイル（`.kiro/`・`.claude/`・`CLAUDE.md`、`.github/`、`.vscode/`、リポジトリの最上位の `.gitignore`・`.gitattributes`・`.python-version`、`ruff.toml`・`mypy.ini`・`jupytext.toml`）を含まない
2. When sdist をビルドしたとき, the sdist shall パッケージのソース（`src/` と生成された版のファイル）、`tests/`（参照画像を含む）、`examples/`、`docs_src/`、`LICENSE`、`README.md`、`CHANGELOG.md`、`pyproject.toml` を含む
3. When sdist から wheel をビルドしたとき, the ビルド shall 成功し、wheel の中身は sdist の中身を絞る設定によって変わらない
