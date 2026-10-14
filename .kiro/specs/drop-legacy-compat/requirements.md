# Requirements Document

## Project Description (Input)
yikit 0.4.x は、作者の古いオフライン環境（Python 3.8、scikit-learn 0.24.1、ngboost 0.3.6、lightgbm 3.2.1）で動くように依存の下限を低く保ち、ライブラリの古い版を見分ける分岐と、それを確かめるテストを持っている（scikit-learn<1.0、LightGBM 3、ngboost 0.3、optuna 3.0〜3.6 と古い optuna-integration、seaborn<0.12）。0.5.0 では packaging-and-ci が Python の下限を 3.10 に上げる。Python 3.10 で入るライブラリはこれらの古い版より新しいので、分岐は保守の手間になるだけである。ngboost の指定（`ngboost>=0.3.6,<0.5.9; python_version < '3.14'`）も、Python 3.14 に対応した ngboost 0.5.9〜0.5.11 が出ている今の実態からずれている。

この spec では、古い版のための分岐と、それだけを確かめるテストを消し、依存の下限を、消した分岐が前提にしていた版の分だけ上げる（作者が 2026-10-14 に決定）。ngboost は Python 3.14 での除外をやめ、上限は新しい版での振る舞いを確かめてから決める。steering の `tech.md`（互換性・Key Libraries・Framework・Core deps と、Key Technical Decisions の「古い環境や軽い用途で」の句）と `product.md` を 0.5.0 の方針に書き換え、0.5.0 の CHANGELOG の材料を残す。詳しい背景は `brief.md` を参照。

## Introduction
yikit の作者と利用者（licond など）のために、0.5.0 でライブラリの古い版のための分岐をなくし、依存の下限と上限を「残るコードが実際に使う機能のある版」にそろえる。宣言した範囲の版では、公開 API と計算の結果を変えない。変わるのは、入れられるライブラリの版の範囲（scikit-learn・seaborn・lightgbm・ngboost・optuna・optuna-integration の下限と ngboost の上限）と、Python 3.14 でも ngboost が入ることである。古い環境（Python 3.8 など）は 0.4.x の系列が受け持つ。

## Boundary Context
- **In scope**: `pyproject.toml` の `dependencies` と、`optional-dependencies` の `optional` の extra にあるライブラリの下限・上限と ngboost の環境の条件（`test`・`dev`・`docs` の extra は変えない）。`src/yikit` のうち、ライブラリの古い版のための分岐と、その版に触れた docstring・コメント。それだけを確かめるテストと、テストの中の版の分岐・注記。steering の `tech.md` の互換性・Key Libraries・Core Technologies の Framework と Core deps、Key Technical Decisions の「古い環境や軽い用途で」の句、`product.md`。0.5.0 の CHANGELOG の材料
- **Out of scope**: Python の版、ビルド、版番号、workflow（Python 3.14 で test の依存だけを入れる CI の job を含む）、`.github/dependabot.yml`（ngboost の 0.5.11 以上を無視する `ignore` の行を含む）、ruff・mypy の対象の版、Python 3.8・3.9 向けの書き分け（packaging-and-ci）。`pyproject.toml` の `test`・`dev`・`docs` の extra（どちらの spec も変えない）。新しい機能と公開 API の変更。下限の版を入れて試す CI の job（作らないと 2026-10-14 に決定）。CHANGELOG の本文、tag、リリース（release-0.5.0）。上限の外の ngboost に yikit を合わせるためのコードの変更（ngboost 0.5.11 への対応は後のリリースに回すと作者が 2026-10-14 に決定）。optional の依存がないときに `import yikit` が失敗しない仕組みの変更。下限以上の版どうしの違いのための分岐（残す）
- **Adjacent expectations**:
  - packaging-and-ci が main に入った後に作業する。Python 3.10〜3.14 の CI、Python 3.14 で test の依存だけを入れて pytest を回す CI の job（pandas 3 と、optional の依存がないときの動作を受け持つ。作者が 2026-10-14 に決定）、3.10 を対象にした ruff・mypy、`typing_extensions` の行が消えた `pyproject.toml` を前提にする。`pyproject.toml` と `tech.md` は packaging-and-ci と共有し、この spec はライブラリの依存の行と、上に書いた `tech.md` の節と句だけを触る
  - 手元の yikit-dev は、packaging-and-ci の後に editable で入れ直してあること（`src/yikit/_version.py` がある）を前提にする。入れ直していなければ、作者の承認を得てから入れ直す
  - release-0.5.0 は、この spec が残す材料を使って、上げた下限・上限とその理由を CHANGELOG に書く
  - licond など yikit 0.5.0 を求める側は、上げた下限も満たす必要がある

## Requirements

### Requirement 1: 依存の下限
**Objective:** As yikit の作者, I want 依存の下限が、残るコードが実際に使う機能のある版を示すこと, so that 宣言と実態がずれず、要らない下限で利用者の環境を縛らない

#### Acceptance Criteria
1. The yikit のパッケージ情報 shall 古い版のための分岐を消したライブラリ（scikit-learn・seaborn・lightgbm・ngboost・optuna・optuna-integration）について、残るコードが使う機能が入った最初の版を下限として宣言する
2. The yikit のパッケージ情報 shall 消す分岐のないライブラリ（numpy・pandas・scipy・matplotlib・joblib など）に新しい下限を宣言せず、今ある下限（Boruta）を変えない
3. The 宣言する下限 shall それぞれ、使う機能がその版にあることを、リリースノート・changelog・その版のソースのいずれかで確かめたものにする
4. When Python 3.10〜3.14 のそれぞれで test と optional の依存を指定して yikit を入れたとき, the 依存の解決 shall 宣言した下限と上限を満たす版の組み合わせで成功する

### Requirement 2: ngboost の条件
**Objective:** As yikit の利用者, I want Python 3.14 でも ngboost を使う機能が入り、入る ngboost の版で yikit が正しく動くこと, so that Python の版によって使える機能が欠けず、ngboost の更新で結果が黙って変わらない

#### Acceptance Criteria
1. The yikit のパッケージ情報 shall Python の版によって ngboost を除外せず、Python 3.10〜3.14 のすべてで optional の依存として ngboost を入れる
2. The yikit のパッケージ情報 shall ngboost の上限を、`log_likelihood` と、yikit が頼る ngboost の振る舞い（`set_params`・`get_params` での入れ子の名前の扱い）が変わらないことを確かめた版までにする
3. When Python 3.14 で test と optional の依存を入れてテストを実行したとき, the ngboost を使うテスト shall skip されずに実行され、通る
4. If 確かめた範囲より新しい ngboost で yikit の振る舞いが変わることが分かったとき, the 開発の記録 shall その版を上限で外した理由と、対応に要る作業を残す

### Requirement 3: 古い版のための分岐の削除
**Objective:** As yikit の開発者, I want 宣言した下限より古い版のためだけの分岐がコードにないこと, so that 保守する分岐が減り、どの版を対象に書いたコードかが読み取れる

#### Acceptance Criteria
1. The yikit のコード shall 宣言した下限より古いライブラリの版でだけ通る処理（版の番号や引数・属性の有無による判定、古い版の場所からの import、古い版の振る舞いを補う上書き）を持たない
2. The yikit のコード shall 下限以上の版どうしの違いに対する分岐を残す（例: scikit-learn 1.6 の estimator の tags、scikit-learn 1.9 の木の `criterion`、LightGBM 4.7 の `eval_X`・`eval_y`）
3. The yikit の docstring とコメント shall 宣言した下限より古い版について書いた記述を持たず、下限以上の版の違いの記述を残す
4. Where 同じ判定が複数のモジュールにあり、古い版の分岐を消すと同じ処理になるとき, the yikit のコード shall その判定を 1 か所に持ち、他のモジュールはそれを使う
5. The yikit shall optional の依存が入っていなくても `import yikit` とサブパッケージの import に失敗しない作りを保つ

### Requirement 4: テスト
**Objective:** As yikit の開発者, I want テストがサポートする版の振る舞いだけを確かめること, so that 古い版を真似るテストの保守がなくなり、テストの結果がサポートする範囲を表す

#### Acceptance Criteria
1. The yikit のテスト shall 宣言した下限より古い版を真似る、版で結果を分ける、または古い版のために skip するテストと分岐を持たない
2. The yikit のテスト shall 下限以上の版どうしの違いを確かめるテストを残す
3. When 下限によって不要になった版の判定を消したとき, the yikit のテスト shall 消した処理が前提にしていたライブラリの振る舞い（例: `OptunaSearchCV.predict` が `fit` の前からあるメソッドであること、ngboost の `set_params` が入れ子の名前を無視すること）を、入っている版で確かめる
4. When Python 3.10〜3.14 の CI で test と optional の依存を入れて pytest を実行したとき, the テスト shall すべての版で通る
5. When 開発者が手元で `src/` と `tests/` に ruff の静的検査と書式の検査を、`src/yikit` に mypy を実行したとき, the 検査 shall エラーを 0 件で終える
6. When packaging-and-ci の CI の、Python 3.14 で test の依存だけ（optional の依存なし）を入れる job で pytest を実行したとき, the テスト shall 通る（optional の依存が要るテストは skip される）

### Requirement 5: 振る舞いを変えないこと
**Objective:** As yikit の利用者, I want 0.5.0 に上げても、宣言した範囲の版では同じコードが同じ結果を返すこと, so that 依存の範囲の変更だけを気にすればよい

#### Acceptance Criteria
1. The drop-legacy-compat の変更 shall 公開 API の追加・削除・名前と引数の変更を含まない
2. While 宣言した下限と上限を満たす版を使っているとき, when 同じ入力と同じ乱数の種で公開 API を呼んだとき, the yikit shall この spec の変更の前と同じ結果（返り値・警告・例外）を返す
3. If ある分岐を消すと、宣言した範囲の版で結果が変わるとき, the yikit のコード shall その分岐を残す（古い版のための分岐とは見なさない）
4. If optuna-integration が入っていない環境で `OptunaSearchCV` を使う機能（`EnsembleRegressor` の `opt=True`）を使ったとき, the yikit shall optuna 本体の古い場所に頼らず、optuna-integration を入れるか使わない設定を案内する ImportError を出す

### Requirement 6: steering
**Objective:** As yikit の開発者と AI のエージェント, I want steering が 0.5.0 の依存の方針を示すこと, so that 後の作業が「古い環境のために下限を上げない」という古い前提で進まない

#### Acceptance Criteria
1. When この spec の変更が入ったとき, the steering の `tech.md` shall 互換性の方針（下限は消した分岐が要る版だけ、下限の版は CI で確かめない、古い環境は 0.4.x の系列）、各ライブラリの下限・上限とその理由、Core Technologies の Framework と Core deps の下限を 0.5.0 の内容で示す
2. When この spec の変更が入ったとき, the steering の `product.md` shall 「古い環境でも動く」「1.0.0 までは依存の下限を上げない」の記述を、0.5.0 の方針（古い環境は 0.4.x の系列が受け持つ）に書き換える
3. The steering の更新 shall packaging-and-ci が持つ `tech.md` の節（Core Technologies の Language、型、コードの品質、テストの CI の項目、Development Environment、Release）を変えない
4. When この spec の変更が入ったとき, the steering の `tech.md` の Key Technical Decisions shall 「optional extra に重い依存を寄せる」の理由を、古い環境を挙げない 0.5.0 の方針の書き方で示し、他の項目を変えない

### Requirement 7: 記録と確認の扱い
**Objective:** As yikit の作者, I want 変更の理由が残り、環境に影響する確認が自分の承認のもとで行われること, so that release-0.5.0 で CHANGELOG を書け、意図しないインストールが起きない

#### Acceptance Criteria
1. When この spec の実装が終わったとき, the 実装の記録 shall 上げた下限・上限・環境の条件の変更と、それぞれの理由を、release-0.5.0 が CHANGELOG に使える形で残す
2. If 確かめるために手元の環境へパッケージを入れる（新しい ngboost、editable の入れ直しに要る setuptools-scm など。scratchpad へのダウンロードや `--target` への展開も含む）必要があるとき, the 開発の手順 shall 作者の承認を得てから行う
3. The drop-legacy-compat の作業 shall 下限の版を入れて試す CI の job を作らず、tag の作成とリリースを行わない
4. The 開発の手順 shall 読み取り専用の古い環境（`conductivity-prediction-old`）にパッケージを入れない
