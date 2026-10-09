# Requirements Document

## Project Description (Input)
yikit 0.4.0 に向けて、開発と検査の土台を整える。CI で ruff・mypy・pytest を走らせ、Python 3.8〜3.14 で動くことを確かめる。依存の下限と extra を実際に動く範囲に合わせ、dependabot が下限を上げるだけの PR を出さないようにする。パッケージ全体の型注釈を新しい書き方に揃えつつ、Python 3.8 でも動くようにする。main の CI を失敗させている画像比較テストを安定させる。詳しい背景は `brief.md` を参照。

## Introduction
yikit の作者と、古いオフライン環境で yikit を使う利用者（licond など）のために、変更が壊れていないことを自動で確かめられる土台を作る。この spec は公開 API の振る舞いを変えない。後に続く spec（module-quality、optuna-tuning、ensemble-on-sklearn）と直接実装の作業（gbdt-fix）は、この土台の上で進める。

## Boundary Context
- **In scope**: CI の検査（書式、静的検査、型、テスト）、対応する Python の版、依存の下限と extra の宣言、dependabot の設定、パッケージ全体の型注釈の書き方、テストの実行設定、画像比較テストの安定化、テストでの `OptunaSearchCV` の import 元
- **Out of scope**: 依存の下限の組み合わせでの動作の保証と、そのための CI のジョブ（個人開発のライブラリなので、CI で確かめるのは各 Python の版で入る依存の組み合わせまでとする）、新しいテストの追加と docstring の英語化（module-quality と各 spec）、モデルの振る舞いの変更（optuna-tuning、ensemble-on-sklearn、gbdt-fix）、docs の生成の仕組み、リリースと変更履歴（release-0.4.0）
- **Adjacent expectations**: Python 3.8 で動かない箇所が、後の spec で書き換えるモジュール（`models` など）に見つかった場合、この spec では 3.8 で動くための最小限の修正にとどめる。GBDTRegressor が LightGBM 4 で動かない問題は gbdt-fix が持つ

## Requirements

### Requirement 1: CI での自動検査
**Objective:** As yikit の作者, I want 変更のたびに書式・静的検査・型・テストが自動で確かめられること, so that 壊れた変更を main に入れずに済む

#### Acceptance Criteria
1. When main への push があったとき、または main 向けの pull request が作成・更新されたとき, the CI shall `src/` と `tests/` に対して ruff の静的検査と書式の検査を実行する
2. When main への push があったとき、または main 向けの pull request が作成・更新されたとき, the CI shall `src/yikit` に対して mypy の型検査を実行する
3. When main への push があったとき、または main 向けの pull request が作成・更新されたとき, the CI shall Python 3.8・3.9・3.10・3.11・3.12・3.13・3.14 のそれぞれで、test と optional の依存を入れてテストを実行する
4. If ある Python の版でテストが失敗したとき, the CI shall 他の版のテストを中断せずに最後まで実行し、版ごとの結果を報告する
5. If 静的検査・書式・型検査・テストのいずれかが失敗したとき, the CI shall その実行全体を失敗として報告する
6. The CI shall `examples/` を静的検査・書式・型検査の対象に含めない
7. When CI の設定ファイル、ruff・mypy・pytest の設定、またはパッケージの依存の宣言が変更されたとき, the CI shall 検査を実行する

### Requirement 2: 対応する Python の版
**Objective:** As 古い Python の環境で yikit を使う利用者, I want yikit が Python 3.8 でも入って動くこと, so that 環境を新しくできなくても yikit を使える

#### Acceptance Criteria
1. The yikit パッケージ shall Python 3.8 以上を対応する版として宣言する
2. The yikit のパッケージ情報 shall 対応する Python の版（3.8〜3.14）を分類（classifiers）に示す
3. When Python 3.8 で yikit と、その各サブパッケージの公開 API を import したとき, the yikit パッケージ shall 構文や型注釈の評価によるエラーを起こさない
4. Where Python の版が ngboost に対応していないとき（3.14 以上）, the yikit パッケージ shall ngboost なしで install と import ができる
5. If Python 3.8 での CI を通せないと判明したとき, the yikit パッケージ shall 対応する版を Python 3.9 以上に戻し、通せなかった理由を spec の文書に記録する

### Requirement 3: 型注釈の書き方と型検査
**Objective:** As yikit の開発者, I want パッケージ全体の型注釈が1つの書き方に揃い、型検査が通ること, so that 型の誤りを早く見つけられ、後の spec でも同じ書き方を続けられる

#### Acceptance Criteria
1. The yikit パッケージ shall すべてのモジュールの型注釈を、`X | None`・`X | Y` と組み込みの型のジェネリクス（`list[int]`、`dict[str, Any]` など）で書く
2. The yikit パッケージ shall 型注釈に `typing` の `Optional`・`Union`・`List`・`Dict`・`Tuple`・`Set`・`Type` を使わない
3. While Python 3.8 で実行しているとき, the yikit パッケージ shall 上の書き方の型注釈を実行時に評価してエラーを起こさない
4. When `src/yikit` を mypy で検査したとき, the 型検査 shall エラーを0件で終える
5. The 型注釈の書き換え shall 公開 API の引数・返り値・例外・計算結果を変えない

### Requirement 4: 依存と extra の宣言
**Objective:** As yikit を入れる利用者, I want 宣言された依存の範囲が実際に動く範囲と一致していること, so that 入れた後に壊れた組み合わせで動かなくなることがない

#### Acceptance Criteria
1. The yikit のパッケージ情報 shall Boruta の下限を 0.4.3 として宣言する
2. The yikit のパッケージ情報 shall optuna の下限を 3.0 として宣言する
3. The yikit のパッケージ情報 shall optuna-integration を optional の依存に含め、Python の各版で入手できる版が選ばれるように宣言する
4. The yikit のパッケージ情報 shall この Requirement の 1・2 で決める下限の他に、既存の依存の下限を上げない
5. The yikit のパッケージ情報 shall 開発用の extra に、静的検査・書式の検査・型検査に必要な道具（ruff、mypy）を含める
6. When テストが `OptunaSearchCV` を使うとき, the テスト shall optuna-integration の新しい import 元から読み込み、それが入っていない環境では optuna の元の import 元から読み込む
7. When 新しい版の optuna でテストを実行したとき, the テスト shall `OptunaSearchCV` の import 元が非推奨であるという警告を出さない

### Requirement 5: 依存の更新の自動化
**Objective:** As yikit の作者, I want 下限を上げるだけの依存の更新 PR が来ないこと, so that 古い環境を壊さない方針を保ったまま、必要な更新だけに対応できる

#### Acceptance Criteria
1. When 依存の新しい版が公開され、その版が今の宣言の範囲に含まれるとき, the dependabot shall 依存の宣言を変える pull request を作らない
2. When 依存の新しい版が公開され、その版が今の宣言の範囲に含まれないとき, the dependabot shall 範囲を広げる pull request を作る
3. When この spec の変更が main に入ったとき, the 開いたままの dependabot の pull request（#25 setuptools、#26 Boruta） shall 閉じられている。#26 の Boruta の下限の変更は Requirement 4 で取り込む

### Requirement 6: テストの実行設定と画像比較テストの安定化
**Objective:** As yikit の開発者, I want テストが環境の違いで揺れず、実行してもリポジトリを汚さないこと, so that CI の失敗がコードの不具合だけを示すようになる

#### Acceptance Criteria
1. When リポジトリの最上位でテストを実行したとき, the テストの設定 shall `tests/` の中のテストだけを集める
2. When 図を参照画像と比べるテストを実行したとき, the テスト shall 差分の画像などの出力を一時ディレクトリに出し、リポジトリの中にファイルを残さない
3. The 図を比べるテスト shall CI のすべての Python の版（Requirement 1）で通る
4. If 図の内容（描いた要素の数、軸、ラベル、値など）が変わったとき, the 図を比べるテスト shall 失敗する
5. If 描画の細かな違い（フォントや描画ライブラリの版による違い）だけがあるとき, the 図を比べるテスト shall 失敗しない
6. The テストの変更 shall 既存のテストが確かめている対象（どの図・どの結果を検査しているか）を減らさない

### Requirement 7: 振る舞いを変えないこと
**Objective:** As yikit の利用者, I want この土台の整備で yikit の使い方や結果が変わらないこと, so that 0.4.0 の振る舞いの変更を、後の spec の変更だけに絞って把握できる

#### Acceptance Criteria
1. The dev-foundation の変更 shall 公開 API の追加・削除・名前の変更を含まない
2. When 同じ入力と同じ乱数の種で公開 API を呼んだとき, the yikit パッケージ shall この spec の変更の前と同じ結果を返す
3. If Python 3.8 で動かすための修正が既存の振る舞いを変えざるを得ないとき, the 開発の手順 shall その変更を spec の文書に記録し、作者の承認を得てから行う
