# Implementation Plan

- [ ] 1. 基盤: パッケージの宣言と検査の設定
- [x] 1.1 パッケージの宣言を、実際に動く範囲と新しい検査の道具に合わせる
  - 対応する Python を 3.8 以上にし、分類に 3.8〜3.14 を並べる
  - optional の依存を Boruta 0.4.3 以上、optuna 3.0 以上にし、optuna-integration を加える。ngboost の Python の版の条件はそのまま残す
  - test の extra は pytest だけにし、dev の extra に ruff と mypy を加える。必須の依存の下限は変えない
  - pytest がリポジトリの最上位から実行されても `tests/` だけを集めるように設定する
  - 完了の状態: 最上位で `pytest --collect-only -q` を実行すると `tests/` のテストだけが並び、`pip install --dry-run ".[test,optional,dev]"` が依存を解決できる（実際のインストールはしない。新しいパッケージが必要になる場合は作者の承認を得る）
  - _Requirements: 2.1, 2.2, 2.4, 4.1, 4.2, 4.3, 4.4, 4.5, 6.1_

- [x] 1.2 ruff と mypy の規則と対象を固定する
  - ruff の対象版を Python 3.8 にし、選ぶ規則を design の一覧（E4・E7・E9・F・W・I・UP・FA）に固定する
  - すべてのモジュールに `from __future__ import annotations` を必須とする設定を加える
  - `examples/` と `docs_src/` を ruff の対象から外す
  - mypy が引数なしで `src/yikit` を検査するように対象を設定する
  - 完了の状態: `ruff check src tests --statistics` に選んだ規則の違反だけが出る。`mypy` を引数なしで実行すると `src/yikit` だけが検査される
  - _Requirements: 3.1, 3.2, 3.3, 8.2, 8.3_

- [ ] 2. パッケージ全体の型注釈の書き換え
- [x] 2.1 ruff の自動修正で、future import の追加・注釈の書き換え・古い版の分岐の削除を行う
  - `src/` と `tests/` のすべてのモジュールに future import を入れ、`Optional`・`Union`・`List` などを `X | None` と組み込みの型の書き方に置き換える
  - Python 3.8 未満向けの分岐を消し、`Literal` などは標準の `typing` から使う
  - 自動修正の結果だけを1つのコミットにまとめる
  - 完了の状態: `src/` と `tests/` に `typing` の `Optional`・`Union`・`List`・`Dict`・`Tuple`・`Set`・`Type` の import がなく、手元（Python 3.12）で `test_boruta`・`test_filtermethod`・`test_optuna` が書き換えの前と同じ結果で通る（`test_visualize` の画像比較の既知の失敗 3 件は 3 で扱う）
  - _Requirements: 2.3, 3.1, 3.2, 3.3, 3.5, 7.1, 7.2_

- [x] 2.2 自動修正で残った違反と mypy のエラーを手で直す
  - 選んだ規則の残りの違反を、公開 API の名前・引数・既定値・返り値を変えずに直す
  - mypy のエラー（`_optuna.py` の dict-item、`_yyplot.py` の arg-type 2 件）を、注釈を正すか値の型を明示して直す。計算は変えない
  - 実行時に評価される場所で、PEP 585/604 の書き方を使っていないことを確かめる
  - 振る舞いを変えないと直せないものが出たら、作業を止めて作者の承認を得る
  - 完了の状態: `ruff check src tests`、`ruff format --check src tests`、`mypy` がすべてエラー 0 件で終わり、2.1 と同じテストが同じ結果で通る
  - _Requirements: 3.4, 3.5, 7.1, 7.2, 7.3, 8.1_

- [ ] 3. テストの調整
- [x] 3.1 図を、文字を除いた画像として参照画像と比べる仕組みを作る
  - 図のすべての文字を見えなくしてから保存し、実際の図と参照画像の写しを一時ディレクトリに置いて比べる
  - 許容値を超えたときは、RMS の値・許容値・一時ファイルの場所・参照画像の作り直し方を含むメッセージで失敗させる。参照画像がないときも、作り直し方を示して失敗させる
  - pytest の引数で、参照画像を作り直すモードに切り替えられるようにする
  - 比較の仕組み自体のテストを加える: 同じ図は通る、線を足した図や線の値を変えた図は失敗する、文字だけを変えた図は通る、失敗しても `tests/imgs/` に新しいファイルができない
  - 完了の状態: 比較の仕組みのテストが手元で通り、その実行の後に `git status` で `tests/imgs/` に変化がない
  - _Requirements: 6.2, 6.4, 6.5_

- [x] 3.2 5 つの図のテストを固定の入力に書き換え、参照画像を作り直す
  - SummarizePI、optuna の学習曲線、ngboost の分布の図、学習曲線（NGBoost・LightGBM）の 5 つを、今と同じ参照画像のファイル名のまま残す
  - 図の入力を、モデルの学習や最適化に頼らない固定の値から作る（LightGBM の学習曲線だけは、小さな固定のデータと決定的な設定で学習する）
  - 各テストで、画像を比べる前に、図のタイトル・軸のラベル・凡例の文字と、軸や線などの要素の数を値として確かめる
  - 直接実行で参照画像を保存する仕組みをやめ、3.1 の作り直しのモードで 5 枚の参照画像を作り直す
  - ngboost を使うテストは、今と同じく Python 3.14 以上で skip する
  - 完了の状態: 手元で `pytest tests/test_visualize.py` がすべて通り、`tests/imgs/` の 5 枚が文字を除いた画像に置き換わっている
  - _Requirements: 6.2, 6.3, 6.4, 6.5, 6.6_

- [x] 3.3 (P) テストの `OptunaSearchCV` を新しい場所から読み込む
  - optuna-integration から読み込み、入っていない環境では optuna の元の場所から読み込む
  - 完了の状態: `pytest tests/test_optuna.py -W 'error:`optuna.integration:FutureWarning'` の確認（警告の文はバッククォートで始まる）で、import 元が非推奨であるという警告が出ず、テストが通る
  - _Boundary: OptunaImportInTests_
  - _Requirements: 4.6, 4.7_

- [ ] 4. CI と依存の更新の設定
- [x] 4.1 CI のテストを Python 3.8〜3.14 の行列にする
  - テストのジョブは Python 3.8〜3.14 の各版で、test と optional の依存を入れて pytest を実行し、ある版が失敗しても他の版を止めない
  - 起動の条件（main への push、pull request、手動実行）と対象のパスは今のものを保つ。ruff・mypy のジョブは加えない
  - 完了の状態: workflow の定義が YAML として読め、テストのジョブ 7 つ（3.8〜3.14、fail-fast なし）が定義されている
  - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [x] 4.2 (P) dependabot が宣言の範囲内の更新で PR を作らないようにする
  - pip の更新の方針を `increase-if-necessary` にし、他の設定（週ごと、コミットの prefix、PR の上限）は変えない
  - 完了の状態: dependabot の設定が YAML として読め、pip の項目に更新の方針が入っている
  - _Boundary: DependabotConfig_
  - _Requirements: 5.1, 5.2_

- [ ] 5. 統合と検証
- [ ] 5.1 作業ブランチで CI を手動実行し、全版で通ることを確かめる
  - 手動実行で、テストの 7 つの版の結果を確かめる
  - Python 3.8 で失敗が出たら、コードの不具合ならコードを直す。依存の版による数値の違いなら、research.md の方針（同じ種での再現性と妥当な範囲の検査への置き換え）に従い、理由を research.md に記録する。直せない、または振る舞いを変える必要があるときは、作者に相談する。最終的に 3.8 を通せなければ、design の戻し方に従って 3.9 以上に戻す
  - 各版の画像比較の RMS を集め、それをもとに許容値を決めて research.md に記録する
  - 完了の状態: 作業ブランチでの CI の実行で 7 つの版のテストがすべて成功し、research.md に各版の RMS と決めた許容値（と 3.8 で行った対応）が書かれている
  - _Depends: 2.2, 3.2, 3.3, 4.1_
  - _Requirements: 1.1, 1.2, 1.3, 2.3, 2.5, 6.3, 7.3_

- [ ] 5.2 開いている dependabot の PR を後始末する
  - この spec の PR が main に入った後に、#25（setuptools）と #26（Boruta）に、方針（宣言の範囲内の更新はしない、Boruta の下限はこの PR で取り込んだ）を書いたコメントを付けて閉じる
  - 完了の状態: #25 と #26 がコメント付きで閉じられ、Boruta の下限 0.4.3 が main の宣言に入っている
  - _Depends: 1.1, 4.2_
  - _Requirements: 4.1, 5.3_

## Implementation Notes
- 1.1: `setuptools>=77.0.3` は Python 3.9 以上が必要で、3.8 ではソースからビルドできなかった。作者の判断で `setuptools>=61`・表の形の `license`・`license-files` の削除にした。新しい setuptools は 2027-02-18 の期限つきで非推奨の警告を出す。3.8 の確認には `/Users/yu9824/opt/miniforge3/envs/conductivity-prediction-old/bin/python3.8` を読み取りだけで使える（`pip install --dry-run` に限る）
- 共通: `tests/test_visualize.py` を実行すると、既知の失敗で `tests/imgs/*-failed-diff.png` ができる（3.1 で直すまで）。実行の後に消す
- 1.2: ruff 0.16 の `ruff format .` は Markdown（`.kiro/`、README など）も対象にして書き換える。手元では `ruff format src tests` のようにパスを指定して実行する。`force-exclude = true` は、VS Code の拡張のようにファイルを名指ししても除外を効かせるため
- 2.1: 自動修正の順番は I002 → I001・UP037 → UP045・UP007（unsafe）→ UP036（unsafe）→ F401（CLI で `--fixable F401`）→ I001。isort の修正でモジュールの docstring と import の間の空行が消えるので、`ruff format` で整える（2.2）。test_optuna の最初の study は、浮動小数の最後の桁が実行ごとにぶれる（変更の前からある非決定性）
- 2.2: `cast` の型は文字列で書く（例: `cast("NDArray[Any] | Sequence[float]", x)`）。実行時に評価されないので Python 3.8 でも動く。`numpy.typing.NDArray` は numpy 1.20 で入ったので、`TYPE_CHECKING` の中で import する（numpy に下限がないため）。`np.min` に Series を渡すと `Series.min`（NaN を飛ばす）に任されるので、`np.asarray` に変えると振る舞いが変わる
- 3.1: 比較の仕組みは `check(fig, name, *, reference_dir=None)`。許容値 20 では、既定の大きさの図で線を1本変えても RMS は約 7 で通ってしまう（5.1 で下げる）。yikit の図の関数は作るときに `tight_layout()` を呼ぶので、文字を消す前にフォントの寸法で軸の位置が決まる。凡例の枠も見えない文字の大きさで決まる。3.2 では、文字を消した後に layout をやり直すか、CI の RMS で確かめる
- 3.2: 手元（3.12）では、フォントや大きさを変えても参照画像との RMS は最大 0.033。内容を変えたときの RMS は 2.5〜18.6 なので、5.1 では CI の全版の RMS を見て許容値を 2.5 より十分小さくできるか確かめる。内容の変化は、まず各テストの値の確認（文字、要素の数、描いたデータ、軸の範囲）で捕まえる。matplotlib 3.5 未満は凡例の並びが違うので、凡例の文字は並べ替えて比べる。LightGBM の学習曲線は lightgbm 3.2.1 と 4.7 で同じ値になった
- 3.3: `-W error::FutureWarning` は、ngboost の import で sklearn の `friedman_mse` の FutureWarning が出て失敗するので使えない。src の docstring の例（`_optuna.py` の `from optuna.integration import OptunaSearchCV`）は optuna-tuning で直す
- 4.1: runner は `ubuntu-24.04` に固定した。setup-python は Python 3.8・3.9 を ubuntu-26.04 向けに配っていない。GitHub が ubuntu-24.04 の image をなくすときに見直す
