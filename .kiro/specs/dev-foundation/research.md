# Research & Design Decisions

## Summary
- **Feature**: `dev-foundation`
- **Discovery Scope**: Extension（既存の CI・設定・コードへの手入れ。light discovery）
- **Key Findings**:
  - ruff 0.16 は既定で有効な規則が大幅に増えており、今の `ruff.toml` のままだと `src`・`tests` で 79 件の違反が出る（`FA100` 32 件、`UP032` 11 件など）。規則を明示的に選ばないと、ruff を更新するたびに CI の結果が変わる
  - mypy 2.4 は検査の対象として Python 3.10 未満を指定できない。Python 3.8 での動作は、ruff の `target-version = "py38"` による書き方の検査と、CI の 3.8 でのテストで保証する。今の mypy のエラーは 3 件だけ
  - main の CI は、画像比較のテストで失敗している（Python 3.11 で RMS 30.7、許容値 20）。原因は、Python の版ごとに選ばれる matplotlib などの版の違い（3.10 は古い matplotlib で通り、3.11 以上は新しい matplotlib で失敗する）。図の入力をモデルの学習から作っているので、依存の版によって数値そのものも変わりうる

## Research Log

### ruff の規則と自動修正
- **Context**: Requirement 1・3（静的検査、型注釈の書き方）
- **Sources Consulted**: `ruff check --statistics`、`ruff check --show-settings`（yikit-dev 環境、ruff 0.16.10）
- **Findings**:
  - 設定ファイルは `ruff.toml`（行の長さ 79、`unfixable = ["F401"]`、pydocstyle の numpy 形式）。`select` を書いていないので、ruff の既定の規則がそのまま使われる
  - 既定の規則で出る違反: FA100（`from __future__ import annotations` があれば新しい書き方にできる注釈）32 件、UP032・UP030（format の書き方）18 件、UP036（古い版の分岐）4 件、B006・B008（引数の既定値）5 件、BLE001・S110 など
  - `target-version = "py38"` と `lint.isort.required-imports = ["from __future__ import annotations"]`（I002）を使うと、ruff の自動修正で、全ファイルへの `from __future__ import annotations` の追加、`Optional`・`Union`・`List` などの書き換え（UP006・UP007・UP045）、Python 3.8 未満の分岐の削除（UP036）ができる
- **Implications**: 規則を `select` で明示する。型注釈の書き換えは手作業ではなく ruff の自動修正で行い、残った違反だけを手で直す

### mypy の対象版
- **Context**: Requirement 3.4
- **Sources Consulted**: `mypy --python-version 3.8`（mypy 2.4.0）
- **Findings**: `Python 3.8 is not supported (must be 3.10 or higher)`。3.9 も同じ。既定（実行している Python）での検査では、エラーは 3 件（`_optuna.py` の dict-item、`_yyplot.py` の arg-type 2 件）
- **Implications**: mypy の設定に対象版は書かない。3.8 で動くかは ruff（py38）と CI のテストで確かめる

### Python 3.8 で入る依存
- **Context**: Requirement 2
- **Sources Consulted**: PyPI の JSON API（各パッケージの `requires_python`）、actions/python-versions の versions-manifest
- **Findings**:
  - Python 3.8 で入る最新の版の目安: scikit-learn 1.3.2、numpy 1.24.4、pandas 2.0.3、matplotlib 3.7.5、seaborn 0.13.2、optuna 4.5.0、optuna-integration 4.5.0、lightgbm 4.6.0、Boruta 0.4.3。ngboost も 3.8 に対応する版がある
  - optuna-integration は 4.6.0 から `>=3.9`。`requires_python` の情報があるので、pip は Python の版に合った版を自動で選ぶ（marker は不要）
  - GitHub Actions の ubuntu-24.04 に Python 3.8.18 がある
- **Implications**: 依存の宣言に Python の版の marker を足す必要はない。3.8 では古い scikit-learn・optuna で結果の数値が変わる可能性があり、値を固定で比べるテスト（`test_optuna.py` の最良値、`test_boruta.py` の選ばれた特徴量）が 3.8 で失敗するおそれがある

### 3.8 で動かない書き方
- **Context**: Requirement 2.3・3.3
- **Sources Consulted**: `grep`（`typing` の import、`sys.version_info`、3.9 以降の API）
- **Findings**:
  - `from __future__ import annotations` を使っているファイルはない（全 28 ファイル）
  - 実行時に評価される注釈で 3.8 では動かないもの: `_filter.py` の入れ子の関数（`tuple[int, int]`）、`_optuna.py` の `Objective.__init__`（`dict[str, Any]`）
  - `removeprefix`・`functools.cache` など 3.9 以降の API は使っていない。関数の外の式で `list[...]` などを評価している箇所もない
  - `sys.version_info >= (3, 8)` の分岐（`_yyplot.py`、`_wrapper.py`）は、3.8 以上では片方が使われない
- **Implications**: `from __future__ import annotations` を全ファイルに入れれば、3.8 の問題はほぼ解消する

### 画像比較テストが揺れる原因
- **Context**: Requirement 6
- **Sources Consulted**: CI の実行 37877717334 のログ、`tests/test_visualize.py`、`src/yikit/visualize/_gbdt.py`・`_optuna.py`
- **Findings**:
  - 3.10 は通り、3.11 は失敗する（fail-fast で他は中断）。手元（Python 3.12、matplotlib 3.11.2）では 3 件が失敗（RMS 30・22・22）
  - `compare_images` は、差分の画像を参照画像の隣（`tests/imgs/`）に書き出す
  - 図の入力は、RandomForest・NGBoost・LightGBM・optuna の study を実際に学習・最適化して作っている。依存の版によって数値が変わると、線の形そのものが変わる
  - 図の関数が実際に使う入力は単純: `SummarizePI` は重要度の DataFrame、`get_learning_curve_optuna` は study の各試行の値と方向、`get_dist_figure` は ngboost の分布オブジェクト（`pdf` を呼ぶ）、`get_learning_curve_gb` はモデルの `evals_result`（NGBoost は普通の属性、LightGBM は学習済みかを確かめる property）
  - matplotlib には、目盛りとタイトルを消す `matplotlib.testing.decorators.remove_ticks_and_titles` がある（3.11.2 で確認）
- **Implications**: 図の入力は、モデルの学習を通さずに固定の値から作る（LightGBM だけは学習が必要なので、小さく決定的な設定で学習する）。画像を比べる前にすべての文字を消し、文字の内容は文字列として別に比べる。参照画像は一時ディレクトリに写してから比べ、差分の画像がリポジトリに出ないようにする

## Architecture Pattern Evaluation

| Option | Description | Strengths | Risks / Limitations | Notes |
|--------|-------------|-----------|---------------------|-------|
| 許容値を上げる | RMS の許容値を 30 以上にする | 変更が最小 | 図の内容が変わっても気づけない（6.4 を満たせない）。依存がさらに変われば再び失敗する | 不採用 |
| 版ごとに参照画像を持つ | matplotlib の版ごとに参照画像を作る | 厳密に比べられる | 版が増えるたびに画像が増える。手元で全版の画像を作れない | 不採用 |
| 固定の入力・文字を消して比べる・文字は文字列で比べる | 入力を学習から切り離し、描画の細かな違いの元（文字）を画像の比較から外す | 依存の版の影響を受けにくい。内容の変化（線・要素の数、ラベル）は検出できる | 学習の結果が正しく図になるかの確認は、学習を伴うテストに任せる | 採用 |

## Design Decisions

### Decision: ruff の規則を明示して固定する
- **Context**: ruff 0.16 で既定の規則が増え、版によって CI の結果が変わる
- **Alternatives Considered**:
  1. 既定の規則のまま、出た違反をすべて直す
  2. 規則を `select` で明示する
- **Selected Approach**: `select = ["E4", "E7", "E9", "F", "W", "I", "UP", "FA"]`、`target-version = "py38"`、`required-imports` で `from __future__ import annotations` を必須にする。`examples`・`docs_src` は除外する
- **Rationale**: 型注釈の書き方（UP・FA）と import の並び（I）は、この spec の要件に直結する。B（bugbear）などは、既定値の引数を変えると公開 API の見た目が変わる（7.1）ので、ここでは入れない
- **Trade-offs**: B・D などの規則は後の spec で必要に応じて足す
- **Follow-up**: module-quality で docstring の規則（D）を入れるかを検討する

### Decision: 型注釈の書き換えは ruff の自動修正で行う
- **Context**: Requirement 3.1・3.2 と、28 ファイルの書き換え
- **Selected Approach**: `ruff check --fix` で future import の追加、注釈の書き換え、不要な版の分岐の削除を行い、残りを手で直す
- **Rationale**: 機械的で漏れがなく、振る舞いを変えない（7.2）
- **Trade-offs**: 自動修正の結果は差分が大きくなる。レビューは「自動修正」と「手での修正」をコミットで分けて楽にする

### Decision: mypy は実行している Python を対象にする
- **Context**: mypy 2.4 は 3.10 未満を対象にできない
- **Selected Approach**: `mypy.ini` に対象のファイル（`src/yikit`）を書き、対象の版は書かない。CI の型検査は 1 つの Python（3.12）で行う
- **Trade-offs**: 3.8 固有の型の誤りは mypy では見つからない。実行時の問題は CI の 3.8 のテストで見つける

### Decision: 画像比較を、固定の入力と、文字を除いた比較に変える
- **Context**: Requirement 6.2〜6.6
- **Selected Approach**: 比較の仕組みを `tests/conftest.py` の fixture にまとめる。図のすべての文字を見えなくしてから画像を比べ、文字の内容（タイトル、軸のラベル、凡例）と要素の数は、テストの中で値として比べる。参照画像は一時ディレクトリに写してから比べる。参照画像を作り直すときは pytest の引数（`--save-reference-figures`）で行う
- **Rationale**: 依存の版による描画と数値の揺れを比較から外しつつ、図の内容の変化は検出できる
- **Trade-offs**: 「学習した結果がそのまま正しく図になるか」は、このテストでは確かめなくなる。図の関数は、受け取ったデータを描くことだけを責務にしているので、学習はモデルのテストに任せる
- **Follow-up**: 許容値は、CI の全版で出た RMS の最大値をもとに決め、research.md に記録する

### Decision: 値を固定で比べるテストが 3.8 で失敗した場合の扱い
- **Context**: 3.8 では古い scikit-learn・optuna が選ばれ、学習や探索の数値が変わりうる
- **Selected Approach**: 3.8 の CI で値の違いによる失敗が出たら、その値の比較を「同じ種で2回実行すると同じ結果になる」と「値が妥当な範囲にある」に置き換え、理由を research.md に記録する。コードの不具合による失敗なら、コードを直す
- **Rationale**: 依存の版による数値の違いは、yikit の不具合ではない。再現性と妥当性の検査は版に左右されない

## Risks & Mitigations
- 3.8 で入る依存の組み合わせで、予想していない失敗が出る — CI の 3.8 で確かめ、直せないものは作者に相談し、最終的には `>=3.9` に戻す（2.5）
- 文字を消しても、線の描き方（アンチエイリアスなど）で RMS が出る — 全版の CI で RMS を測ってから許容値を決める
- LightGBM の学習曲線は学習が必要で、版によって数値が変わりうる — 小さなデータ・決定的な設定で学習する。それでも揺れる場合は、学習済みの判定を通る形で固定の `evals_result` を与える
- ruff の自動修正で、`__init__.py` の `__all__` の並びなど、意図しない変更が入る — 自動修正のコミットを分け、`RUF022` などの規則は選ばない

## References
- [Dependabot options reference: versioning-strategy](https://docs.github.com/en/code-security/dependabot/working-with-dependabot/dependabot-options-reference#versioning-strategy--) — pip でも使え、`increase-if-necessary` は範囲内の新しい版では宣言を変えない
- [actions/python-versions versions-manifest](https://github.com/actions/python-versions) — ubuntu-24.04 向けの Python 3.8.18
- [Ruff settings: required-imports / target-version](https://docs.astral.sh/ruff/settings/) — future import の必須化と対象版
- [matplotlib.testing](https://matplotlib.org/stable/api/testing_api.html) — `compare_images`、`remove_ticks_and_titles`
