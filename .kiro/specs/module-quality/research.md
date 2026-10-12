# Research & Design Decisions

## Summary
- **Feature**: `module-quality`
- **Discovery Scope**: Extension（既存の5つのサブパッケージの docstring・テスト・小さな不具合の修正）
- **Key Findings**:
  - joblib の `Parallel.print_progress` は、逐次実行（`n_jobs=1`・`None`）でも並列でも、作業が1つ終わるたびに `n_completed_tasks` を増やして呼ばれる（joblib 1.5.3 と、Python 3.8 の古い環境の joblib で確認）。今の `tqdm_joblib` が差し替える `BatchCompletionCallBack` は、joblib 1.3 以降の逐次実行では呼ばれないので、進捗バーが進まない
  - 公開 API ごとの docstring・テスト・不具合の状況は、調査の結果（会話の中のサブエージェントの報告）にもとづき brief の Current State にまとめた
  - boruta 本体の `_fit` も `self.random_state = check_random_state(self.random_state)` で引数を書き換える（boruta_py.py:295）。yikit の `__init__` での変換をやめるだけでは足りず、`perc` と同じく `_fit` の間だけ差し替えて戻す必要がある

## Research Log

### tqdm_joblib を逐次実行で動かす
- **Context**: 要件 1.4（`n_jobs` によらず進捗を進める。作者の指示）
- **Sources Consulted**: joblib 1.5.3 の `parallel.py`（`_get_sequential_output`、`print_progress`）、古い環境の joblib での実行
- **Findings**:
  - `n_jobs=1`・`None`・`2` のどれでも、`print_progress` が `n_completed_tasks` = 1, 2, …, 5 で呼ばれる（逐次実行では最後にもう1回）
  - `tqdm_joblib` の中で動く別の `Parallel`（RandomForest やスレッドの中の並列など）も `print_progress` を呼ぶ。差し替えはプロセス全体に効くので、追う `Parallel` を絞らないと進捗が混ざる
- **Implications**: `Parallel.__call__` と `Parallel.print_progress` を差し替え、context に入ったスレッドで、追っている呼び出しが走っていない間に始まった `Parallel` だけを追う。進捗バーは、追っている呼び出しの `n_completed_tasks` の合計まで進める

### root_mean_squared_error の削除
- **Context**: 作者の指示（scikit-learn に `root_mean_squared_error` が入って久しいので削除）
- **Findings**: yikit の中で使うのは `visualize/_yyplot.py` の RMSE だけ。依存の下限は scikit-learn 0.24.1 で、`sklearn.metrics.root_mean_squared_error`（1.4 から）は使えない
- **Implications**: `yyplot` の中で `np.sqrt(mean_squared_error(...))` を計算する小さな関数を持つ。`yikit.metrics` は ngboost があるときの `log_likelihood` だけになる

### FilterSelector の n_features_in_ と feature_names_in_
- **Findings**: `SelectorMixin.get_feature_names_out`（scikit-learn 1.0 以上）は `n_features_in_` と、あれば `feature_names_in_` を使う。`_validate_data`（0.23〜1.5）と `validate_data`（1.6 以上）は版で場所が変わる非公開・新しい API
- **Implications**: 版で分岐する非公開の API は使わず、`check_array` の後に `n_features_in_` を入れ、列名がすべて `str` のとき（scikit-learn 1.0 以上）だけ `feature_names_in_` を入れる。この判定は ensemble-on-sklearn の `_feature_names` と同じ規則。feature_selection は models に依存しないので、helpers に非公開の小さな関数として置く

## Design Decisions

### Decision: tqdm_joblib は print_progress を差し替える
- **Alternatives Considered**:
  1. `BatchCompletionCallBack` を差し替える（今の方式）— 逐次実行で動かない
  2. 逐次実行の経路（`_get_sequential_output`）も差し替える — 非公開の関数で、版で名前が変わる
  3. `print_progress` を差し替える — 逐次でも並列でも、古い joblib でも呼ばれる
- **Selected Approach**: 3。`__call__` も差し替えて追う呼び出しを決め、`print_progress` で進捗バーを `n_completed_tasks` に合わせる。抜けるときに（例外のときも）元に戻す
- **Trade-offs**: どちらもプロセス全体の差し替えなので、入れ子や別々のスレッドでの同時の利用（EnsembleRegressor をスレッドで並列にしたときの BorutaPy など）では、差し替えをロックと数で管理し、進捗はスレッドごとの context に渡す

### Decision: BorutaPy の random_state は perc と同じく _fit の間だけ差し替える
- **Selected Approach**: `__init__` は引数を入れるだけにする。`_fit` で `check_random_state(self.random_state)` を作り、boruta の `_fit` の間だけ `self.random_state` に入れて、try/finally で元の値に戻す。`_use_tqdm`・`_use_logging` も `_fit` で `verbose` から決める
- **Rationale**: boruta 本体が `_fit` で引数を書き換えるため

### Decision: BorutaPy の perc="auto" のシャッフルの種をモデルの外で決める
- **Selected Approach**: `_calc_auto_perc` で、シャッフルごとの種を `random_state` から先に引き、並列のワーカーには種を渡す。各ワーカーはその種で自分の RandomState を作る
- **Trade-offs**: 同じ種でも、今の版と `perc_` の値が変わる（CHANGELOG に書く）

## Risks & Mitigations
- 図の見た目が変わる修正（`yyplot` の配置、フォントの戻し方）で参照画像が変わる — 参照画像を作り直し、文字を消した比較の仕組みで確かめる
- `tqdm_joblib` の差し替えを戻し忘れると、他のコードの joblib の進捗に影響する — try/finally で戻し、テストで確かめる
