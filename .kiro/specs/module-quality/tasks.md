# Implementation Plan

- [x] 1. helpers
- [x] 1.1 is_installed・is_argument・dummy_tqdm の docstring とテスト、列名の補助関数を作る
  - `is_installed`・`is_argument`・`dummy_tqdm` の振る舞いは変えず、英語の numpy 形式の docstring を整える（`is_installed` の名前・引数・返り値は models が使うので変えない）
  - 非公開の `yikit/helpers/_validation.py` に `feature_names_of(X)` を作る（列名がすべて `str`（`numpy.str_` を除く）で scikit-learn 1.0 以上なら `np.array(columns, dtype=object)`、そうでなければ None）
  - テスト（新規 `tests/test_helpers.py`）: `is_installed` の True と False、`is_argument`、`dummy_tqdm` が iterable をたどり `update`・`close` を受け付ける、`feature_names_of` の str・int・混在・`numpy.str_`・ndarray
  - 完了の状態: `tests/test_helpers.py` のこれらのテストが通り、`yikit.helpers` の公開する名前が変わっていない
  - _Requirements: 1.1, 1.2, 1.3, 7.1, 7.3_

- [x] 1.2 tqdm_joblib を n_jobs によらず進むように作り直す
  - `Parallel.__call__` と `Parallel.print_progress` を差し替え、context に入ったスレッドで、追っている呼び出しが走っていない間に始まった `Parallel` だけを追い、終わった作業の数だけ進捗バーを進める（design の Helpers）。`return_as` が list 以外の呼び出しは追わない。抜けるときに（例外のときも）元に戻して進捗バーを閉じる
  - 差し替えはモジュールに1つのロックと数で管理する（最初に入った context が差し替え、最後に出た context が戻す）。context はスレッドごとのスタックに積み、呼び出したスレッドで動く差し替えた `__call__` の中で、いちばん上の context とそのときの合計（`base`）を `Parallel` のインスタンスに結び付ける。`print_progress` はインスタンスから引き、進捗バーを `base + n_completed_tasks` まで進める（進めるだけで戻さない）。`__call__` が終わったら finally で合計を確定し、進捗バーを合わせてから結び付けを外す（loky では `print_progress` が `__call__` の後にも遅れて呼ばれうるため。design の Helpers）。元の関数は差し替えのときに取っておいた参照で呼ぶ。モジュールのロックは差し替えと数の増減の間だけ持ち、元の `__call__` の間と `print_progress` の中では取らない（デッドロックを避ける）
  - docstring を直す（Examples を動く形に、`total` は必須、`silent`、入れ子と別々のスレッドでの同時の利用に対応すること、進捗は context に入ったスレッドで始まった最も外側の `Parallel` の作業を数えること（入れ子では内側の context が数える。`return_as` が list 以外の呼び出しは数えない））
  - テスト: `n_jobs` が None・1・2 で進捗バーが単調に進み total に達する（進捗バーの `update` を記録する。n_jobs=2 ではバッチ単位で進んでよく、1 ずつの歩みや `update` の回数は確かめない）、中で別の `Parallel` を動かしても数が混ざらない、`return_as="generator"` の呼び出しは数えない（joblib 1.3 以上のときだけ確かめる）、抜けると差し替えが戻る、例外のときも戻る、入れ子の `tqdm_joblib` と、2つのスレッドで同時に使ったときも（threading.Event で順番を決めて）それぞれの進捗が正しく、最後に差し替えが戻る、`silent=True` と tqdm がないときは `dummy_tqdm`
  - 完了の状態: `tqdm_joblib(total=5)` の中の `Parallel(n_jobs=1)` で進捗バーが 1 ずつ進み、抜けると `Parallel` が元に戻っていて、`tests/test_helpers.py` が通る
  - _Requirements: 1.4, 1.5, 7.3_

- [x] 2. (P) logging の docstring とテスト、catch_default_handler の戻し方を直す
  - ロガーの振る舞い（import で既定のハンドラを付ける、INFO、propagate False）は変えず、docstring に明記する
  - `catch_default_handler` が抜けるときに、入る前の状態（付いていたか外れていたか）に戻すようにする
  - `enable_default_handler`・`disable_default_handler` の実行時の `assert` を、明示的な検査と意味のある例外にする
  - テスト（新規 `tests/test_logging.py`。テストの後にロガーの状態を元に戻す）: `get_root_logger` の設定、`get_child_logger` の yikit の中・`__main__`・それ以外の ValueError、`enable`・`disable` を何度呼んでも重複しない、`catch_default_handler` が両方の状態から元に戻す、`get_handler` の水準とフォーマッタ
  - 完了の状態: `tests/test_logging.py` が通り、他のテストのログの設定に影響しない
  - _Boundary: Logging_
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 7.1, 7.3_

- [x] 3. (P) metrics から root_mean_squared_error を削除する（境界をまたぐ小さな統合）
  - `yikit/metrics/_common.py` を消し、`yikit.metrics` は ngboost があるときだけ `log_likelihood` を公開する。`log_likelihood` を、学習の `Score` によらず予測した分布の平均の対数尤度（大きいほど良い）を返すようにし（`Score=LogScore` では今と同じ値。作者の判断）、docstring を合わせる
  - `visualize/_yyplot.py` の `root_mean_squared_error` の import を、scikit-learn の RMSE の既定と同じ値（多出力では出力ごとの RMSE の平均）を `mean_squared_error(..., multioutput="raw_values")` から計算する非公開の関数に置き換える（scikit-learn 0.24.1 でも動く。yyplot のそれ以外は 6.1 で扱う）
  - テスト（新規 `tests/test_metrics.py`）: `from yikit.metrics import root_mean_squared_error` が ImportError、`log_likelihood` が NGBRegressor と NGBClassifier で `Score=LogScore` なら `-score` と等しく、`Score=CRPScore` でも予測した分布の平均の対数尤度になり、他のモデルで TypeError（ngboost がなければ飛ばす）、yyplot の非公開の RMSE の関数の値が、1次元では平均二乗誤差の平方根、多出力では出力ごとの RMSE の平均と等しい（`test_metrics.py` で直接確かめる）
  - 完了の状態: src に `root_mean_squared_error` の参照が残らず、tests では ImportError を確かめるテストだけが名前を使い（scikit-learn の scorer の文字列 `neg_root_mean_squared_error` は別）、テストが通る
  - _Boundary: Metrics, Visualize（yyplot の import の1か所）_
  - _Requirements: 3.1, 3.2, 3.3, 7.1, 7.3_

- [x] 4. FilterSelector を直す
  - 相関の判定を絶対値にし（`|r| > r` かつ `p < alpha`）、再帰の削除を同じ順番のループに書き直す
  - 学習で `n_features_in_` と、`feature_names_of(X)` があれば `feature_names_in_` を入れる。一定の列は警告を出さずに相関なしとして扱う
  - クラスの docstring（Parameters・Attributes・Examples）
  - テスト（`tests/test_filtermethod.py` に追記）: 強い負の相関の列を消す、正の相関だけのデータでは今と同じ列を選ぶ、`get_feature_names_out`（DataFrame と ndarray）、特徴量の数が違う `transform` の ValueError、1000 を超える削除で再帰の上限に当たらない、一定の列で警告なし
  - 完了の状態: 上のテストと既存の `test_filter_method` が通る
  - _Boundary: FilterSelector_
  - _Depends: 1.1_
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 7.1, 7.3_

- [x] 5. (P) BorutaPy を scikit-learn の決まりに合わせる
  - `__init__` は引数を入れるだけにする。`_use_tqdm`・`_use_logging` は `_fit` の初めに決める
  - `_fit` で `random_state` から RandomState を作り、boruta の `_fit` の間だけ `self.random_state` と `self.perc` を差し替えて、try/finally で元に戻し、開いた進捗バーを閉じる（ensemble-on-sklearn の `perc` の扱いは保つ）
  - `perc="auto"` のシャッフルの種を先に引いてワーカーに渡し、`n_jobs` によらず同じ `perc_` にする
  - クラスの docstring を numpy 形式で書き直し（`perc_`・`r_ccmax_`・`pears_` などの属性、Examples、References）、誤字を直す
  - テスト（`tests/test_boruta.py` に追記。小さな `max_iter`・`max_shuf`、`verbose=0`）: 学習の後も `get_params()["random_state"]` が渡した値、同じインスタンスで2回学習して同じ特徴量と `perc_`、`n_jobs` が 1 と 2 で同じ `perc_`、`set_params(verbose=...)` が次の学習で効く、例外のとき進捗バーが閉じて `random_state`・`perc` が戻る
  - 既存の `test_boruta`（`perc="auto"`、`random_state=334` で `support_` を固定値で確かめる）の期待値が変わったときは、変わった原因が新しい種の扱い（シャッフルの種を先に引くこと）だけであることを確かめてから期待値を作り直し、その理由をテストのコメントに書く
  - 完了の状態: 上のテストと既存の BorutaPy のテスト（ensemble-on-sklearn のものを含む）が通る
  - _Boundary: BorutaPy_
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 7.1, 7.3_

- [x] 6. visualize
- [x] 6.1 テストの収集を optional の依存から切り離し、yyplot を直す
  - `tests/test_visualize.py` の optuna・lightgbm・ngboost の import を、使うテストの中の `pytest.importorskip` にする（ファイルが収集で失敗しない）
  - `yyplot`: 未知の指標を描く前に ValueError、`ax` を渡したときは `tight_layout` をかけない、ラベルを書いたとおりに表示する、docstring
  - テスト: 描いた点の値と指標の文字、渡した ax の図の配置が変わらない、未知の指標で何も描かずに ValueError、`$`・`_`・`%` を含むラベル
  - 完了の状態: optuna・lightgbm・ngboost を見えなくしても `tests/test_visualize.py` が収集でき（それらのテストは飛ばされる）、yyplot のテストが通る
  - _Depends: 3_
  - _Requirements: 6.2, 7.3, 7.4_

- [x] 6.2 フォントの設定と SummarizePI を直す
  - `set_font` の docstring に全体の設定を変えることを書く。`with_custom_matplotlib_settings` の日本語のコメントを英語にする
  - `SummarizePI`: クラスの docstring、`get_figure` を `with_custom_matplotlib_settings` で包む、`get_data` が図と同じ値（特徴量ごとの正規化した重要度の平均と標準偏差、大きい順）を DataFrame で返す、日本語のコメントを英語に
  - テスト: `set_font` が設定を変える、`with_custom_matplotlib_settings` が `restore=True` で戻し `False` で戻さない、`get_figure` の後にフォントの設定が元のまま、`get_data` の値と順番、既存の SummarizePI の図のテスト
  - 完了の状態: 上のテストが通り、SummarizePI の参照画像との比較が通る（見た目が変わるなら作り直す）
  - _Requirements: 6.1, 6.6, 6.7, 7.1, 7.2, 7.3_

- [x] 6.3 get_learning_curve_optuna を直す
  - 完了した試行だけを使い、完了した試行がない・方向がない・多目的の study では理由を示す ValueError
  - docstring を整え、`loc` に注釈を付ける
  - テスト（optuna がなければ飛ばす）: 失敗・枝刈り・実行中の試行を含む study で、完了した試行だけの曲線になる、3つの ValueError、既存の図のテスト
  - 完了の状態: 上のテストと既存の optuna の図のテストが通る
  - _Requirements: 6.4, 6.7, 7.1, 7.3_

- [x] 6.4 get_dist_figure と get_learning_curve_gb を直す
  - `get_dist_figure`: サンプルが 1・2 個でも落ちない（`squeeze=False`）、`titles` の既定を None、docstring の誤字と書きかけを直す
  - `get_learning_curve_gb`: `with_custom_matplotlib_settings` で包む、任意の評価データの名前に色を割り当てる（既知の名前は今の色）、評価の記録が空なら ValueError、NGBoost・LightGBM 以外は TypeError、日本語のコメントを英語に
  - テスト（ngboost・lightgbm がなければ飛ばす）: 分布の図の 1・2 サンプル、LightGBM の既定の名前（"training"・"valid_0"）で図を返す、呼んだ後にフォントの設定が元のまま、空の記録の ValueError、既存の図のテスト
  - 完了の状態: 上のテストと既存の分布・学習曲線の図のテストが通る（見た目が変わるなら参照画像を作り直す）
  - _Requirements: 6.1, 6.3, 6.5, 6.7, 7.1, 7.2, 7.3_

- [x] 7. 英語化と公開 API の網羅を確かめる
  - `tests/conftest.py` の日本語の docstring とコメント、対象のモジュールに残る日本語のコメントを英語にする。モジュールの docstring のないもの（`_helpers.py`・`_ngboost.py`・`_filter.py`・`_wrapper.py` など）に英語の docstring を付ける
  - 5つのサブパッケージの `__all__` の名前が、英語の numpy 形式の docstring を持ち、少なくとも1つのテストで使われていることを確かめるテストを足す。規則: docstring の確認では `yikit.logging` が再公開する標準ライブラリの定数（`CRITICAL`・`DEBUG`・`ERROR`・`INFO`・`NOTSET`・`WARNING`）を除く。テストでの使用は定数も含めて確かめる。optional の依存がない環境では、そのとき `__all__` にある名前だけを確かめる。docstring は概要の行があり、引数を取るなら Parameters の見出しを持つこと（Returns などの見出しは必須にしない）。テストでの使用は、この spec のテストのファイルに名前が出てくることで確かめる
  - 作業中に見つかった不具合（作者の判断）: BorutaPy の `get_support(weak=True)` が confirmed と tentative の両方（`support_ | support_weak_`）を返すようにし、テストを足す
  - 完了の状態: 対象のファイルに日本語の文字が残らず（ライセンスの見出しを除く）、網羅のテストが通り、`get_support(weak=True)` のテストが通る
  - _Depends: 1.2, 2, 3, 4, 5, 6.4_
  - _Requirements: 5.7, 7.1, 7.2, 7.3_

- [x] 8. 検証
- [x] 8.1 全体の検査を通す
  - 手元で `pytest`、`ruff check src tests`、`ruff format --check src tests`、`mypy`、変えたモジュールの doctest を通す
  - この spec のテストのファイル（test_helpers・test_logging・test_metrics・test_filtermethod・test_boruta・test_visualize）を、optional の依存を見えなくした状態（`sys.modules` に None を入れる、または別のプロセス）で収集できることを確かめる
  - 完了の状態: すべてエラーなく終わる
  - _Requirements: 7.3_

- [x] 8.2 作業ブランチで CI を実行し、全版で通ることを確かめる
  - 作業ブランチを push し、CI を手動で実行して Python 3.8〜3.14 の7つの版の結果を確かめる。版による失敗は原因を直す
  - 完了の状態: 7つの版のテストがすべて成功する
  - _Requirements: 7.4, 7.5_

## Implementation Notes
- 手元の検査は yikit-dev の環境の実行ファイルを絶対パスで使う（`/Users/yu9824/opt/miniforge3/envs/yikit-dev/bin/{python,pytest,ruff,mypy}`）。`ruff format` は `src tests` に限る
- joblib の `Parallel.print_progress` は、逐次実行でも並列でも作業が終わるたびに `n_completed_tasks` を増やして呼ばれる（新旧の joblib で確認。research.md）
- boruta 本体の `_fit` も `random_state` を書き換えるので、`perc` と同じく `_fit` の間だけ差し替えて戻す
- テストの小さなデータは各テストファイルの中に置く。`tests/conftest.py` は英語化だけ（共通の fixture は変えない）
- `.vscode/settings.json` の未コミットの変更は利用者のもの。触らない
- docstring の Examples は doctest で動く形にするか、遅いもの・optional の依存やフォントの有無に頼るもの（NGBRegressor の学習、optuna の多くの試行、`set_font("Arial")` など）は `# doctest: +SKIP` を付ける（8.1 で変えたモジュールの doctest を通すため）
- R7.4 の範囲はこの spec のテストのファイル。models のテストのファイル（`tests/test_optuna.py`・`test_gbdt.py`・`test_search_space.py`）は先頭で optuna を import するので、optional の依存なしでは収集できない。直すなら別の作業として扱う
- release-0.4.0 の CHANGELOG に書くこと（module-quality）: `yikit.metrics.root_mean_squared_error` の削除（`sklearn.metrics.root_mean_squared_error` を使う）、`log_likelihood` が学習の `Score` によらず対数尤度を返すようになったこと（`Score=CRPScore` などで値が変わる）、FilterSelector が相関の絶対値で判定するようになったこと、BorutaPy の `get_support(weak=True)` が confirmed と tentative の両方を返すようになったこと（以前は tentative だけ）、BorutaPy の `random_state` を加工しなくなったことと `perc="auto"` の `perc_` と、境目の特徴量の選ばれ方が同じ種でも変わること、`tqdm_joblib` が逐次実行でも進むことと、`return_as` が list 以外の `Parallel` を数えなくなったこと、`SummarizePI.get_data` の実装、図の関数が落ちていた入力の修正とフォントの設定を戻すようになったこと、`catch_default_handler` が入る前の状態に戻すこと
- `tqdm_joblib`（1.2）: リリース済みの属性 `tqdm_joblib.tqdm`（入る前は None、中では進捗バー、抜けた後は閉じた進捗バー）を互換のために残した。有効な同じインスタンスにもう一度入ったときと、入る前に抜けたときは RuntimeError（CHANGELOG に書く）
- logging（2）: 既定のハンドラは `import yikit` では付かず、最初の `get_root_logger()`（`yikit.feature_selection` の import が呼ぶ）で付く。止め方の案内は `get_root_logger()` を通す形にした（それより前に `logging.getLogger("yikit")` に入れた水準は上書きされるため）。CHANGELOG に書く: `catch_default_handler` が入る前の状態に戻すこと、同じインスタンスに入れ子で入れること、入る前に抜けると RuntimeError、既定のハンドラがないときの例外が AssertionError から RuntimeError に変わったこと
- metrics（3）: `log_likelihood` は `ngboost.manifold.manifold(LogScore, Dist)` を予測したパラメータ（`pred_dist(X)._params`、`NGBoost.score` と同じ）に当てて計算する。打ち切り付きの分布（LogNormal・Exponential）は ngboost が包んだ `DistWithUncensoredScore` を名前で見分けて LogScore で包み直す（ngboost の `__getstate__` と同じ見分け方）。ngboost 0.3.6 は LogNormal を CRPScore で学習できない。NormalFixedVar は ngboost 自身の `score` と同じく落ちる（upstream）
- FilterSelector（4）: 削除の順番は旧実装と同じ（つながる組が最も多い特徴量から、同数なら番号の小さいほう）。CHANGELOG に書く: `corr_` の一定の列を含む組が NaN（または丸めによる意味のない値）から r=0.0・p=1.0 になったこと、`|r| > r` の負の組に実際の p 値が入ること（以前は p=1.0）
- BorutaPy（5、8.2）: `perc_` は 79.58 から 81.00 に変わった。既存の `test_boruta` は手元では通ったが、Linux の CI では全版で落ちた。特徴量 2 は tentative のまま残り、boruta の最後の判定（重要度の中央値と影の閾値の中央値の比較）で、差が 0.5% 未満しかない。種の扱いを変えたので乱数の列が変わり、この判定が環境の丸め誤差で反転する（旧コードでも 20 回に 3 回反転した）。そのため、境目でない 9 つの特徴量の `support_` と `ranking_` を確かめる形にした。範囲の中で足したこと: 数値の `perc` で学習すると前の `r_ccmax_`・`pears_` を消す、閉じた進捗バーを消して学習した BorutaPy を pickle できるようにした（以前は tqdm があり verbose>=1 だと TypeError。CHANGELOG に書く）。7 で直す docstring: `perc_` の型は「float or int」、`_transform`・`get_support`・`_print_results` の古い書き方（`array-like, shape = [...]`）
- yyplot（6.1）: CHANGELOG に書く: 指標の行のラベルを数式の外に置くので、見た目が「R²_train = 0.98」（下付き）から「R² (train) = 0.98」に変わる（ラベルのない行は同じ）、ラベルを数式として読まなくなる（凡例でも書いたとおりの文字）、渡した ax の図に `tight_layout` をかけない、未知の指標の ValueError を描く前に出す（メッセージも変わる）。空のラベル `""` は None と同じに扱う
- SummarizePI（6.2）: CHANGELOG に書く: `get_figure` がばらつき（正規化した反復の標準偏差、ddof=0）を誤差棒で描き、x の範囲が広がった（参照画像を作り直した）、数値の特徴量名でも重要度の大きい順に並ぶ（seaborn 0.13 では名前の順だった）、`get_figure` の後にフォントの設定を戻す、`get_data` の実装（列は `importances_mean`・`importances_std`、同じ値は入力の順）
- optuna の学習曲線（6.3）: CHANGELOG に書く: 完了していない試行を描かなくなった（以前は失敗・実行中の試行が NaN の点になり、枝刈りの試行の中間値が最良値の線を変えうた）、多目的・方向のない・完了した試行のない study で ValueError（以前は optuna の RuntimeError・UnboundLocalError・空の図）。方向のない study は公開の API では作れないので、テストは InMemoryStorage で作る（optuna 3.0 と 4.5 は手元で未確認。CI で確かめる）
- 分布の図と GB の学習曲線（6.4）: CHANGELOG に書く: 1・2 サンプルの分布の図が動く（予測平均の幅が 0 のときは x の範囲を「平均 ± 有限で正の標準偏差の最大値の 3 倍」、なければ ±1.0）、`titles` の既定が None（空の列も None と同じ、配列も可、長さ違いは ValueError）、`return_axis=True` の axes が常に 2 次元、`get_learning_curve_gb` が呼んだ後にフォントの設定を戻す、未知の評価データの名前を ValueError でなく matplotlib の色の順で描く、記録がない・空・未学習のモデルで理由付きの ValueError、受け付けるクラスを `LGBMModel`・`NGBoost` の基底に広げた、optional の依存がないとき None を渡すと TypeError（以前は AttributeError）
- 最後の検証で見つかった CHANGELOG の追加: FilterSelector が `n_features_in_`・`feature_names_in_` を持ち `get_feature_names_out` が使える（そのため `transform` が scikit-learn の検査を受け、特徴量の数が違うと IndexError でなく ValueError、DataFrame で学習して ndarray を渡すと UserWarning、列名の違う DataFrame は ValueError）、BorutaPy の `set_params(verbose=...)` が次の学習から効く、`tqdm_joblib` が数えるのは context に入ったスレッドで始まった最も外側の `Parallel` だけ（以前は数えている作業の中の `Parallel` や別のスレッドの `Parallel` でも進むことがあった）
