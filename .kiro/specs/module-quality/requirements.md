# Requirements Document

## Project Description (Input)
yikit を使う人（作者、licond など）は、モデル以外のモジュール（helpers・logging・metrics・feature_selection・visualize）が今の依存の版で正しく動くのかを、テストで確かめられない。docstring も形式が崩れていたり、なかったりし、Sphinx の API リファレンスが欠ける。読んでみると、図の関数が特定の入力で落ちる、matplotlib のフォントの設定を戻さない、FilterSelector が強い負の相関を消さない、といった不具合もある。

この spec では、5つのサブパッケージの公開 API に英語の numpy 形式の docstring とテストを揃え、見つかった小さな不具合を直す。機能の追加はしない（例外は、中身のない `SummarizePI.get_data` を実装すること）。詳細は `brief.md` を参照。

## Introduction
yikit の作者と利用者のために、models 以外の公開 API（helpers・logging・metrics・feature_selection・visualize）が、Python 3.8〜3.14 と各版の依存で意図どおり動くことをテストで確かめ、英語の docstring で説明する。あわせて、意図した振る舞いから外れている箇所を直す。振る舞いが変わるのは、FilterSelector の相関の判定（絶対値にする）、BorutaPy の乱数の扱い、図の関数が落ちていた入力、`SummarizePI.get_data` の中身、`tqdm_joblib` の逐次実行での進捗、`yikit.metrics.root_mean_squared_error` の削除で、0.4.0 の CHANGELOG に書く。ロガーの振る舞いは変えない。

## Boundary Context
- **In scope**: `yikit.helpers`・`yikit.logging`・`yikit.metrics`・`yikit.feature_selection`・`yikit.visualize` の公開 API（各 `__init__.py` の `__all__` にある名前）の docstring・テスト・小さな不具合の修正。これらのモジュールの日本語のコメントと、`tests/conftest.py` の日本語の docstring の英語化。テストが optional の依存なしでも収集できること
- **Out of scope**: `yikit.models` の各クラス（optuna-tuning・ensemble-on-sklearn・gbdt-fix で扱った）。ロガーの振る舞いの変更（import で既定のハンドラを付けることなど。変えるなら 1.0.0）。公開していない関数（`calc_vip`、`is_correct_dist` など）。docs のサイトの構成（Sphinx の設定、手で書くページ）。新しい機能（`SummarizePI.get_data` の実装と、`tqdm_joblib` の逐次実行への対応を除く）
- **Adjacent expectations**:
  - `yikit.helpers.is_installed` は models からも使われ、テストでは置き換えられている。名前・引数（文字列1つ）・返り値（bool）を変えない
  - BorutaPy は ensemble-on-sklearn で `perc` の解決と DataFrame の変換を直した。その振る舞いとテストを保つ
  - 図のテストは dev-foundation の仕組み（文字を消して参照画像と比べる）に乗せる。図の見た目が変わるときは参照画像を作り直す

## Requirements

### Requirement 1: helpers
**Objective:** As yikit の開発者, I want 補助の関数の振る舞いがテストで確かめられていること, so that optional の依存の判定や進捗の表示を安心して使える

#### Acceptance Criteria
1. When 入っているパッケージと入っていないパッケージの名前を `is_installed` に渡したとき, the helpers shall それぞれ True と False を返す（名前・引数・返り値の型は変えない）
2. When 関数とその引数の名前を `is_argument` に渡したとき, the helpers shall その引数があれば True、なければ False を返す
3. When `dummy_tqdm` を tqdm の代わりに使ったとき, the helpers shall 渡された iterable をそのまま順にたどり、`update`・`close` などの呼び出しを何もせずに受け付ける
4. When `tqdm_joblib` の中で joblib の `Parallel` を実行したとき, the helpers shall `n_jobs` によらず（逐次実行の `n_jobs=1` や `None` を含む）、作業が終わるたびに終わった数だけ進捗バーを進め、抜けるときに進捗バーを閉じる
5. When `tqdm_joblib` を抜けたとき, the helpers shall joblib に加えた変更を元に戻す（例外のときも）

### Requirement 2: logging
**Objective:** As yikit の利用者, I want yikit のロガーの振る舞いが説明され、テストで固められていること, so that ログを止めたり、自分のハンドラを付けたりする方法が分かる

#### Acceptance Criteria
1. The logging shall 今の振る舞い（`get_root_logger` が "yikit" のロガーに既定のハンドラを付け、水準を INFO、propagate を False にすること）を変えず、docstring に明記する
2. When `get_child_logger` に yikit の中のモジュール名か `__main__` の中の名前を渡したとき, the logging shall その子のロガーを返し、それ以外の名前では ValueError を出す
3. When `disable_default_handler` と `enable_default_handler` を呼んだとき, the logging shall 既定のハンドラを外し、また付ける（何度呼んでも重複しない）
4. When `catch_default_handler` を抜けたとき, the logging shall 既定のハンドラを、入る前の状態（付いていたか外れていたか）に戻す
5. When `get_handler` にハンドラを渡したとき, the logging shall 指定した水準とフォーマッタ（なければ既定のもの）を付けて返す

### Requirement 3: metrics
**Objective:** As yikit の利用者, I want yikit の指標が、scikit-learn にないものだけになっていること, so that 同じ指標を2つの場所から選ばずに済む

#### Acceptance Criteria
1. The metrics shall `root_mean_squared_error` を提供しない（scikit-learn 1.4 以降の `sklearn.metrics.root_mean_squared_error` を使ってもらう。0.4.0 で削除し、import すると Python の通常の ImportError になる）。yikit の中でこの値を使う箇所（`yyplot` の RMSE）は、scikit-learn 0.24.1 でも動く形で同じ値を計算する
2. Where ngboost が入っているとき, when `log_likelihood` に学習した NGBRegressor か NGBClassifier を渡したとき, the metrics shall 平均の対数尤度（大きいほど良い値）を返し、それ以外のモデルでは TypeError を出す
3. The metrics shall `log_likelihood` の docstring を、返す値（負の対数尤度ではなく対数尤度）に合わせる

### Requirement 4: FilterSelector
**Objective:** As yikit の利用者, I want 相関の強い特徴量を符号によらず取り除けること, so that 冗長な特徴量を確実に減らせる

#### Acceptance Criteria
1. When FilterSelector を学習したとき, the FilterSelector shall 相関係数の絶対値が `r` より大きく、その p 値が `alpha` より小さい組から特徴量を取り除く（強い負の相関も取り除く）
2. When 学習した FilterSelector の `get_feature_names_out` を呼んだとき, the FilterSelector shall 残した特徴量の名前を返す（列名のある入力で学習したときはその列名）。学習後に `n_features_in_` を持ち、列名のある入力では `feature_names_in_` も持つ（その属性を持つ版の scikit-learn のとき）
3. When 学習のときと違う数の特徴量で `transform` を呼んだとき, the FilterSelector shall ValueError を出す
4. When 取り除く特徴量が多いとき（1000 を超えるなど）, the FilterSelector shall 再帰の深さの上限で失敗せずに学習を終える
5. If 値が一定の列があるとき, the FilterSelector shall その列を他の列と相関がないものとして扱い、学習を終える（NumPy の警告を利用者に見せない）
6. The FilterSelector shall クラスの docstring に、引数・学習後の属性・使い方の例を持つ

### Requirement 5: BorutaPy
**Objective:** As yikit の利用者, I want BorutaPy が scikit-learn の決まりに従い、同じ乱数の指定で同じ結果になること, so that 同じインスタンスで学習し直しても、clone しても結果が再現する

#### Acceptance Criteria
1. The BorutaPy shall `__init__` で引数を加工せず、`get_params()["random_state"]` は渡した値（整数や None）のままにする
2. When 整数の `random_state` を渡した同じインスタンスで2回学習したとき, the BorutaPy shall 同じ特徴量を選び、同じ `perc_` を得る
3. When `perc="auto"` で `n_jobs` を変えて学習したとき, the BorutaPy shall 整数の `random_state` に対して同じ `perc_` を得る（並列のワーカーが同じシャッフルを繰り返さない）
4. When `set_params(verbose=...)` で表示の設定を変えたとき, the BorutaPy shall 次の学習からその設定に従う
5. If 学習の途中で例外が出たとき, the BorutaPy shall 進捗バーを閉じ、`perc` を元に戻す（ensemble-on-sklearn で入れた振る舞いを保つ）
6. The BorutaPy shall クラスの docstring を numpy 形式で正しく書き（見出しの形、例、学習後の属性 `perc_`・`r_ccmax_` など）、誤字を直す

### Requirement 6: visualize
**Objective:** As yikit の利用者, I want 図の関数がどんな正しい入力でも図を返し、自分の matplotlib の設定を変えないこと, so that ノートブックの他の図に影響を受けずに使える

#### Acceptance Criteria
1. The visualize shall 図を作る関数とメソッド（`yyplot` を除く `get_*` と `SummarizePI.get_figure`）の中で変えた matplotlib のフォントの設定を、呼び出しの後に元に戻す（`set_font` は全体の設定を変える関数として残し、そのことを docstring に書く）
2. When `yyplot` に ax を渡したとき, the visualize shall その ax にだけ描き、図全体の配置（`tight_layout`）を変えない。未知の指標の名前を渡したときは、描く前に ValueError を出す。ラベルは書いたとおりの文字で表示する
3. When `get_dist_figure` にサンプルが1つか2つの分布を渡したとき, the visualize shall 落ちずに図を返す
4. When `get_learning_curve_optuna` に失敗・枝刈り・実行中の試行を含む study を渡したとき, the visualize shall 完了した試行だけで曲線を描く。完了した試行がない study、方向のない study、多目的の study では、理由を示す ValueError を出す
5. When `get_learning_curve_gb` に任意の名前の評価データ（LightGBM の既定の "training"・"valid_0" など）を持つモデルを渡したとき, the visualize shall 落ちずに図を返す。評価の記録がないモデルには、理由を示す ValueError を出す
6. When `SummarizePI.get_data` を呼んだとき, the visualize shall `get_figure` が描く値（特徴量ごとの正規化した重要度の平均とばらつき）を DataFrame で返す
7. When 図の関数を呼んだとき, the visualize shall 既存の図のテスト（参照画像との比較と値の確認）を通す。図の見た目が意図して変わるときは、参照画像を作り直す

### Requirement 7: docstring、テスト、互換性
**Objective:** As yikit の作者, I want 対象の公開 API がすべて英語の docstring とテストを持ち、どの版でもテストが通ること, so that API リファレンスが欠けず、変更が壊れていないことを CI で確かめられる

#### Acceptance Criteria
1. The yikit パッケージ shall 5つのサブパッケージの `__all__` にあるすべての名前（`yikit.logging` が再公開する標準ライブラリの定数 `CRITICAL`・`DEBUG` などを除く）に、英語の numpy 形式の docstring（概要、Parameters、Returns か Attributes、必要なら Raises と Examples）を持つ。モジュールの docstring も英語で持つ
2. The yikit パッケージ shall 対象のモジュールと `tests/conftest.py` の日本語のコメントと docstring を英語にする
3. The yikit のテスト shall 1〜6 の受け入れ基準を確かめ、5つのサブパッケージの `__all__` のすべての名前を少なくとも1つのテストで使う
4. Where optional の依存（optuna、lightgbm、ngboost、boruta、tqdm）が入っていない環境のとき, the yikit のテスト shall この spec のテストのファイル（helpers・logging・metrics・feature_selection・visualize のもの）で、その依存を使うテストを飛ばし、残りを実行する（ファイルが収集の段階で失敗しない。models のテストのファイルは対象外）
5. When CI が Python 3.8〜3.14 の各版でテストを実行したとき, the yikit のテスト shall すべて通る
