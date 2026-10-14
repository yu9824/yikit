# Implementation Plan

- [ ] 1. 前提を確かめ、依存の宣言を消す分岐が要る版にそろえる
  - packaging-and-ci が main に入っていることを確かめる（`requires-python = ">=3.10"`、`typing_extensions` の行がない、`ruff.toml` が `py310`、CI の行列が 3.10〜3.14）。入っていなければ作業を止めて作者に知らせる
  - main から作業ブランチを切る。この spec の文書（`.kiro/specs/drop-legacy-compat/`）が main になければ、基準を取る前に作業ブランチにコミットする（Implementation Notes と CHANGELOG の材料は tasks.md に書くため）
  - 何かを変える前に、手元で `python -m pytest -ra --junitxml=<リポジトリの外の固定の場所>/baseline.xml` を実行し、テストごとの結果（node ID と passed・skipped・failed）をリポジトリの外に残す。置いた場所のパスも Implementation Notes に書く（4.1 が別の作業セッションで行われても読めるように、セッションごとに消える scratchpad には置かない）。全体とファイルごとの passed・skipped などの数と skip の理由の一覧、design の Validation Hooks の legacy grep の行数を Implementation Notes に記録する（2.5 と 4.1 で比べる基準。yikit-dev の editable は作業ツリーを読むので、後からは取り直せない）
  - `dependencies` の scikit-learn・seaborn と、`optional` の lightgbm・ngboost・optuna・optuna-integration の行を design の DependencyDeclaration の表のとおりにする。ngboost は環境の条件を消して上限を `<0.5.11` にし、上限の理由の 1 行のコメントを付ける。他の行、他の extra、`[build-system]` などの節は変えない
  - 完了の状態: `tomllib` で読んだ依存の行が design の表と一致し、`python_version` を含む行がなく、yikit-dev に入っている各ライブラリの版が新しい範囲を満たす（`packaging.requirements` で照らす。読み取りだけ）。基準の XML がリポジトリの外にあり、Implementation Notes に基準の数（全体とファイルごと）と grep の行数がある
  - _Requirements: 1.1, 1.2, 1.3, 2.1, 2.2_

- [ ] 2. 古い版のための分岐を、ファイルのまとまりごとに消す
- [ ] 2.1 (P) helpers と feature_selection から scikit-learn<1.0 と Boruta<0.4 の記述と判定を消す
  - 特徴量の名前を取り出す共通の関数から scikit-learn の版の判定と、その docstring の記述を消し、使われなくなる import を消す。シグネチャと戻り値は変えない
  - feature_selection の 3 つの記述（import のコメント、`feature_names_in_` の説明、例のコメント）と、Boruta の `return_df` のコメントを、版に触れない説明にする
  - テストから、scikit-learn の版の差し替え、1.0 より前で None になるテスト、読めない版で失敗するテストを消し、`get_feature_names_out` のテストの skip の印（`requires_feature_names_out`）を消す。入っている scikit-learn と同じ名前になるテストは残す
  - 完了の状態: `pytest tests/test_helpers.py tests/test_filtermethod.py -ra` が通り、消した skip の印によるものを含め skip が yikit-dev で 0 件で、legacy grep がこの 5 つのファイルで何も出さない
  - _Boundary: ValidationAndSelection_
  - _Requirements: 3.1, 3.3, 4.1, 4.2, 5.1, 5.2_

- [ ] 2.2 (P) 探索範囲から scikit-learn<1.0 と ngboost<0.4 の道を消す
  - 木の `criterion` の候補から 1.0 より前の道とその記述を消す。1.9 の道と、読めない版での ValueError は残す
  - 入れ子の名前を無視するモデルの判定から `set_params` の比較と「Older ngboost」の記述を消し、ngboost が入っていれば NGBRegressor を返す。docstring を合わせ、使われなくなる import を消す
  - テストの候補の表から 0.24 の行を消し、判定のテストのコメントを版に触れない説明にする。判定が実際の `set_params` の振る舞いと合うことを確かめるテストと、ngboost がないときのテストは残す
  - 完了の状態: `pytest tests/test_search_space.py -ra` が通り、入っている ngboost で判定のテストが走って通り、legacy grep がこの 2 つのファイルで何も出さない
  - _Boundary: SearchSpace_
  - _Requirements: 3.1, 3.2, 3.3, 4.1, 4.2, 4.3, 5.2, 5.3_

- [ ] 2.3 (P) optuna の探索から古い optuna と optuna-integration のための処理を消す
  - `OptunaSearchCV` を optuna-integration からだけ import し、optuna 本体の古い場所への切り替えと、その説明を消す。`ParamDistributions` の例の import も 1 行にする
  - `predict` の上書きと、それを説明する docstring を消す（optuna-integration 4.1 から親のメソッドが同じ呼び方をする）。`feature_names_in_` の docstring の scikit-learn 1.0 の記述を消す。tags と `_estimator_type` は残す
  - 探索のテストは、optuna-integration がなければ module ごと skip する形にし、import の切り替え、scikit-learn 1.0 の判定と skip、`predict` が property の古い親を真似るクラスとそのテスト、それだけのための引数を消す。`predict` が `OptunaSearchCV` の普通のメソッドで、yikit のクラスが上書きしていないことを確かめるテストを足す
  - optuna のテストから、import の切り替え、ngboost の `set_params` の判定と、それによる分岐と skip を消す。許容値のコメントから 0.24.1 の記述を消す（値は変えない）
  - 完了の状態: `pytest tests/test_search_cv.py tests/test_optuna.py -ra` が通り、足したテストが走り、`"predict" not in vars(OptunaSearchRegressor)` が成り立ち、legacy grep がこの 4 つのファイルで何も出さない
  - _Boundary: TuningSearch_
  - _Requirements: 3.1, 3.2, 3.3, 3.5, 4.1, 4.2, 4.3, 5.1, 5.2, 5.4_

- [ ] 2.4 (P) GBDTRegressor から LightGBM 3 のための判定と記述を消す
  - early stopping の引数を決める関数から `verbose` の判定とその説明を消す。`eval_X`・`eval_y`（LightGBM 4.7）の分岐は残す
  - `best_iteration_` を LightGBM の回帰モデルの `best_iteration_` から読み、属性と class の docstring の LightGBM 3 の記述を直す
  - テストから LightGBM 3 の署名と `verbose` の次元を消し、4.0〜4.6 と 4.7 以降の署名のどちらでも `verbose` を渡さないことを確かめる。`best_iteration_` が回帰モデルの値と booster の値に等しいことを確かめる
  - 完了の状態: `pytest tests/test_gbdt.py -ra` が通り、legacy grep がこの 2 つのファイルで何も出さない
  - _Boundary: GBDTModel_
  - _Requirements: 3.1, 3.2, 3.3, 4.1, 4.2, 4.3, 5.2_

- [ ] 2.5 (P) visualize とそのテストから seaborn<0.12・LightGBM 3・ngboost<0.4・optuna<3.1 のための処理を消す
  - permutation importance の図で、`errorbar=None` を常に渡し、`ci` の道と、その説明と、使われなくなる import を消す。seaborn 0.13 の並びのための指定は残す
  - y-y プロットの `NDArray` のコメントを型のためだけの import という説明にし、RMSE の docstring の scikit-learn 0.24.1 の記述を消す。学習曲線の LightGBM 3 のコメントを直す（判定のコードは変えない）
  - 図のテストの冒頭の Python 3.14 の注記、LightGBM の 2 つのコメント、study の方向を作る補助の optuna の版の分岐、Student's t の skip と、正規分布で代わりに試すテストのコメントを直す
  - 参照画像は作り直さない
  - 完了の状態: `pytest tests/test_visualize.py -ra` が通り、1 の基準の XML と比べて、このファイルのテストの node ID と結果（passed・skipped）が変わらず（今の環境では消す skip が実行されていない）、`git status tests/imgs` に変更がなく、legacy grep がこの 4 つのファイルで何も出さない
  - _Boundary: VisualizeModule_
  - _Requirements: 2.3, 3.1, 3.2, 3.3, 4.1, 4.2, 5.2_

- [ ] 2.6 (P) ngboost についての記述を下限と上限に合わせる
  - `log_likelihood` が使う非公開の属性について、確かめた範囲を「0.4.0 through 0.5.11」にする（コードは変えない）
  - log-likelihood のテストで LogNormal と CRPScore を外す理由を、design の T13 のとおりに直す（組み合わせは外したまま）。そのとき「ngboost 0.3」と続けて書かない（「as in 0.3.6」のように書き、legacy grep に掛からないようにする）。パラメータのテストの `deep=False` のコメントを版に触れない説明にする
  - 完了の状態: `pytest tests/test_metrics.py tests/test_params.py -ra` が通り、`git diff -- src/yikit/metrics/_ngboost.py tests/test_metrics.py tests/test_params.py` の変更がコメントの行だけで、legacy grep がこの 3 つのファイルで何も出さない
  - _Boundary: NgboostSweep_
  - _Requirements: 2.2, 3.3, 4.1_

- [ ] 2.7 アンサンブルの特徴量の名前の判定を共通の関数に寄せ、古い版の記述を直す
  - 複製した判定（版の判定と名前の取り出し）を消し、名前を記録するモデルがないときは helpers の共通の関数で名前を決める。使われなくなる import を消す（共通の関数のシグネチャと戻り値は 2.1 でも変わらない）
  - `feature_names_in_` の docstring の scikit-learn 1.0 の記述を消し、NGBRegressor が regressor と見なされない条件の記述を「ngboost before 0.5.2」に直す
  - テストから scikit-learn 1.0 の判定と skip、import の切り替えを消し、import を塞ぐテストは optuna-integration だけを塞ぐ。案内つきの ImportError が出て、原因の例外の名前が optuna-integration であることは今のまま確かめる。この確かめは、2.3 が optuna 本体の古い場所への切り替えを消した後でないと通らない（optuna 5 の `optuna.integration` が、塞がれた import を名前のない例外にして出し直すため）
  - 完了の状態: `pytest tests/test_ensemble.py -ra` が通り、DataFrame の名前の 3 つのテストが skip されずに走り、legacy grep がこの 2 つのファイルで何も出さない
  - _Boundary: EnsembleModule_
  - _Depends: 2.3_
  - _Requirements: 3.1, 3.3, 3.4, 3.5, 4.1, 4.2, 5.1, 5.2, 5.4_

- [ ] 3. steering の tech.md と product.md を 0.5.0 の依存の方針に書き換える
  - tech.md の互換性、Key Libraries、Core Technologies の Framework と Core deps の行を、design の SteeringDocs のとおりに書き換える（下限は消した分岐が要る版だけ、下限は CI で確かめない、古い環境は 0.4.x の系列、6 つのライブラリの下限と ngboost の上限とその理由、0.5.11 の扱い、dependabot の上限の PR の扱い）
  - product.md の Target Use Cases と Value Proposition の古い環境についての 2 つの記述を書き換える。両方の `updated_at` を更新する
  - packaging-and-ci が持つ節（Language、型、コードの品質、テストの CI の項目、Development Environment、Release）と、他の節、structure.md、roadmap.md には触らない
  - 完了の状態: tech.md と product.md に「1.0.0 まで上げない」「古い環境でも動く」の記述がなく、Key Libraries に 6 つのライブラリの下限（ngboost は上限も）と理由があり、`git diff -- .kiro/steering` で packaging-and-ci の節に変更がない
  - _Requirements: 1.3, 2.4, 6.1, 6.2, 6.3_

- [ ] 4. 統合の確認
- [ ] 4.1 手元で静的検査・型検査・legacy grep・テストの全体を通し、基準と比べる
  - `ruff check src tests`、`ruff format --check src tests`、`mypy` を実行する
  - design の Validation Hooks の legacy grep を実行する
  - `python -m pytest -ra --junitxml=<1 と同じ場所>/after.xml` で全体を実行し、1 の基準の XML とテストの node ID ごとに比べる。違いが、消したテスト・足したテスト・古い版のための skip の分だけであることを確かめる
  - 各サブパッケージの `__init__.py` と `__all__` に変更がないことを `git diff` で確かめる
  - 完了の状態: ruff と mypy がエラー 0 件、legacy grep の出力が `tests/test_boruta.py` の 1 行だけ、pytest に failed と error がなく、node ID ごとの違いの内訳（消した・足した・skip でなくなった）が Implementation Notes に書かれている
  - _Requirements: 3.1, 3.2, 3.3, 3.5, 4.1, 4.2, 4.5, 5.1, 5.2, 5.3_

- [ ] 4.2 作者の承認を得て ngboost 0.5.10 でテストを回し、0.5.11 の違いを記録する
  - 作者に、yikit-dev へ `ngboost==0.5.10`（sympy が一緒に入る）を入れる承認を求める。調べるためだけに 0.5.11 を入れて 0.5.10 に戻すことも同じ確認で尋ねる。承認がなければ何も入れず、0.5.10 は 4.3 の CI だけで確かめると Implementation Notes に書いて、4.3 に進む
  - 承認があれば、0.5.10 を入れて `python -m pytest -ra` の全体を回す。0.5.11 の承認もあれば、0.5.11 を入れて ngboost に関わるテスト（metrics、search_space、optuna、params、ensemble、visualize の ngboost のテスト）を回し、落ちたテストと理由を記録してから 0.5.10 に戻す。0.5.11 に合わせるコードの変更はしない
  - 0.5.11 の結果が、1 と 3 で書いた上限の理由（`pyproject.toml` のコメント、tech.md の Key Libraries）と食い違えば、その記述を直す
  - scratchpad への `pip download` や `pip install --target` も承認なしに行わない。読み取り専用の `conductivity-prediction-old` には何も入れない
  - 完了の状態: Implementation Notes に、入れた ngboost の版、0.5.10 での pytest の結果、0.5.11 で落ちたテストの一覧（または承認がなかったこと）が書かれ、上限の理由の記述がその結果と合っている
  - _Requirements: 2.2, 2.4, 7.2, 7.4_

- [ ] 4.3 作業ブランチで CI を手動実行し、Python 3.10〜3.14 で通ることと、3.14 で ngboost のテストが走ることを確かめる
  - 作業ブランチを push し、CI を手動実行する。下限の版を入れて試す job は足さず、tag も付けない
  - 5 つの job の結果と、各 job の `pip install` の log から scikit-learn・lightgbm・ngboost・optuna・optuna-integration・seaborn・pandas の版を記録する。3.14 の job の skip の一覧に ngboost のテストがないことを確かめる
  - 失敗したら原因を直す。分岐を残す必要があるとき（宣言した範囲で結果が変わるとき）や、コードの振る舞いを変える必要があるときは作者に相談する
  - pandas 3 が CI のどの版でも入らなくなること（ngboost の依存の lifelines が pandas<3.0 を求める）を作者に報告し、その判断を Implementation Notes に書く。pandas 3 を CI で試し続けるための workflow の変更が要るなら、それは packaging-and-ci の側の作業で、この spec では判断を記録するだけにする
  - Implementation Notes の CHANGELOG の材料を、確かめた結果（CI で入った版、0.5.11 の扱い）で更新する
  - 完了の状態: その run でテストの 5 つの job がすべて成功し、5 つの job のすべてに ngboost 0.5.10 が入り、3.14 の job で ngboost のテストが skip されていない。run の ID、各 job の版、pandas 3 についての作者の判断、更新した CHANGELOG の材料が Implementation Notes に書かれている
  - _Requirements: 1.4, 2.1, 2.3, 4.4, 7.1, 7.3_

## Implementation Notes
- 共通: 何かを入れる・ダウンロードする前（scratchpad への `pip download` や `pip install --target` も含む）に作者の承認を得る。`conductivity-prediction-old` には何も入れない。tag を付けず、下限の版を試す CI の job を作らない
- 共通: packaging-and-ci と共有するファイル（`pyproject.toml`、`.kiro/steering/tech.md`、`src/yikit/visualize/_yyplot.py`）では、design の Boundary Commitments の行と節だけを触る
- 共通: 書き換えたコメントや docstring が legacy grep（design の Validation Hooks）に掛からないようにする（例: 「ngboost 0.3」「scikit-learn < 1.0」と書かない）。ファイルを限って grep を実行すると、各タスクの完了を確かめられる
- 1・4.1: pytest の基準の XML と後の XML は、リポジトリの外の固定の場所に置き、そのパスをここに書く。リポジトリには入れない。セッションごとに消える scratchpad には置かない
- 2.x: `ruff.toml` では F401 が unfixable。使われなくなった import は手で消す（例: `_validation.py`・`_ensemble.py` の `re`・`sklearn`、`_search_space.py` の `BaseEstimator`、`_permutation_importance.py` の `is_argument`、テストの `sklearn`・`SelectorMixin`）
- 2.5: 図のテストの参照画像は作り直さない（`--save-reference-figures` を使わない）
- 2.7: `tests/test_ensemble.py` の import を塞ぐテストは、2.3 の前に `optuna_integration` だけを塞ぐと、optuna 5 の `optuna.integration` が名前のない `ModuleNotFoundError` を出し直して `cause.name in blocked` が落ちる
- 0.5.11 への対応（この spec の外）の材料: 0.5.11 は `set_params` が入れ子の名前を `Base` に渡し、`get_params(deep=True)` が `Base__*` を返す。合わせるには、`ignores_nested_set_params` を振る舞いで判定し、`ParamDistributions` の警告と `EnsembleRegressor` の docstring を版に合わせ、`test_ngboost_detection_matches_its_set_params` などを見直す。0.5.11 の既定の木の `criterion` は `"squared_error"`
- release-0.5.0 の CHANGELOG の材料:
  - 依存の下限を上げた: scikit-learn>=1.0（0.24.1 から。`feature_names_in_`・`get_feature_names_out`・`"squared_error"` を前提にする）、seaborn>=0.12（`barplot` の `errorbar`）、lightgbm>=4.0（LightGBM 3 の扱いをやめた）、ngboost>=0.4（0.3.6 から。`set_params` の扱い）、optuna>=3.2（3.0 から。optuna-integration 4.1 以上が `optuna.terminator` を使うため）、optuna-integration>=4.1（下限なしから。`OptunaSearchCV.predict` が普通のメソッド）
  - ngboost は Python 3.14 でも入る。上限は `<0.5.9` から `<0.5.11` に上げた。0.5.11 は `set_params`・`get_params` の変更に yikit がまだ合わせていないので外している（入っていれば pip が 0.5.10 に下げる）
  - `OptunaSearchCV` は optuna-integration からだけ使う。optuna-integration を入れずに optuna 3.x の `optuna.integration` で `EnsembleRegressor(opt=True)` を使っていた環境では ImportError になる（`pip install "yikit[optional]"` で入る）
  - 公開 API と、宣言した範囲の版での結果は変わらない。古い環境（Python 3.8・3.9、scikit-learn<1.0 など）は 0.4.x を使う
  - 3.14 でも ngboost が入るので、lifelines を通じて pandas<3.0 になる（ngboost を入れた環境）
