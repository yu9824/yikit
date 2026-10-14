# Roadmap

## Overview
yikit 0.4.0 を出すための残りの作業です。0.4.0-rc.0 はすでに PyPI に公開しているので、それを消さずに 0.4.0-rc.1、0.4.0 と進めます。
中心になるのは optuna による調整の作り直しです。下流の licond は、モデルの `n_jobs` と `random_state` を自分で決めて `Objective` に渡したいと考えていますが、今の `Objective` はそれを上書きしてしまいます。あわせて、探索範囲が2か所に書かれていてずれ始めていること、sklearn の Pipeline などを探索できないことも直します。
周辺の作業として、開発環境と CI を整え、テストと英語の docstring を全体にそろえ、EnsembleRegressor を sklearn の部品で作り直し、GBDTRegressor の不具合を直します。

0.4.0 は 2026-10-14 に公開しました。次の 0.5.0 は、末尾の「Phase 2: 0.5.0」にまとめています。

## Approach Decision
- **Chosen**: 案A。optuna まわり（探索範囲の共通化と、`Objective` の引数の扱い）を1つの spec `optuna-tuning` にまとめ、残りは責務ごとに spec か直接実装に分ける
- **Why**: 探索範囲の共通化と `Objective` の引数の扱いは、どちらも `Objective.__call__` と `get_best_params` の同じ範囲を書き換える。1つの spec にすれば設計もレビューも1回で済む。0.4.0 はまとめて出すので、licond 向けの修正だけを急いで先に出す理由はない
- **Rejected alternatives**:
  - 案B（探索範囲と `Objective` の修正を別の spec にする）: `Objective` の引数の扱いを2回設計することになり、境界で食い違いやすい
  - 0.3.x への backport: 同じ種で結果が変わる変更はパッチ版にふさわしくない。また、v0.3.19 からの保守ブランチに、中身の違う2系統の修正が必要になる
  - v0.4.0-rc.0 の削除: PyPI では版番号を使い直せないので、消しても rc.1 以降になる
  - EnsembleRegressor の「各分割で作ったモデルの平均で予測する」方式の維持: sklearn にない機能で、独自の実装が残る。全データで学習し直す sklearn の標準の方式に合わせる
  - LinearModelRegressor・SupportVectorRegressor を非推奨にして残す: 入れ子を解く仕組みと、Ridge・Lasso・ElasticNet・SVR の探索範囲で置き換えられるので、0.4.0 で削除する
  - 固定値の暗黙の上書きを続ける: モデルの値から、利用者が明示したのか既定値のままなのかは判定できない。推奨値は `RecommendedParams` で明示的に適用してもらう

## Scope
- **In**: 開発環境と CI、Python 3.8 への対応、依存の下限の見直し、探索範囲の共通化と追加、`Objective` の引数の扱い、LinearModelRegressor・SupportVectorRegressor の削除、EnsembleRegressor の作り直し、GBDTRegressor の修正、全体のテストと英語の docstring、examples、CHANGELOG、0.4.0-rc.1 と 0.4.0 のリリース、licond への版番号の連絡
- **Out**: 新しい種類の機能（新しい特徴量選択の手法、新しい図など）、分類タスクへの対応、1.0.0 に向けた依存の下限の引き上げ

## Constraints
- **古い環境**: 0.4.x では依存の下限を上げない（0.5.0 からは「Phase 2: 0.5.0」の Constraints に従う）。例外は Boruta>=0.4.3（0.3 は NumPy 1.24 以上で動かない）と optuna>=3.0（`IntDistribution` などを使うため）
- **Python**: 3.8 でも動くことを目指す。CI で通せなければ `>=3.9` に戻し、その理由を記録する
- **scikit-learn**: 下限は 0.24.1。使う API はその版にあるものに限る
- **lightgbm**: 3.x と 4.x の両方で動かす
- **振る舞いの変更**: 0.4.0 は破壊的変更を含む minor。同じ種でも結果が変わることを CHANGELOG とリリースノートに書く
- **進め方**: spec ごとに main からブランチを切る。ひとまとまりの作業が終わったら PR を出し、作者がレビューしてマージする。タグとリリースは作者の確認を取ってから行う

## Boundary Strategy
- **Why this split**:
  - `dev-foundation` は、他のすべての spec が乗る土台（CI、型の方針、依存）なので先に入れる。main の CI は画像を比べるテストで失敗しているので、CI を通すための画像テストの安定化もここで持つ
  - `module-quality` は、モデル以外のモジュール（helpers・logging・metrics・feature_selection・visualize）だけを持つ。optuna まわりと並行して進められる
  - `optuna-tuning` は「何を探索するか（探索範囲）」と「探索の結果をモデルにどう入れるか（`Objective` の引数の扱い）」を両方持つ
  - `ensemble-on-sklearn` は、探索範囲と `ParamDistributions` を使う側なので最後に置く
- **Shared seams to watch**:
  - `src/yikit/models/__init__.py`: 複数の spec が公開する名前を足したり消したりする
  - `src/yikit/models/_ensemble.py` は今の `Objective`（`objective.model`、`fixed_params_`）に依存している。`optuna-tuning` で `Objective` を変えるときは、`ensemble-on-sklearn` で作り直すまで EnsembleRegressor が壊れないようにする（例: `get_best_estimator` を使う形にだけ直す）
  - `Python 3.8` と型の方針: `dev-foundation` で全体を書き換えた後は、各 spec が steering（tech.md）の型の決まりを守る
  - テストの共通 fixture（`tests/conftest.py`）: `module-quality` と `optuna-tuning` の両方が使う。変えるときは互いの影響を確かめる

## Existing Spec Updates
- なし。初期化だけしていた `objective-respect-estimator-params` は `optuna-tuning` に吸収し、ディレクトリを削除した（背景と、加えるテストの一覧は `optuna-tuning` の brief に引き継いだ）

## Direct Implementation Candidates
- [x] gbdt-fix -- GBDTRegressor の修正。LightGBM 3 と 4 の両方で動く callback 方式の early stopping、`X_train` で学習する、`**kwargs` をやめる、LightGBM 4.7 で非推奨になった `eval_set` への対応、テストと英語の docstring。optuna-tuning の探索範囲の表の GBDTRegressor の行の引数で学習できることもテストする（optuna-tuning のテストは GBDTRegressor を学習させない）。`**kwargs` をやめた後も、表の5つの名前（`n_estimators`・`min_child_weight`・`colsample_bytree`・`subsample`・`num_leaves`）はコンストラクタの引数に残す（`apply_params` が名前を検査するため）。修正の範囲が1つのクラスに収まるので spec にしない。Dependencies: dev-foundation
- [x] release-0.4.0 -- examples の更新（作り直した EnsembleRegressor と探索範囲に合わせる。`wip/0.4.0-draft` の simulate の例を参考にする）、CHANGELOG.md の作成、版番号を 0.4.0-rc.1 にしてタグ、確認の後に 0.4.0、licond への版番号の連絡。リリースの作業なので spec にしない。Dependencies: すべての spec と gbdt-fix。2026-10-14 に 0.4.0 を公開した（PR #35・#36）。licond への連絡は残っている
- [ ] release-0.5.0 -- 0.5.0 の CHANGELOG、rc の tag、確認の後に v0.5.0 の tag。setuptools-scm にした後は版を上げる PR が要らない。リリースの作業なので spec にしない。Dependencies: packaging-and-ci、drop-legacy-compat
  - CHANGELOG の材料: 2つの spec の tasks.md の Implementation Notes にある（Python の下限、版番号の決め方、sdist の中身、optional なしの CI の job、上げた依存の下限とその理由、ngboost の上限、optuna-integration が必須になること）
  - packaging-and-ci から引き継ぐ確認: コードを含まない push で CI のテストが飛ばされ成功と出ること、rc と正式版の tag の push で PyPI・TestPyPI・GitHub Release に正しい版で出ること（同じ commit に2つの tag があるときも）、正式版の tag で docs に正しい版が出ること、dependabot の更新が group にまとまること

## Specs (dependency order)
- [x] dev-foundation -- 開発環境と CI（pytest を Python 3.8〜3.14 で。ruff・mypy は手元で実行）、依存の下限と extra の見直し、dependabot、Python 3.8 でも動く型の書き方への全体の書き換え、画像を比べるテストの安定化。Dependencies: none
- [x] module-quality -- モデル以外のモジュール（helpers・logging・metrics・feature_selection・visualize）の英語の docstring とテスト。Dependencies: dev-foundation
- [x] optuna-tuning -- 探索範囲の共通ファイル、入れ子の解決、RecommendedParams、モデルの追加と削除、`Objective` が利用者の引数を上書きしない扱い。Dependencies: dev-foundation
- [x] ensemble-on-sklearn -- EnsembleRegressor を sklearn の Voting・Stacking で作り直す。Dependencies: optuna-tuning
- [ ] packaging-and-ci -- 0.5.0。Python>=3.10、PEP 639 の license と setuptools>=77、setuptools-scm で tag から版を決める、CI・release・docs の workflow と dependabot を python-template にそろえる、ruff・mypy の対象を 3.10 にして 3.8・3.9 向けの書き分けを消す。Dependencies: none
- [ ] drop-legacy-compat -- 0.5.0。scikit-learn<1.0・LightGBM 3・ngboost 0.3・optuna 3.0〜3.6 と古い optuna-integration・seaborn<0.12 のための分岐とテストを消し、依存の下限をそれに必要な分だけ上げる。ngboost の 3.14 での除外と `<0.5.9` の上限を見直す。Dependencies: packaging-and-ci

## Phase 2: 0.5.0

### Overview
0.5.0 では Python の下限を 3.10 に上げ、ビルドと CI を作者の python-template（https://github.com/yu9824/python-template）にそろえます。PEP 639 の license を使い、版番号は setuptools-scm で git の tag から決めます。あわせて、依存の下限を、古い版のための分岐を消すのに必要な分だけ上げ、その分岐を消します。作者の古いオフライン環境（Python 3.8）は 0.4.x の系列が受け持ちます。

### Approach Decision
- **Chosen**: 2つの spec に分ける。ビルド・版番号・workflow・lint の対象の版（`packaging-and-ci`）を先に、ライブラリの依存の下限と古い版のための分岐の削除（`drop-legacy-compat`）を後に進める。CHANGELOG と tag は直接実装の `release-0.5.0` で行う
- **Why**:
  - 設定ファイルの変更と、コードの削除を別々にレビューできる
  - Python の下限（作者が 3.10 と決めた）が決まれば、ビルドと CI はライブラリの下限と独立に進められる
  - ライブラリの下限は、消す分岐から決まるので、分岐の削除と同じ spec に置く
- **Rejected alternatives**:
  - 1つの spec にまとめる: 設定の変更とコードの削除がレビューで混ざる
  - 0.4.x のパッチ版で出す: Python の下限を上げるのは破壊的変更なので、minor を上げる
  - 主要な依存（numpy・pandas・scipy・matplotlib）にも、Python 3.10 向けの wheel がある最初の版を下限として書く: 消す分岐がないのに下限を書くことになる
  - SPEC 0 に合わせる（2年以内の版を下限にする）: 厳しすぎる
  - 下限の版を入れてテストする CI の job を足す: 古い版どうしの組み合わせを保つ手間が大きい。CI の 3.10 の job に少し古い版が入るので、ある程度は確かめられる

### Scope
- **In**
  - Python>=3.10（CI は 3.10〜3.14）
  - PEP 639、setuptools-scm
  - CI・release・docs の workflow と dependabot
  - ruff・mypy の対象の版
  - Python 3.8・3.9 とライブラリの古い版のための分岐の削除
  - 依存の下限と、ngboost の上限・marker の見直し
  - steering（`tech.md`・`product.md`）の更新
  - 0.5.0 の CHANGELOG とリリース
- **Out**
  - 新しい機能、公開 API の変更
  - 下限の版を入れてテストする CI の job
  - 0.4.x の保守用ブランチ（必要になったときに v0.4.0 の tag から切る）
  - ngboost 0.5.11 への対応（`set_params` と `get_params` が入れ子の名前を扱うようになった）。0.5.0 では上限で外し、後の版で対応する（作者が 2026-10-14 に決定）

### Constraints
- **Python**: `requires-python = ">=3.10"`（作者が 2026-10-14 に決定。3.10 のサポートは 2026-10-31 に終わるが、3.10 は残す）
- **依存の下限**: 消す分岐に必要な分だけ上げる（作者が 2026-10-14 に決定）。下限の版で動くことは CI で確かめない
- **古い環境**: 0.5.0 からは対象にしない。古い環境向けの修正は 0.4.x で出す
- **振る舞い**: サポートする版の範囲では、公開 API と計算の結果を変えない
- **CI**: ngboost が 3.14 にも入ると pandas 3 が入らなくなるので、3.14 で `.[test]` だけを入れる job を足す。pandas 3 と、optional の依存がない状態をここで確かめる（作者が 2026-10-14 に決定）
- **sdist**: setuptools-scm は git で管理しているファイルを sdist に入れるので、`MANIFEST.in` で開発用のファイル（`.kiro/`・`.claude/`・`CLAUDE.md`・`.github/` など）を外す。`tests/` と `examples/` は残す（作者が 2026-10-14 に決定）
- **進め方**
  - ブランチを切って作業する。何かを入れるとき（scratchpad への `pip download` や `pip install --target` も含む）は、作者の承認を得る
  - ruff・mypy は手元で実行し、CI では pytest だけを回す
  - ひとまとまりの作業が終わったら PR を出し、作者がレビューしてマージする。tag とリリースは作者の確認を取ってから行う

### Boundary Strategy
- **Why this split**
  - `packaging-and-ci` は、Python の版とビルド・配布・CI を持つ。触るのはおもに設定ファイルで、コードの変更は ruff で機械的に直せる書き分けに限る
  - `drop-legacy-compat` は、ライブラリの版を持つ。依存の宣言とコードの分岐が対になっているので、まとめて扱う
- **Shared seams to watch**
  - `pyproject.toml`
    - `packaging-and-ci` は build-system・`requires-python`・classifiers・license・版（setuptools-scm）、それと `typing_extensions` の行を持つ
    - `drop-legacy-compat` は `[project.dependencies]` と `optional` の extra にあるライブラリの行を持つ。`test`・`dev`・`docs` の extra はどちらも変えない
  - steering の `tech.md`
    - `packaging-and-ci` は Core Technologies の Language の行、型、コードの品質、テストの CI の項、Development Environment、Release を持つ
    - `drop-legacy-compat` は Core Technologies の Framework と Core deps の行、互換性、Key Libraries、Key Technical Decisions の「古い環境」の記述、`product.md` を持つ
  - `src/yikit/visualize/_yyplot.py`: `packaging-and-ci` は `sys.version_info` の分岐と import を、`drop-legacy-compat` は `NDArray` のコメントと docstring を持つ
  - `.github/dependabot.yml`: `packaging-and-ci` が持つ。ngboost>=0.5.11 を無視する設定も `packaging-and-ci` が入れる（上限は `drop-legacy-compat` が決める）
  - CI の job: 下限は API の入った版で決め、CI では確かめない。3.10 の job に入る少し古い版で動くことは、`drop-legacy-compat` の分岐の一覧の「残す」の行で確かめる。3.14 の `.[test]` だけの job（`packaging-and-ci`）は、ngboost が 3.14 に入った後の pandas 3 と、optional なしの状態を確かめる
  - 手元の環境: `drop-legacy-compat` を始める前に、`yikit-dev` を setuptools-scm で入れ直しておく（`packaging-and-ci` の手元のビルドのタスク。入れるには作者の承認が要る）
  - 版番号: setuptools-scm にした後は `src/yikit/__init__.py` に版を手で書かない。`release-0.5.0` は tag だけで版を決める

---
_updated_at: 2026-10-14_
