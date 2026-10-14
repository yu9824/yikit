# Brief: drop-legacy-compat

## Problem
yikit 0.4.x は、作者の古いオフライン環境（Python 3.8、scikit-learn 0.24.1、ngboost 0.3.6、lightgbm 3.2.1）で動くように、依存の下限を低く保っています。そのために、ライブラリの古い版を見分ける分岐と、それを確かめるテストを持っています。

0.5.0 では Python の下限を 3.10 に上げます（packaging-and-ci）。Python 3.10 で入るライブラリはこれらの古い版より新しいので、分岐は保守の手間になるだけです。

ngboost の指定も実態からずれています。
- いまは `ngboost>=0.3.6,<0.5.9; python_version < '3.14'` で、Python 3.14 では ngboost を入れていない
- ngboost は 0.5.11（2026-06）まで出ていて、3.14 にも対応している（requires_python `<3.15`）

## Current State
古い版のための分岐（2026-10-14 時点）:

- **scikit-learn < 1.0**
  - `helpers/_validation.py` の `_sklearn_has_feature_names` と `feature_names_of`: `feature_names_in_` がないとき None を返す
  - `models/_ensemble.py`: 同じ判定を別に持っている（`_sklearn_has_feature_names`、`_feature_names`）
  - `models/_search_space.py` の `_criterion_choices`: 1.0 より前は `"mse"`。1.9 で `"friedman_mse"` が消えるための分岐は、古い版のためではない
  - テスト: `tests/test_helpers.py`（sklearn の版を差し替えて確かめる）、`tests/test_filtermethod.py` の `requires_feature_names_out`、`tests/test_search_cv.py` と `tests/test_ensemble.py` の版の判定
- **optuna 3.0〜3.6 と optuna-integration 4.0.0 以前**
  - `models/_search_cv.py`: `optuna_integration` がないとき `optuna.integration` から import する
  - `OptunaSearchRegressor.predict`: `predict` が property として定義されている古い `OptunaSearchCV` を避けるために上書きしている
  - `models/_optuna.py` の docstring に、同じ import の例がある
  - テスト: `tests/test_search_cv.py` に、古い `OptunaSearchCV` を真似るテストがある
- **LightGBM 3**
  - `models/_gbdt.py`: `verbose` の既定値を見て渡すかどうかを決めている。`best_iteration_` は booster から読む（LightGBM 3 では None のままのため）
  - `visualize/_gbdt.py`: 検証データがないときの `evals_result_`（LightGBM 3 は None、4 は `{}`）
- **ngboost 0.3**
  - `metrics/_ngboost.py`: `_params` を使っている。「0.3.6〜0.5.8 にある」と書いている
  - `models/_ensemble.py` の docstring（ngboost 0.3.x の扱い）
  - `tests/test_metrics.py` の 0.3.6 についての注記
  - `models/_params.py` の ngboost 0.4.0 以降の扱いは今の版のための処理なので、残る
- **seaborn < 0.12**: `visualize/_permutation_importance.py` が `errorbar` と `ci` を書き分けている
- **numpy < 1.21**: `NDArray` を `TYPE_CHECKING` の中で import している
- **docstring と注記**: 「scikit-learn >= 0.24.1 で動く」などの記述が残っている
- **`pyproject.toml`**
  - `scikit-learn>=0.24.1`
  - `ngboost>=0.3.6,<0.5.9; python_version < '3.14'`
  - `optuna>=3.0`
  - `optuna-integration`、`lightgbm`、`seaborn` は下限なし

## Desired Outcome
- **下限は、消した分岐に必要な分だけ上げる**（2026-10-14 に作者が決定）
  - 目安は scikit-learn>=1.0、lightgbm>=4.0、ngboost>=0.4、seaborn>=0.12。optuna と optuna-integration は、消す分岐に合わせて design で決める
  - numpy・pandas・scipy・matplotlib・joblib などは、分岐がなければ下限を書かない（Python 3.10 で実質的に決まる）
  - 各下限の理由を steering と CHANGELOG の材料に残す
- **ngboost**
  - Python 3.14 での除外をやめる
  - 上限 `<0.5.9` は、`log_likelihood`（`_params` を使う）が 0.5.9〜0.5.11 で正しく動くかを確かめてから、上げるか外すかを決める
- **分岐とテストの整理**: 上の分岐と、それだけを確かめるテストを消す。新しい版のための分岐は残す（例: scikit-learn 1.9 の criterion。Python 3.10 には scikit-learn 1.7 までしか入らない）
- **CI**: Python 3.10〜3.14 のすべてで pytest が通る。3.10 の job には少し古い依存が入るので、下限に近い版の確認を兼ねる
- **ruff と mypy**: 手元で通る
- **steering**
  - `tech.md` の「互換性」と「Key Libraries」、`product.md` の「古い環境でも動く」「1.0.0 までは下限を上げない」を、0.5.0 の方針に書き換える
  - 0.4.x の系列が古い環境を受け持つことを書く
- **CHANGELOG の材料**: 0.5.0 の CHANGELOG に書くこと（上げた下限と、その理由）を、tasks.md の Implementation Notes に残す

## Approach
ライブラリごとに分岐を洗い出し、それぞれ次の順に進めます。

1. 下限を決める
2. 分岐を消す
3. テストを直す

下限は「消した分岐が前提にしている版」だけにします。古い版で動くかを CI では確かめないので（2026-10-14 に作者が決定）、下限は、使う API が入った版をリリースノートや changelog で確かめてから決めます。

ngboost の上限は、手元の新しい版で `log_likelihood` のテストを回して決めます。新しい版を入れるには作者の承認が要ります。

## Scope
- **In**
  - `pyproject.toml` の `[project.dependencies]` と `[project.optional-dependencies]` にあるライブラリの下限と上限、ngboost の marker
  - `src/yikit` のうち、ライブラリの古い版のための分岐と docstring の記述
  - それだけを確かめているテスト
  - steering の `tech.md`（互換性・Key Libraries）と `product.md`
- **Out**
  - Python の版、ビルド、版番号の決め方、workflow、ruff・mypy の対象の版（packaging-and-ci が持つ）
  - Python 3.8・3.9 向けの書き分け（`sys.version_info`、`typing_extensions`。packaging-and-ci が持つ）
  - 新しい機能、API の変更。分岐を消すと、サポートする版で結果が変わる場合、その分岐は古い版のためのものではないので残す
  - 下限の版を入れてテストする CI の job（作らないと決めた）
  - CHANGELOG の本文とリリース（release-0.5.0 が持つ）

## Boundary Candidates
- **scikit-learn**: `helpers/_validation.py`、`models/_ensemble.py`（重複している判定の整理を含む）、`models/_search_space.py` とそのテスト
- **optuna と optuna-integration**: `models/_search_cv.py`、`models/_optuna.py` とそのテスト
- **LightGBM**: `models/_gbdt.py`、`visualize/_gbdt.py` とそのテスト
- **ngboost**: `metrics/_ngboost.py`、`models/_ensemble.py` の docstring、`pyproject.toml` の marker と上限、そのテスト
- **seaborn と numpy**: `visualize/_permutation_importance.py`、`NDArray` の import
- **依存の宣言と steering**: `pyproject.toml` の依存の節、`tech.md`、`product.md`

## Out of Boundary
- `[build-system]`、`requires-python`、classifiers、license、版番号（packaging-and-ci）
- `is_installed()` で optional の依存の有無を見る仕組み。依存がなくても `import yikit` が失敗しない作りは残す
- 今の版のための処理（例: `models/_params.py` の ngboost 0.4.0 以降の `set_params` の扱い、scikit-learn 1.6 以降で NGBRegressor が regressor と見なされない扱い）

## Upstream / Downstream
- **Upstream**: packaging-and-ci（Python>=3.10、CI の 3.10〜3.14、ruff・mypy の対象 3.10）。`pyproject.toml` の依存の節は、packaging-and-ci が `typing_extensions` の行を消した後に触る
- **Downstream**
  - release-0.5.0: CHANGELOG に下限の変更を書く
  - licond: 0.5.0 を求めると、上げた下限も満たす必要がある
  - 作者の古い環境: 0.4.x のまま使い続ける

## Existing Spec Touchpoints
- **Extends**: なし。dev-foundation（`Boruta>=0.4.3`、`optuna>=3.0`、optuna-integration からの import）、module-quality（`feature_names_of`）、optuna-tuning（`_criterion_choices`、`OptunaSearchRegressor`）、ensemble-on-sklearn（`_ensemble.py`）、gbdt-fix（`_gbdt.py`）が入れた互換の処理を消す。各 spec は 0.4.0 の記録として残す
- **Adjacent**: packaging-and-ci とは `pyproject.toml` と steering の `tech.md` を共有する。この spec は依存の節と、`tech.md` の互換性・Key Libraries の節だけを触る

## Constraints
- **インストールには作者の承認が要る**: ngboost の新しい版などを手元の環境に入れるときは、作者の承認を得る。scratchpad への `pip download` や `pip install --target` も含む
- **読み取り専用の古い環境には入れない**: `conductivity-prediction-old`（Python 3.8）は読むだけにする
- **公開 API と計算の結果は変えない**: サポートする版の範囲では変えない。変わるのは、入れられる依存の版の範囲だけ
- **ruff と mypy は手元で実行する**: CI では pytest だけを回す
- **下限は過度に厳しくしない**: 分岐を消すのに要らない下限は書かない
