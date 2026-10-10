from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
import sklearn
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.datasets import make_regression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

from yikit.models._params import apply_params, find_unspecified_random_states

SEED = 334
N_SAMPLES = 40
N_FEATURES = 4


@pytest.fixture(scope="module")
def small_data() -> tuple[np.ndarray, np.ndarray]:
    X, y = make_regression(
        n_samples=N_SAMPLES,
        n_features=N_FEATURES,
        noise=1.0,
        random_state=SEED,
    )
    return X, y


def _is_estimator(value: Any) -> bool:
    return hasattr(value, "get_params") and not isinstance(value, type)


def _held_objects(estimator: Any) -> list[Any]:
    """Return every estimator and RandomState reachable from ``estimator``.

    ``get_params(deep=False)`` is walked recursively by hand because the
    ``get_params`` of ngboost ignores ``deep`` and hides its ``Base``.
    The ``(name, estimator)`` tuples of ``Pipeline`` steps are followed.
    """
    found: list[Any] = []

    def visit(value: Any) -> None:
        if isinstance(value, np.random.RandomState):
            found.append(value)
        elif _is_estimator(value):
            found.append(value)
            for child in value.get_params(deep=False).values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    visit(estimator)
    return found


def _snapshot(estimator: Any) -> list[tuple[Any, dict[str, Any]]]:
    return [
        (obj, dict(obj.get_params(deep=False)))
        for obj in _held_objects(estimator)
        if _is_estimator(obj)
    ]


def _assert_unchanged(
    estimator: Any, snapshot: list[tuple[Any, dict[str, Any]]]
) -> None:
    """Assert that the same objects hold the same parameter objects."""
    current = _snapshot(estimator)
    assert len(current) == len(snapshot)
    for (obj, params), (obj_before, params_before) in zip(current, snapshot):
        assert obj is obj_before
        assert params.keys() == params_before.keys()
        for name, value in params.items():
            assert value is params_before[name], name


def _assert_shares_nothing(result: Any, estimator: Any) -> None:
    """Assert that no estimator or RandomState is shared between the two."""
    original = _held_objects(estimator)
    original_ids = {id(obj) for obj in original}
    shared = [obj for obj in _held_objects(result) if id(obj) in original_ids]
    assert shared == []


def _assert_same_state(
    random_state: np.random.RandomState, state: tuple[Any, ...]
) -> None:
    current = random_state.get_state()
    assert current[0] == state[0]
    np.testing.assert_array_equal(current[1], state[1])
    assert current[2:] == state[2:]


def _scaled_svr() -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("svr", SVR())])


def test_apply_params_sets_constructor_params_of_a_copy():
    estimator = SVR(kernel="linear", C=1.0)
    snapshot = _snapshot(estimator)

    result = apply_params(estimator, {"C": 2.0, "epsilon": 0.2})

    assert result is not estimator
    assert type(result) is SVR
    assert result.get_params() == {
        **estimator.get_params(),
        "C": 2.0,
        "epsilon": 0.2,
    }
    _assert_unchanged(estimator, snapshot)


NESTED_CASES = [
    pytest.param(
        _scaled_svr,
        {"svr__C": 3.0, "scaler__with_mean": False},
        id="Pipeline",
    ),
    pytest.param(
        lambda: make_pipeline(StandardScaler(), SVR()),
        {"svr__C": 3.0, "svr__epsilon": 0.5},
        id="make_pipeline",
    ),
    pytest.param(
        lambda: TransformedTargetRegressor(
            regressor=SVR(), transformer=StandardScaler()
        ),
        {"regressor__C": 3.0, "transformer__with_std": False},
        id="TransformedTargetRegressor",
    ),
    pytest.param(
        lambda: TransformedTargetRegressor(regressor=_scaled_svr()),
        {
            "regressor__svr__C": 3.0,
            "regressor__scaler__with_mean": False,
            "check_inverse": False,
        },
        id="TransformedTargetRegressor-Pipeline",
    ),
    pytest.param(
        lambda: make_pipeline(
            StandardScaler(),
            TransformedTargetRegressor(
                regressor=make_pipeline(StandardScaler(), SVR())
            ),
        ),
        {
            "transformedtargetregressor__regressor__svr__C": 3.0,
            "standardscaler__with_mean": False,
        },
        id="three-levels",
    ),
]


@pytest.mark.parametrize(("make_estimator", "params"), NESTED_CASES)
def test_apply_params_sets_nested_names(make_estimator, params, small_data):
    X, y = small_data
    estimator = make_estimator()
    snapshot = _snapshot(estimator)
    original_params = estimator.get_params(deep=True)

    result = apply_params(estimator, params)

    result_params = result.get_params(deep=True)
    assert result_params.keys() == original_params.keys()
    for name, value in params.items():
        assert result_params[name] == value
    for name, value in original_params.items():
        if name in params or _is_estimator(value) or name.endswith("steps"):
            continue
        assert result_params[name] == value, name
    _assert_unchanged(estimator, snapshot)
    _assert_shares_nothing(result, estimator)
    result.fit(X, y)


@pytest.mark.parametrize(
    ("make_estimator", "head", "get_inner"),
    [
        pytest.param(
            _scaled_svr,
            "svr",
            lambda estimator: estimator.named_steps["svr"],
            id="Pipeline-step",
        ),
        pytest.param(
            lambda: TransformedTargetRegressor(regressor=Ridge()),
            "regressor",
            lambda estimator: estimator.regressor,
            id="TransformedTargetRegressor-regressor",
        ),
    ],
)
def test_apply_params_sets_nested_names_on_a_copy_of_the_given_estimator(
    make_estimator, head, get_inner
):
    estimator = make_estimator()
    snapshot = _snapshot(estimator)
    given = SVR(kernel="linear")
    given_params = given.get_params()

    result = apply_params(estimator, {head: given, f"{head}__C": 5.0})

    inner = get_inner(result)
    assert isinstance(inner, SVR)
    assert inner is not given
    assert inner.get_params() == {**given_params, "C": 5.0}
    assert given.get_params() == given_params
    _assert_unchanged(estimator, snapshot)


def test_apply_params_replaces_a_pipeline_step_with_passthrough():
    estimator = _scaled_svr()
    snapshot = _snapshot(estimator)

    result = apply_params(estimator, {"scaler": "passthrough"})

    assert result.named_steps["scaler"] == "passthrough"
    assert isinstance(result.named_steps["svr"], SVR)
    _assert_unchanged(estimator, snapshot)


@pytest.mark.parametrize("params", [{}, {"svr__C": 2.0}])
def test_apply_params_returns_an_unfitted_copy(params, small_data):
    X, y = small_data
    estimator = make_pipeline(StandardScaler(), SVR()).fit(X, y)

    result = apply_params(estimator, params)

    assert result is not estimator
    with pytest.raises(NotFittedError):
        check_is_fitted(result.named_steps["svr"])
    check_is_fitted(estimator.named_steps["svr"])


def test_random_state_objects_in_params_are_not_shared():
    random_state = np.random.RandomState(5)
    state = random_state.get_state()
    estimator = make_pipeline(PCA(n_components=2), DecisionTreeRegressor())

    result = apply_params(
        estimator,
        {
            "pca__random_state": random_state,
            "decisiontreeregressor__random_state": random_state,
        },
    )

    pca_random_state = result.named_steps["pca"].random_state
    tree_random_state = result.named_steps[
        "decisiontreeregressor"
    ].random_state
    assert pca_random_state is not random_state
    assert tree_random_state is not random_state
    assert pca_random_state is not tree_random_state
    _assert_same_state(pca_random_state, state)
    _assert_same_state(tree_random_state, state)


def test_apply_params_applies_direct_names_before_nested_names():
    # Same order as set_params of scikit-learn: the nested name targets the
    # step of the new steps.
    estimator = _scaled_svr()
    snapshot = _snapshot(estimator)
    new_svr = SVR(kernel="linear")
    new_steps = [("scaler", StandardScaler()), ("svr", new_svr)]

    result = apply_params(estimator, {"steps": new_steps, "svr__C": 5.0})

    svr = result.named_steps["svr"]
    assert svr.get_params() == SVR(kernel="linear", C=5.0).get_params()
    assert svr is not new_svr
    assert new_svr.get_params() == SVR(kernel="linear").get_params()
    _assert_unchanged(estimator, snapshot)


def _skip_without_set_output() -> None:
    if not hasattr(PCA(), "set_output"):
        pytest.skip("set_output needs scikit-learn >= 1.2")


def test_apply_params_keeps_the_set_output_config(small_data):
    _skip_without_set_output()
    X, y = small_data
    estimator = make_pipeline(PCA(n_components=2), SVR()).set_output(
        transform="pandas"
    )
    original_config = estimator.named_steps["pca"]._sklearn_output_config

    result = apply_params(estimator, {"pca__random_state": 0})

    pca = result.named_steps["pca"]
    assert pca.random_state == 0
    assert pca._sklearn_output_config == {"transform": "pandas"}
    assert pca._sklearn_output_config is not original_config
    assert isinstance(pca.fit_transform(X), pd.DataFrame)
    result.fit(X, y)


def test_apply_params_keeps_pandas_output_needed_by_a_later_step(
    small_data,
):
    _skip_without_set_output()
    X, y = small_data
    frame = pd.DataFrame(X, columns=["a", "b", "c", "d"])
    estimator = make_pipeline(
        QuantileTransformer(n_quantiles=10),
        ColumnTransformer([("keep", "passthrough", ["a", "b"])]),
        SVR(),
    ).set_output(transform="pandas")

    result = apply_params(estimator, {"quantiletransformer__random_state": 0})

    # ColumnTransformer selects columns by name, so it needs the pandas
    # output of QuantileTransformer.
    result.fit(frame, y)
    assert result.predict(frame).shape == (N_SAMPLES,)


def test_apply_params_keeps_metadata_routing_requests():
    if "enable_metadata_routing" not in sklearn.get_config():
        pytest.skip("metadata routing needs scikit-learn >= 1.3")
    with sklearn.config_context(enable_metadata_routing=True):
        estimator = make_pipeline(
            StandardScaler(), SVR().set_fit_request(sample_weight=True)
        )

        result = apply_params(estimator, {"svr__C": 2.0})

        svr = result.named_steps["svr"]
        assert svr.C == 2.0
        assert svr.get_metadata_routing().fit.requests == {
            "sample_weight": True
        }


@pytest.mark.parametrize(
    ("make_estimator", "params", "match"),
    [
        pytest.param(SVR, {"foo": 1}, r"'foo'.*SVR.*'foo'", id="top-level"),
        pytest.param(
            _scaled_svr,
            {"svr__foo": 1},
            r"'svr__foo'.*SVR.*'foo'",
            id="Pipeline-step",
        ),
        pytest.param(
            _scaled_svr,
            {"nostep__C": 1.0},
            r"'nostep__C'.*Pipeline.*'nostep'",
            id="Pipeline-unknown-step",
        ),
        pytest.param(
            lambda: TransformedTargetRegressor(regressor=_scaled_svr()),
            {"regressor__svr__foo": 1},
            r"'regressor__svr__foo'.*SVR.*'foo'",
            id="TransformedTargetRegressor-Pipeline",
        ),
        pytest.param(
            SVR,
            {"C__foo": 1},
            r"'C__foo'.*'C'.*SVR",
            id="nested-in-a-non-estimator",
        ),
        pytest.param(
            TransformedTargetRegressor,
            {"regressor__C": 1.0},
            r"'regressor__C'.*'regressor'.*TransformedTargetRegressor",
            id="nested-in-None",
        ),
    ],
)
def test_apply_params_rejects_unknown_names(make_estimator, params, match):
    estimator = make_estimator()
    snapshot = _snapshot(estimator)

    with pytest.raises(ValueError, match=match):
        apply_params(estimator, params)

    _assert_unchanged(estimator, snapshot)


def _small_ngb(ngboost: Any, **params: Any) -> Any:
    # Base is always given: the default Base of ngboost is one module-global
    # object shared by every NGBRegressor.
    params.setdefault("Base", DecisionTreeRegressor(max_depth=3))
    return ngboost.NGBRegressor(n_estimators=10, verbose=False, **params)


def test_ngboost_nested_names_and_random_state_take_effect(small_data):
    ngboost = pytest.importorskip("ngboost")
    X, y = small_data
    estimator = _small_ngb(ngboost)
    snapshot = _snapshot(estimator)
    # deep=False: ngboost < 0.4 returns the nested Base__* names otherwise.
    original_params = estimator.get_params(deep=False)
    original_base_params = estimator.Base.get_params()

    result = apply_params(
        estimator,
        {"Base__max_depth": 5, "random_state": 7, "minibatch_frac": 0.5},
    )

    assert type(result) is type(estimator)
    assert result.Base.get_params() == {
        **original_base_params,
        "max_depth": 5,
    }
    assert result.minibatch_frac == 0.5
    assert isinstance(result.random_state, np.random.RandomState)
    _assert_same_state(
        result.random_state, np.random.RandomState(7).get_state()
    )
    result_params = result.get_params(deep=False)
    assert result_params.keys() == original_params.keys()
    for name, value in original_params.items():
        if name not in {"Base", "random_state", "minibatch_frac"}:
            assert result_params[name] == value, name
    assert estimator.Base.get_params() == original_base_params
    assert estimator.minibatch_frac == 1.0
    _assert_unchanged(estimator, snapshot)
    _assert_shares_nothing(result, estimator)

    result.fit(X, y)
    assert result.predict(X).shape == (N_SAMPLES,)


@pytest.mark.parametrize(
    "explicit", [False, True], ids=["unspecified", "explicit"]
)
def test_ngboost_copy_shares_no_random_state_with_the_input(
    explicit, small_data
):
    ngboost = pytest.importorskip("ngboost")
    X, y = small_data
    if explicit:
        estimator = _small_ngb(
            ngboost,
            Base=DecisionTreeRegressor(
                max_depth=3, random_state=np.random.RandomState(1)
            ),
            random_state=np.random.RandomState(2),
        )
    else:
        estimator = _small_ngb(ngboost)
        # ngboost turns random_state=None into the global RandomState.
        assert estimator.random_state is check_random_state(None)
    state = estimator.random_state.get_state()
    snapshot = _snapshot(estimator)

    result = apply_params(
        estimator, {"Base__max_depth": 5, "minibatch_frac": 0.5}
    )

    assert result.Base is not estimator.Base
    assert result.random_state is not estimator.random_state
    assert result.random_state is not check_random_state(None)
    _assert_same_state(result.random_state, state)
    if explicit:
        base_state = estimator.Base.random_state.get_state()
        assert result.Base.random_state is not estimator.Base.random_state
        _assert_same_state(result.Base.random_state, base_state)
    _assert_shares_nothing(result, estimator)

    result.fit(X, y)
    if explicit:
        # Fitting the copy does not advance the RandomState of the input.
        # (With random_state=None, fitting Base draws from the global
        # RandomState by design, so it is not checked.)
        _assert_same_state(estimator.random_state, state)
        _assert_same_state(estimator.Base.random_state, base_state)
    _assert_unchanged(estimator, snapshot)


# Cloning the default Base warns on scikit-learn >= 1.9 ("friedman_mse").
@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_ngboost_default_base_is_not_modified():
    ngboost = pytest.importorskip("ngboost")
    from ngboost.learners import default_tree_learner

    default_params = default_tree_learner.get_params()
    estimator = ngboost.NGBRegressor(verbose=False)
    assert estimator.Base is default_tree_learner

    result = apply_params(estimator, {"Base__max_depth": 7, "random_state": 0})

    assert result.Base.max_depth == 7
    assert result.Base is not default_tree_learner
    assert estimator.Base is default_tree_learner
    assert default_tree_learner.get_params() == default_params


def test_ngboost_random_state_objects_in_params_are_not_shared():
    ngboost = pytest.importorskip("ngboost")
    random_state = np.random.RandomState(5)
    state = random_state.get_state()
    estimator = _small_ngb(ngboost)

    result = apply_params(
        estimator,
        {"random_state": random_state, "Base__random_state": random_state},
    )

    assert result.random_state is not random_state
    assert result.Base.random_state is not random_state
    assert result.random_state is not result.Base.random_state
    _assert_same_state(result.random_state, state)
    _assert_same_state(result.Base.random_state, state)


@pytest.mark.parametrize(
    ("params", "match"),
    [
        pytest.param(
            {"foo": 1}, r"'foo'.*NGBRegressor.*'foo'", id="top-level"
        ),
        pytest.param(
            {"Base__foo": 1},
            r"'Base__foo'.*DecisionTreeRegressor.*'foo'",
            id="Base",
        ),
    ],
)
def test_ngboost_rejects_unknown_names(params, match):
    ngboost = pytest.importorskip("ngboost")
    estimator = _small_ngb(ngboost)
    snapshot = _snapshot(estimator)

    with pytest.raises(ValueError, match=match):
        apply_params(estimator, params)

    _assert_unchanged(estimator, snapshot)


def test_lightgbm_accepts_extra_keyword_arguments(small_data):
    lightgbm = pytest.importorskip("lightgbm")
    X, y = small_data
    estimator = lightgbm.LGBMRegressor(n_estimators=5, n_jobs=2)
    snapshot = _snapshot(estimator)

    result = apply_params(
        estimator, {"num_leaves": 16, "max_bin": 63, "verbosity": -1}
    )

    result_params = result.get_params()
    assert result_params["num_leaves"] == 16
    assert result_params["max_bin"] == 63
    assert result_params["verbosity"] == -1
    assert result_params["n_jobs"] == 2
    assert "max_bin" not in estimator.get_params()
    _assert_unchanged(estimator, snapshot)
    result.fit(X, y)


def _find_as_set(estimator: Any) -> set[str]:
    names = find_unspecified_random_states(estimator)
    assert isinstance(names, list)
    assert len(names) == len(set(names)), names
    return set(names)


@pytest.mark.parametrize(
    ("random_state", "expected"),
    [
        pytest.param(None, {"random_state"}, id="None"),
        # ngboost turns None into this object, so it means the same.
        pytest.param(
            check_random_state(None), {"random_state"}, id="global-RandomState"
        ),
        pytest.param(0, set(), id="int-0"),
        pytest.param(7, set(), id="int"),
        pytest.param(np.random.RandomState(7), set(), id="RandomState"),
    ],
)
def test_find_unspecified_random_states_by_value(random_state, expected):
    estimator = DecisionTreeRegressor(random_state=random_state)

    assert _find_as_set(estimator) == expected
    assert estimator.random_state is random_state


@pytest.mark.parametrize(
    "make_estimator",
    [
        pytest.param(SVR, id="SVR"),
        pytest.param(PLSRegression, id="PLSRegression"),
        pytest.param(_scaled_svr, id="Pipeline-SVR"),
        pytest.param(
            lambda: TransformedTargetRegressor(
                regressor=make_pipeline(StandardScaler(), PLSRegression())
            ),
            id="TransformedTargetRegressor-PLSRegression",
        ),
    ],
)
def test_find_unspecified_random_states_skips_models_without_random_state(
    make_estimator,
):
    assert find_unspecified_random_states(make_estimator()) == []


@pytest.mark.parametrize(
    ("make_estimator", "expected"),
    [
        pytest.param(
            lambda: make_pipeline(
                QuantileTransformer(n_quantiles=10),
                PCA(n_components=2, random_state=3),
                DecisionTreeRegressor(),
            ),
            {
                "quantiletransformer__random_state",
                "decisiontreeregressor__random_state",
            },
            id="Pipeline",
        ),
        pytest.param(
            lambda: TransformedTargetRegressor(
                regressor=DecisionTreeRegressor(),
                transformer=QuantileTransformer(n_quantiles=10),
            ),
            {"regressor__random_state", "transformer__random_state"},
            id="TransformedTargetRegressor",
        ),
        pytest.param(
            lambda: TransformedTargetRegressor(
                regressor=make_pipeline(
                    QuantileTransformer(n_quantiles=10),
                    RandomForestRegressor(n_estimators=5),
                ),
                transformer=QuantileTransformer(
                    n_quantiles=10, random_state=np.random.RandomState(2)
                ),
            ),
            {
                "regressor__quantiletransformer__random_state",
                "regressor__randomforestregressor__random_state",
            },
            id="TransformedTargetRegressor-Pipeline",
        ),
    ],
)
def test_find_unspecified_random_states_returns_prefixed_names(
    make_estimator, expected
):
    estimator = make_estimator()
    snapshot = _snapshot(estimator)

    assert _find_as_set(estimator) == expected
    _assert_unchanged(estimator, snapshot)


def test_ngboost_default_random_states_are_unspecified():
    ngboost = pytest.importorskip("ngboost")
    from ngboost.learners import default_tree_learner

    default_params = default_tree_learner.get_params()
    global_state = check_random_state(None).get_state()
    # The default Base is safe here: nothing is set on the estimator.
    estimator = ngboost.NGBRegressor()
    assert estimator.random_state is check_random_state(None)
    snapshot = _snapshot(estimator)

    assert _find_as_set(estimator) == {"random_state", "Base__random_state"}
    assert _find_as_set(ngboost.NGBRegressor(random_state=7)) == {
        "Base__random_state"
    }

    _assert_unchanged(estimator, snapshot)
    assert estimator.Base is default_tree_learner
    assert default_tree_learner.get_params() == default_params
    _assert_same_state(check_random_state(None), global_state)


@pytest.mark.parametrize(
    ("params", "expected"),
    [
        pytest.param(
            {}, {"random_state", "Base__random_state"}, id="unspecified"
        ),
        pytest.param(
            {"random_state": 7}, {"Base__random_state"}, id="random_state"
        ),
        pytest.param(
            {"Base": DecisionTreeRegressor(max_depth=3, random_state=1)},
            {"random_state"},
            id="Base",
        ),
        pytest.param(
            {
                "random_state": np.random.RandomState(2),
                "Base": DecisionTreeRegressor(
                    max_depth=3, random_state=np.random.RandomState(1)
                ),
            },
            set(),
            id="both-RandomState",
        ),
    ],
)
def test_ngboost_unspecified_random_states(params, expected):
    ngboost = pytest.importorskip("ngboost")
    estimator = _small_ngb(ngboost, **params)
    snapshot = _snapshot(estimator)

    assert _find_as_set(estimator) == expected
    _assert_unchanged(estimator, snapshot)


def test_ngboost_unspecified_random_states_in_nested_estimators():
    ngboost = pytest.importorskip("ngboost")
    estimator = TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), _small_ngb(ngboost)),
        transformer=QuantileTransformer(n_quantiles=10),
    )
    snapshot = _snapshot(estimator)

    assert _find_as_set(estimator) == {
        "regressor__ngbregressor__random_state",
        "regressor__ngbregressor__Base__random_state",
        "transformer__random_state",
    }
    _assert_unchanged(estimator, snapshot)
