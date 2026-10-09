from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from optuna.distributions import (
    CategoricalDistribution,
    FloatDistribution,
    IntDistribution,
)
from sklearn.cross_decomposition import PLSRegression
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor

import yikit.models._search_space as search_space_module
from yikit.helpers import is_installed
from yikit.models._search_space import _criterion_choices, get_search_space

if TYPE_CHECKING:
    from optuna.distributions import BaseDistribution

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


def _svr_space() -> dict[str, BaseDistribution]:
    return {
        "C": FloatDistribution(2**-5, 2**10, log=True),
        "epsilon": FloatDistribution(2**-10, 2**0, log=True),
    }


def _alpha_space() -> dict[str, BaseDistribution]:
    return {"alpha": FloatDistribution(0.1, 10.0, log=True)}


def _lightgbm_space() -> dict[str, BaseDistribution]:
    return {
        "n_estimators": IntDistribution(10, 1000, log=True),
        "min_child_weight": FloatDistribution(0.001, 10.0, log=True),
        "colsample_bytree": FloatDistribution(0.6, 0.95),
        "subsample": FloatDistribution(0.6, 0.95),
        "num_leaves": IntDistribution(2**3, 2**9, log=True),
    }


def _ngboost_space() -> dict[str, BaseDistribution]:
    return {
        "Base__max_depth": IntDistribution(2, 100),
        "Base__criterion": CategoricalDistribution(_criterion_choices()),
        "n_estimators": IntDistribution(10, 1000, log=True),
        "minibatch_frac": FloatDistribution(0.5, 1.0),
    }


SKLEARN_SPACES = [
    pytest.param(
        RandomForestRegressor(),
        {
            "min_samples_split": IntDistribution(2, 16),
            "max_depth": IntDistribution(10, 100),
            "n_estimators": IntDistribution(10, 1000, log=True),
        },
        id="RandomForestRegressor",
    ),
    pytest.param(SVR(), _svr_space(), id="SVR"),
    pytest.param(LinearSVR(), _svr_space(), id="LinearSVR"),
    pytest.param(
        MLPRegressor(),
        {
            "hidden_layer_sizes": IntDistribution(50, 300),
            "alpha": FloatDistribution(1e-5, 1e-3, log=True),
            "learning_rate_init": FloatDistribution(1e-5, 1e-3, log=True),
        },
        id="MLPRegressor",
    ),
    pytest.param(
        PLSRegression(),
        {"n_components": IntDistribution(1, 10)},
        id="PLSRegression",
    ),
    pytest.param(Lasso(), _alpha_space(), id="Lasso"),
    pytest.param(
        ElasticNet(),
        {**_alpha_space(), "l1_ratio": FloatDistribution(0.1, 1.0)},
        id="ElasticNet",
    ),
    pytest.param(Ridge(), _alpha_space(), id="Ridge"),
]


class _CustomSVR(SVR):
    """Subclass of SVR that must match the SVR row."""


class _CustomLasso(Lasso):
    """Subclass of Lasso that must match the Lasso row."""


class _CustomElasticNet(ElasticNet):
    """Subclass of ElasticNet that must match the ElasticNet row."""


class _CustomRandomForest(RandomForestRegressor):
    """Subclass of RandomForestRegressor that must match its row."""


def _registered_estimators() -> list[Any]:
    estimators: list[Any] = [param.values[0] for param in SKLEARN_SPACES]
    if is_installed("lightgbm"):
        from lightgbm import LGBMRegressor

        from yikit.models._gbdt import GBDTRegressor

        estimators += [LGBMRegressor(), GBDTRegressor()]
    if is_installed("ngboost"):
        from ngboost import NGBRegressor

        estimators.append(NGBRegressor())
    return estimators


def _boundary_params(
    space: dict[str, BaseDistribution],
) -> list[dict[str, Any]]:
    """Return parameter sets using every bound and every choice of a row.

    The first set uses every lower bound (and the first choice), the second
    uses every upper bound (and the last choice), and the rest put each
    choice of each categorical distribution on top of the first set.
    """
    lows: dict[str, Any] = {}
    highs: dict[str, Any] = {}
    for name, distribution in space.items():
        if isinstance(distribution, CategoricalDistribution):
            lows[name] = distribution.choices[0]
            highs[name] = distribution.choices[-1]
        else:
            assert isinstance(
                distribution, (IntDistribution, FloatDistribution)
            )
            lows[name] = distribution.low
            highs[name] = distribution.high
    params = [lows, highs]
    for name, distribution in space.items():
        if isinstance(distribution, CategoricalDistribution):
            params += [
                {**lows, name: choice} for choice in distribution.choices
            ]
    return params


def _assert_fits(model: Any, X: np.ndarray, y: np.ndarray) -> None:
    model.fit(X, y)
    assert np.isfinite(model.predict(X)).all()


@pytest.mark.parametrize(("estimator", "expected"), SKLEARN_SPACES)
def test_search_space_of_sklearn_models(estimator, expected):
    space = get_search_space(estimator)

    assert space == expected
    # The order of the names decides the order of the suggestions.
    assert space is not None
    assert list(space) == list(expected)


def test_search_space_of_lightgbm_models():
    lightgbm = pytest.importorskip("lightgbm")
    from yikit.models._gbdt import GBDTRegressor

    for estimator in (lightgbm.LGBMRegressor(), GBDTRegressor()):
        space = get_search_space(estimator)

        assert space == _lightgbm_space()
        assert space is not None
        assert list(space) == list(_lightgbm_space())


def test_search_space_of_ngboost():
    ngboost = pytest.importorskip("ngboost")

    space = get_search_space(ngboost.NGBRegressor())

    assert space == _ngboost_space()
    assert space is not None
    assert list(space) == list(_ngboost_space())


def test_lasso_row_is_matched_before_elastic_net():
    lasso_space = get_search_space(Lasso())
    elastic_net_space = get_search_space(ElasticNet())

    assert issubclass(Lasso, ElasticNet)
    assert lasso_space is not None
    assert "l1_ratio" not in lasso_space
    assert elastic_net_space is not None
    assert "l1_ratio" in elastic_net_space


@pytest.mark.parametrize(
    ("estimator", "parent"),
    [
        pytest.param(_CustomSVR(), SVR(), id="SVR"),
        pytest.param(_CustomLasso(), Lasso(), id="Lasso"),
        pytest.param(_CustomElasticNet(), ElasticNet(), id="ElasticNet"),
        pytest.param(
            _CustomRandomForest(),
            RandomForestRegressor(),
            id="RandomForestRegressor",
        ),
    ],
)
def test_subclass_matches_the_row_of_its_parent(estimator, parent):
    space = get_search_space(estimator)

    assert space is not None
    assert space == get_search_space(parent)


@pytest.mark.parametrize(
    "estimator",
    [DecisionTreeRegressor(), LinearRegression(), KNeighborsRegressor()],
    ids=lambda estimator: type(estimator).__name__,
)
def test_unregistered_model_has_no_search_space(estimator):
    assert get_search_space(estimator) is None
    assert get_search_space(estimator, n_features=N_FEATURES) is None


def test_search_spaces_have_no_fixed_values():
    for estimator in _registered_estimators():
        space = get_search_space(estimator)

        assert space is not None
        for name in space:
            assert name.split("__")[-1] not in {"n_jobs", "random_state"}


@pytest.mark.parametrize(
    ("n_features", "expected_high"),
    [(None, 10), (1, 1), (3, 3), (10, 10), (25, 10)],
)
def test_pls_n_components_is_clipped_by_n_features(n_features, expected_high):
    space = get_search_space(PLSRegression(), n_features=n_features)

    assert space == {"n_components": IntDistribution(1, expected_high)}


def test_n_features_does_not_clip_other_rows():
    space = get_search_space(RandomForestRegressor(), n_features=3)

    assert space == get_search_space(RandomForestRegressor())


@pytest.mark.parametrize("n_features", [0, -1])
def test_n_features_must_be_positive(n_features):
    with pytest.raises(ValueError, match="n_features"):
        get_search_space(PLSRegression(), n_features=n_features)


@pytest.mark.parametrize(
    ("estimator", "n_features"),
    [
        pytest.param(ElasticNet(), None, id="ElasticNet"),
        pytest.param(PLSRegression(), None, id="PLSRegression"),
        pytest.param(PLSRegression(), 3, id="PLSRegression-clipped"),
    ],
)
def test_returns_new_objects_per_call(estimator, n_features):
    first = get_search_space(estimator, n_features=n_features)
    second = get_search_space(estimator, n_features=n_features)

    assert first is not None
    assert second is not None
    assert first == second
    assert first is not second
    for name in first:
        assert first[name] is not second[name]

    first.clear()
    assert get_search_space(estimator, n_features=n_features) == second


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("0.24.1", ["mse", "friedman_mse"]),
        ("0.24.2", ["mse", "friedman_mse"]),
        ("1.0", ["squared_error", "friedman_mse"]),
        ("1.0.2", ["squared_error", "friedman_mse"]),
        ("1.3.2", ["squared_error", "friedman_mse"]),
        ("1.8.0", ["squared_error", "friedman_mse"]),
        ("1.9.0rc1", ["squared_error"]),
        ("1.9.0", ["squared_error"]),
        ("1.9.1", ["squared_error"]),
        ("1.10.dev0", ["squared_error"]),
        ("2.0.0", ["squared_error"]),
    ],
)
def test_criterion_choices_by_sklearn_version(version, expected):
    assert _criterion_choices(version) == expected


def test_criterion_choices_default_to_installed_sklearn():
    import sklearn

    assert _criterion_choices() == _criterion_choices(sklearn.__version__)


def test_criterion_choices_reject_unparsable_version():
    with pytest.raises(ValueError, match="scikit-learn version"):
        _criterion_choices("unknown")


@pytest.mark.parametrize("criterion", _criterion_choices())
def test_criterion_choices_fit_without_future_warning(criterion, small_data):
    X, y = small_data

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        model = DecisionTreeRegressor(criterion=criterion, random_state=SEED)
        _assert_fits(model, X, y)


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
@pytest.mark.parametrize(
    ("model_class", "fixed_kwargs"),
    [
        pytest.param(
            RandomForestRegressor,
            {"random_state": SEED, "n_jobs": 1},
            id="RandomForestRegressor",
        ),
        pytest.param(SVR, {}, id="SVR"),
        pytest.param(LinearSVR, {"random_state": SEED}, id="LinearSVR"),
        pytest.param(MLPRegressor, {"random_state": SEED}, id="MLPRegressor"),
        pytest.param(PLSRegression, {}, id="PLSRegression"),
        pytest.param(Lasso, {"random_state": SEED}, id="Lasso"),
        pytest.param(ElasticNet, {"random_state": SEED}, id="ElasticNet"),
        pytest.param(Ridge, {}, id="Ridge"),
    ],
)
def test_sklearn_rows_fit_at_their_bounds(
    model_class, fixed_kwargs, small_data
):
    X, y = small_data
    space = get_search_space(model_class(), n_features=X.shape[1])
    assert space is not None

    for params in _boundary_params(space):
        _assert_fits(model_class(**fixed_kwargs, **params), X, y)


def test_lightgbm_row_fits_at_its_bounds(small_data):
    lightgbm = pytest.importorskip("lightgbm")
    X, y = small_data
    space = get_search_space(lightgbm.LGBMRegressor())
    assert space is not None

    # GBDTRegressor shares this row but is not fitted here (see gbdt-fix).
    for params in _boundary_params(space):
        model = lightgbm.LGBMRegressor(
            random_state=SEED, n_jobs=1, verbose=-1, **params
        )
        _assert_fits(model, X, y)


def test_ngboost_row_fits_at_its_bounds(small_data):
    ngboost = pytest.importorskip("ngboost")
    X, y = small_data
    space = get_search_space(ngboost.NGBRegressor())
    assert space is not None

    prefix = "Base__"
    for params in _boundary_params(space):
        base = DecisionTreeRegressor(
            random_state=SEED,
            **{
                name[len(prefix) :]: value
                for name, value in params.items()
                if name.startswith(prefix)
            },
        )
        model = ngboost.NGBRegressor(
            Base=base,
            random_state=SEED,
            verbose=False,
            **{
                name: value
                for name, value in params.items()
                if not name.startswith(prefix)
            },
        )
        _assert_fits(model, X, y)


def test_optional_rows_are_left_out_when_not_installed():
    optional_estimators = []
    if is_installed("lightgbm"):
        from lightgbm import LGBMRegressor

        optional_estimators.append(LGBMRegressor())
    if is_installed("ngboost"):
        from ngboost import NGBRegressor

        optional_estimators.append(NGBRegressor())

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            search_space_module,
            "is_installed",
            lambda name: name not in {"lightgbm", "ngboost"},
        )
        monkeypatch.setattr(
            search_space_module,
            "_SEARCH_SPACES",
            search_space_module._build_search_spaces(),
        )

        assert get_search_space(SVR()) == _svr_space()
        for estimator in optional_estimators:
            assert get_search_space(estimator) is None
