"""
Copyright © 2021 yu9824

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pandas as pd
import pytest
from sklearn.datasets import make_regression


@pytest.fixture(scope="session")
def regression_data():
    """テスト用の回帰データセットを生成

    load_diabetes()と同様のサイズ（442サンプル、10特徴量）で生成
    seedは固定（334）で再現性を確保
    """
    X, y = make_regression(
        n_samples=442,
        n_features=10,
        n_informative=10,
        noise=10.0,
        random_state=334,
    )

    # DataFrameとSeriesに変換（load_diabetes()と同様の形式）
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y)

    return {
        "data": X,
        "target": y,
        "feature_names": feature_names,
        "X": X_df,
        "y": y_series,
    }


@pytest.fixture(scope="session")
def X_regression(regression_data):
    """回帰データの特徴量（DataFrame）"""
    return regression_data["X"]


@pytest.fixture(scope="session")
def y_regression(regression_data):
    """回帰データのターゲット（Series）"""
    return regression_data["y"]
