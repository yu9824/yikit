# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.20.0
#   kernelspec:
#     display_name: Python 3.9.13 ('yikit')
#     language: python
#     name: python3
# ---

# %% [markdown]
# ### モジュールのインポート

# %%
from yikit.feature_selection import BorutaPy
from sklearn.datasets import load_diabetes
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# %% [markdown]
# ### サンプルデータの取得

# %%
diabetes = load_diabetes()
X = pd.DataFrame(diabetes['data'], columns = diabetes['feature_names'])
y = pd.Series(diabetes['target'])
X.head()

# %% [markdown]
# ### 削減してみる

# %%
selector = BorutaPy(RandomForestRegressor(n_jobs=-1, random_state=334), random_state=334, n_jobs=-1, perc='auto', verbose=1)
selector.fit(X, y)
X_selected = X.loc[:, selector.get_support()]
X_selected.head()

# %%
selector.get_support()
