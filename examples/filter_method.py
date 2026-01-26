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
from yikit.feature_selection import FilterSelector
from sklearn.datasets import load_diabetes
import seaborn as sns
import pandas as pd

# %% [markdown]
# ### サンプルデータの取得

# %%
diabetes = load_diabetes()
X = pd.DataFrame(diabetes['data'], columns = diabetes['feature_names'])
y = pd.Series(diabetes['target'])
X.head()

# %% [markdown]
# ### 特徴量同士の相関をみてみる

# %%
sns.pairplot(X)

# %% [markdown]
# ### 削減してみる

# %%
selector = FilterSelector(r = 0.90)
selector.fit(X)
X_selected = X.loc[:, selector.get_support()]
X_selected.head()

# %% [markdown]
# ### 削減されたペアを表示

# %%
for i in range(selector.corr_.shape[0]):
    for j in range(i, selector.corr_.shape[1]):
        if 1 > selector.corr_[i][j][0] > 0.9:
            print(selector.corr_[i][j], X.columns[i], X.columns[j])

# %% [markdown]
# ### 相関係数

# %%
pd.DataFrame(selector.corr_[:, :, 0], columns = X.columns, index = X.columns)

# %% [markdown]
# ### p値

# %%
pd.DataFrame(selector.corr_[:, :, 1], columns = X.columns, index = X.columns)
