from scipy.stats import pearsonr
import pandas as pd
import numpy as np

# 例用
from sklearn.datasets import load_boston

'''
X: pd.DataFrame or 2次元のnp.ndarray．すでに作成された特徴量空間．
r: 相関係数を高いとする閾値．これより高いと相関が高いとして，削除対象とする．
alpha: 有意水準．これよりp値が低い時，相関係数の値が信頼に足ると判断．特別な理由がない限りはこれを変えないのがベター．
'''

def filter_method(X, r = 0.9, alpha = 0.05):
    # すでに作成された特徴量空間をXとして読み込み，念のため，np.ndarrayに変換
    X = np.array(X)

    # よくつかうので列数を変数で保存
    length = X.shape[1]

    # 相関係数とそのp値を入れるリストの鋳型を作成
    # [i, j, 0]が相関係数， [i, j, 1]がp値
    corr = np.zeros(2 * length ** 2, dtype = float).reshape(length, length, 2)

    # i番目の要素は，i番目と相関係数がcorr以上かつp値が有意水準alpha未満のものの集合．　これの鋳型を作成
    pair = [set() for _ in range(length)]

    for i in range(length):
        for j in range(i, length):
            if i == j:  # 一緒の時はそもそも相関係数1, p値0は自明なので例外処理
                corr[i, i] = [1.0, 0.0]
            else:
                corr[i, j] = list(pearsonr(X[:, i], X[:, j]))
                corr[j, i] = corr[i, j]
                if corr[i, j, 0] > r and corr[i, j, 1] < alpha:    # 相関係数が高く， かつp値が小さい (信頼に足る) とき．
                    # i, j番目の要素の集合にそれぞれj, iを追加
                    pair[i].add(j)
                    pair[j].add(i)

    # 何個相関係数が高い係数が存在するのかの値を求める．
    def delete(pair, boolean = np.ones(length, dtype = bool)):
        # 相関係数が高いものがいくつあるのかのnp.ndarrayを作成
        order_pair = np.array([len(s) for s in pair])

        # これがすべて0のとき，もう削除は終わってるので終了処理
        if np.sum(order_pair) == 0:
            return boolean

        # 一番相関係数が高いペア数を多く持っているものの要素番号 (index) を取得
        i_max = np.argmax(order_pair)

        # 相関係数高いペアからその番号を削除
        pair = [p.difference({i_max}) for p in pair]

        # その集合は削除かつ，その特徴量はbooleanをFalseに．
        pair[i_max] = set()
        boolean[i_max] = False
        return delete(pair, boolean)

    boolean = delete(pair)

    return boolean, corr

if __name__ == '__main__':
    boston = load_boston()
    X = pd.DataFrame(boston['data'], columns = boston['feature_names'])
    boolean, corr = filter_method(X, r = 0.9)
    print(boolean, corr)
