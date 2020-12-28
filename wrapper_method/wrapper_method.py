from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

class WrapperMethod:
    def __init__(self, clf = RandomForestRegressor):
        pass

if __name__ == '__main__':
    from sklearn.datasets import load_boston
    boston = load_boston()
    X = pd.DataFrame(boston['data'], columns = boston['feature_names'])
    y = pd.Series(boston['target'], name = 'PRICE')