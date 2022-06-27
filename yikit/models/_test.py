if __name__ == '__main__':
    from models import NNRegressor
    from ngboost import NGBRegressor
    from models import Objective
    import pandas as pd
    import optuna
    
    from sklearn.utils.estimator_checks import check_estimator
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_diabetes

    # 0.24からclassを入れる機能は削除された．
    # check_estimator(GBDTRegressor())
    # check_estimator(NNRegressor())
    # check_estimator(EnsembleRegressor())
    # print(EnsembleRegressor)
    # check_estimator(SupportVectorRegressor())
    # print(SupportVectorRegressor)
    # check_estimator(LinearModelRegressor())
    # print(LinearModelRegressor)

    diabetes = load_diabetes()
    X = pd.DataFrame(diabetes['data'], columns = diabetes['feature_names'])
    y = pd.Series(diabetes['target'])

    # objective = Objective(RandomForestRegressor(), X, y, scoring = 'neg_mean_squared_error')
    # trial = optuna.trial.FixedTrial({
    #     'min_samples_split': 2,
    #     'max_depth': 10,
    #     'n_estimators': 100,
    # })
    # print(objective(trial))

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    objective = Objective(NGBRegressor(random_state=334, verbose=False), X, y, scoring = 'neg_mean_squared_error')
    study = optuna.create_study(sampler=objective.sampler)
    # trial = optuna.trial.FixedTrial({
    #     'Base__max_depth': 3,
    #     'Base__criterion': 'mse',
    #     'n_estimators': 100,
    #     'minibatch_frac': 1.0,
    # })
    study.optimize(objective, n_trials=1)
    print(objective.get_best_estimator(study))
    

    # estimator = EnsembleRegressor(scoring = ['r2', 'neg_mean_squared_error'], random_state = 334, verbose = 1, boruta = True, opt = False, method = 'stacking')
    # estimator = EnsembleRegressor(estimators = [RandomForestRegressor(), LinearRegression()], scoring = ['r2', 'neg_mean_squared_error'], random_state = 334, boruta = False, opt = False, verbose = 1, method = 'stacking')
    # estimator = EnsembleRegressor(estimators = [RandomForestRegressor(), LGBMRegressor()], scoring = None, random_state = 334, boruta = False, opt = False)
    # estimator.fit(X_train, y_train)
    # print(mean_squared_error(estimator.predict(X_test), y_test, squared = False))
