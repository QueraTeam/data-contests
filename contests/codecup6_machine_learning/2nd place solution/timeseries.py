# ## XGBoost

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import plot_importance, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error
import optuna
from functools import partial

train = pd.read_csv("../input/code-cup-cabs/cab/train.csv")
test = pd.read_csv("../input/code-cup-cabs/cab/test.csv")

train['time'] = pd.to_datetime(train['time'], format="%Y-%m-%d %H:%M:%S")
test['time'] = pd.to_datetime(test['time'], format="%Y-%m-%d %H:%M:%S")

df = train.set_index("time")


split_date = '2021-11-01'
df_train = df.loc[df.index <= split_date].copy()
df_test = df.loc[df.index > split_date].copy()

def create_features(df, label=None):
    """
    Creates time series features from datetime index
    """
    df['date'] = df.index
    df['minute'] = df['date'].dt.minute
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.weekofyear
    
    X = df[['minute', 'hour','dayofweek','quarter','month','year',
           'dayofyear','dayofmonth','weekofyear']]
    if label:
        y = df[label]
        return X, y
    return X


X_train, y_train = create_features(df_train, label='y')
X_test, y_test = create_features(df_test, label='y')


def run(trial, xtrain, ytrain, xvalid, yvalid):
    learning_rate = trial.suggest_float("learning_rate", 1e-2, 0.25, log=True)
    reg_lambda = trial.suggest_loguniform("reg_lambda", 1e-8, 100.0)
    reg_alpha = trial.suggest_loguniform("reg_alpha", 1e-8, 100.0)
    subsample = trial.suggest_float("subsample", 0.1, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.1, 1.0)
    max_depth = trial.suggest_int("max_depth", 1, 7)
    
    model = xgb.XGBRegressor(
        random_state=42,
        tree_method="gpu_hist",
        gpu_id=1,
        n_estimators=10000,
        predictor="gpu_predictor",
        learning_rate=learning_rate,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        max_depth=max_depth,
    )
    model.fit(xtrain, ytrain, early_stopping_rounds=300, eval_set=[(xvalid, yvalid)], verbose=1000)
    preds_valid = model.predict(xvalid)
    rmse = mean_squared_error(yvalid, preds_valid, squared=False)
    return rmse

opt_fun = partial(
    run,
    xtrain=X_train,
    ytrain=y_train,
    xvalid=X_test,
    yvalid=y_test
)

study = optuna.create_study(direction="minimize")
study.optimize(opt_fun, n_trials=200)
print(study.best_params)


def generate_predictions(params, xtrain, ytrain, xvalid, yvalid, df_test):    
    xtest = df_test.copy()

    model = xgb.XGBRegressor(
        random_state=42,
        tree_method="gpu_hist",
        gpu_id=1,
        n_estimators=10000,
        predictor="gpu_predictor",
        **params,
    )
    model.fit(xtrain, ytrain, early_stopping_rounds=300, eval_set=[(xvalid, yvalid)], verbose=1000)
    preds_valid = model.predict(xvalid)
    test_preds = model.predict(xtest)
    rmse = mean_squared_error(yvalid, preds_valid, squared=False)
    print(rmse)
    return test_preds

test_feat = create_features(test.set_index('time'))


test_preds = generate_predictions(study.best_params, X_train, y_train, X_test, y_test, test_feat)

result = pd.DataFrame({
    'prediction': test_preds * 1.1 # increasing each point by 10 percent
})
result.to_csv("output.csv", index=False)
