import numpy as np
import pandas as pd
import gc
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import optuna
from functools import partial

import xgboost as xgb
from xgboost import plot_importance, plot_tree


train = pd.read_csv("../input/code-cup-hotels/hotels/train.csv", skiprows=range(1, 29_000_000))
test = pd.read_csv("../input/code-cup-hotels/hotels/test.csv")


dates = ['search_date', 'checkIn_date', 'checkOut_date']


for date in dates:
    for df in [train, test]:
        date_format = "%Y-%m-%d %H:%M:%S" if date == 'search_date' else "%Y-%m-%d"
        df[date] = pd.to_datetime(df[date], format=date_format)



for df in [train, test]:
    df['res_days'] = (df['checkOut_date'] - df['checkIn_date']).dt.days
    df['in_advance'] = (df['checkIn_date'] - df['search_date']).dt.days


channel_encoder = LabelEncoder().fit(train['channel'])

for df in [train, test]:
    df['channel'] = channel_encoder.transform(df['channel'])


def create_features(df, date):
    """
    Creates time series features from datetime index
    """
    if date == 'search_date':
        df[f'{date}_hour'] = df[date].dt.hour
        
    df[f'{date}_dayofweek'] = df[date].dt.dayofweek
    df[f'{date}_month'] = df[date].dt.month
    df[f'{date}_dayofmonth'] = df[date].dt.day
    df[f'{date}_weekofyear'] = df[date].dt.weekofyear

for date in dates:
    for df in [train, test]:
        create_features(df, date)

train = train.drop(['user', *dates, 'destination', 'hotel_category'], axis=1)
test = test.drop(['user', *dates, 'destination', 'hotel_category'], axis=1)


kfold = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

for i, (_, val_idx) in enumerate(kfold.split(train, train['is_booking'])):
    train.loc[val_idx, 'fold'] = i


gc.collect()
valid = train[train['fold'] == 0]
train = train.drop(valid.index)
valid = valid.reset_index(drop=True)
train = train.reset_index(drop=True)
print(train.shape, valid.shape)
gc.collect()


y_train = train['is_booking']
X_train = train.drop(['is_booking', 'fold'], axis=1)

y_valid = valid['is_booking']
X_valid = valid.drop(['is_booking', 'fold'], axis=1)
gc.collect()



def run(trial, xtrain, ytrain, xvalid, yvalid):
    learning_rate = trial.suggest_float("learning_rate", 1e-2, 0.25, log=True)
    reg_lambda = trial.suggest_loguniform("reg_lambda", 1e-8, 100.0)
    reg_alpha = trial.suggest_loguniform("reg_alpha", 1e-8, 100.0)
    subsample = trial.suggest_float("subsample", 0.1, 1.0)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.1, 1.0)
    max_depth = trial.suggest_int("max_depth", 1, 7)
    
    model = xgb.XGBClassifier(
        random_state=42,
        tree_method="gpu_hist",
        gpu_id=1,
        n_estimators=5000,
        predictor="gpu_predictor",
        learning_rate=learning_rate,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        max_depth=max_depth,
    )
    model.fit(xtrain, ytrain, early_stopping_rounds=150, eval_set=[(xvalid, yvalid)], verbose=500)
    preds_valid = model.predict_proba(xvalid)[:, 1]
    roc_auc = roc_auc_score(yvalid, preds_valid)
    return roc_auc



opt_fun = partial(
    run,
    xtrain=X_train,
    ytrain=y_train,
    xvalid=X_valid,
    yvalid=y_valid
)

study = optuna.create_study(direction="maximize")
study.optimize(opt_fun, n_trials=80)
print(study.best_params)


def generate_predictions(params, xtrain, ytrain, xvalid, yvalid, xtest):    

    model = xgb.XGBClassifier(
        random_state=42,
        tree_method="gpu_hist",
        gpu_id=1,
        n_estimators=5000,
        predictor="gpu_predictor",
        **params,
    )
    model.fit(xtrain, ytrain, early_stopping_rounds=150, eval_set=[(xvalid, yvalid)], verbose=500)
    preds_valid = model.predict_proba(xvalid)[:, 1]
    roc_auc = roc_auc_score(yvalid, preds_valid)
    print(roc_auc)
    test_preds = model.predict_proba(xtest)
    return test_preds


test_preds = generate_predictions(study.best_params, X_train, y_train, X_valid, y_valid, test)


result = pd.DataFrame(
    {
        'prediction': test_preds[:, 1]
    }
)

result.to_csv("output.csv", index=False)

