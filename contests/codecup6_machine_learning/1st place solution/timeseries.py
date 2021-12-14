import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import plot_importance

train_data = pd.read_csv('train.csv', index_col=[0], parse_dates=[0])
test_data = pd.read_csv('test.csv', index_col=[0], parse_dates=[0])

def double_exponential_smoothing(series, alpha, beta):
    result = [series[0]]
    for n in range(1, len(series)):
        if n == 1:
            level, trend = series[0], series[1] - series[0]
        if n >= len(series):
            value = result[-1]
        else:
            value = series[n]
        last_level, level = level, alpha*value + (1-alpha)*(level+trend)
        trend = beta*(level-last_level) + (1-beta)*trend
        result.append(level+trend)
    return result

series = train_data.copy()
window = 20
series = series.rolling(window=window).mean().shift(-window//2)
series = series[window//2:-window//2].copy()
# rolling_mean = series.copy()
series["y"] = double_exponential_smoothing(series["y"], 0.8, 0.01)

window_size = 30
rolling_mean = series.rolling(window=window_size).mean().shift(-window_size//2)
rolling_mean = rolling_mean[window_size//2:-window_size//2]
# rolling_mean["y"] = double_exponential_smoothing(rolling_mean["y"], 0.01, 0.1)

series = series[window_size//2:-window_size//2]

def create_features(df, label=None):
    df['date'] = df.index
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute
    df['minuteofday'] = df['date'].dt.minute.add(df['hour'] * 60)
    df['dayofweek'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.weekofyear
    
    X = df[['hour', 'minute', 'minuteofday', 'dayofweek', 'month', 'dayofyear','dayofmonth','weekofyear']]
    if label:
        y = df[label]
        return X, y
    return X

X_train, y_train = create_features(series, label='y')
X_test = create_features(test_data)

reg = xgb.XGBRegressor(n_estimators=1000, max_depth=16)
reg.fit(X_train, y_train, eval_set=[(X_train, y_train)], early_stopping_rounds=50)

_ = plot_importance(reg, height=0.9)

test_data['prediction'] = reg.predict(X_test)
all_data = pd.concat([train_data, test_data], sort=False)

output = test_data['prediction']
output.to_csv("output.csv", index=False)
