import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier


fields = ['search_date', 'channel', 'is_mobile', 'is_package', 'destination', 'checkIn_date', 'checkOut_date', 'n_adults', 'n_children', 'n_rooms', 'hotel_category']
train_data = pd.read_csv("train.csv", usecols=fields + ["is_booking"])

# Balance classes

train_data_0 = train_data.loc[train_data["is_booking"] == False]
train_data_1 = train_data.loc[train_data["is_booking"] == True]

train_data_0 = train_data_0.sample(frac=1, random_state=63).reset_index(drop=True)
train_data_0 = train_data_0[:len(train_data_1)]

train_data = pd.concat([train_data_0, train_data_1])

train_data = train_data.sample(frac=1, random_state=63).reset_index(drop=True)

# Train model

y = train_data["is_booking"].apply(lambda x: 1 if x else 0)
X = train_data.drop(columns="is_booking")

# Extract features

def preprocessing(data):    
    data['search_date'] = pd.to_datetime(data['search_date'])
    data['ts'] = data['search_date'].astype('int64') // 10**9
    
    data['month'] = data['search_date'].dt.month
#     data['month_sin'] = np.sin(data['month'] * (2 * np.pi / 12))
#     data['month_cos'] = np.cos(data['month'] * (2 * np.pi / 12))
    
#     data['day'] = data['search_date'].dt.day
#     data['day_sin'] = np.sin(data['day'] * (2 * np.pi / 31))
#     data['day_cos'] = np.cos(data['day'] * (2 * np.pi / 31))
    
    data['day_of_week'] = data['search_date'].dt.weekday
#     data['day_of_week_sin'] = np.sin(data['day_of_week'] * (2 * np.pi / 7))
#     data['day_of_week_cos'] = np.cos(data['day_of_week'] * (2 * np.pi / 7))

    data['hour'] = data['search_date'].dt.hour
    
    data[['checkIn_date', 'checkOut_date']] = data[['checkIn_date', 'checkOut_date']].apply(pd.to_datetime)
    data['duration'] = (data['checkOut_date'] - data['checkIn_date']).dt.days
    data['distance'] = (data['checkIn_date'] - data['search_date']).dt.days
    data['checkInMonth'] = data['checkIn_date'].dt.month
    data['checkInWeekday'] = data['checkIn_date'].dt.weekday
    
    data = data.drop(columns=['search_date', 'checkIn_date', 'checkOut_date'])
    
    for col in ['channel', 'is_mobile', 'is_package', 'destination', 'hotel_category', 'month', 'day_of_week', 'hour', 'checkInMonth', 'checkInWeekday']:
        data[col] = data[col].astype('category')
    
    return data

X = preprocessing(X)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=63)

params = {'loss_function': "Logloss",
          'eval_metric': "AUC",
          'cat_features': ["channel", "is_mobile", "is_package", "hotel_category", "destination", "month", "checkInMonth", "day_of_week", "hour", "checkInWeekday"],
          'ignored_features': [],
          # 'task_type': 'GPU',
          # 'border_count': 32,
          # 'depth': 12,
          'iterations': 20000,
          'verbose': 500,
          'early_stopping_rounds': 500,
          'random_seed': 63
         }

model = CatBoostClassifier(**params)
model.fit(X_train, y_train, eval_set=(X_valid, y_valid), use_best_model=True)

model.get_feature_importance(prettified=True)

# Prediction

test_data = pd.read_csv("test.csv", usecols=fields)
X_test = preprocessing(test_data)

y_prob = model.predict_proba(X_test)
y_prob = pd.DataFrame(y_prob[:, 1], columns=["prediction"])
y_prob.to_csv('output.csv', index=False)
