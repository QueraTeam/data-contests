#!/usr/bin/env python
# coding: utf-8

# In[2]:


# !wget http://156.253.5.172/hotels.zip
# !unzip -q hotels.zip


# In[1]:


import pandas as pd
import numpy as np


# In[2]:


# pd.read_csv("train.csv").sample(frac=1).to_csv("train_random.csv", index=False)


# In[3]:


# df_train = pd.read_csv("hotels/train.csv")
df_test = pd.read_csv("test.csv")


# In[4]:


df_test.head()


# In[290]:


df_test["user"].unique().shape


# In[367]:


features_columns = ["channel", "is_mobile", "is_package", "n_rooms", "n_adults", "n_children", "hotel_category", "search_date", "checkIn_date", "checkOut_date"]


# In[427]:


channels = df_test["channel"].unique()
# destinations = df_test["destination"].unique()
cats = df_test["hotel_category"].unique()
# diffs = ["diff_{}".format(i) for i in range(10)]
data_columns = list(channels) + list(cats) + ["is_mobile", "is_package", "n_adults", "n_children", "n_rooms", "diff", "diff2", "weekday1", "weekday2", "weekday3", "month1", "month2", "month3", "quartet1", "quartet2", "quartet3", "single", "has_child", "rich", "common_hotels"]
data_columns_set = set(data_columns)


# In[426]:


def add_new_columns(_X):
    checkIn_date = pd.to_datetime(_X["checkIn_date"])
    checkOut_date = pd.to_datetime(_X["checkOut_date"])
    search_date = pd.to_datetime(_X["search_date"])
    _X["common_hotels"] = _X['hotel_category'].apply(lambda x: x in ["g7", "g13", "g15", "g32", "g19", "g43", "g6", "g49", "g16"])
    _X["diff"] = (checkIn_date - search_date).dt.days.fillna(0).astype(int)
    _X["diff2"] = (checkOut_date - checkIn_date).dt.days.fillna(0).astype(int)
    _X["weekday1"] = checkIn_date.dt.weekday.fillna(0)
    _X["weekday2"] = checkOut_date.dt.weekday.fillna(0)
    _X["weekday3"] = search_date.dt.weekday.fillna(0)
    _X["month1"] = checkIn_date.dt.month.fillna(0)
    _X["month2"] = checkOut_date.dt.month.fillna(0)
    _X["month3"] = search_date.dt.month.fillna(0)
    _X["quartet1"] = checkIn_date.dt.quarter.fillna(0)
    _X["quartet2"] = checkOut_date.dt.quarter.fillna(0)
    _X["quartet3"] = search_date.dt.quarter.fillna(0)
    _X["single"] = _X['n_adults'].apply(lambda x: x == 0)
    _X["has_child"] = _X['n_children'].apply(lambda x: x > 0)
    _X["rich"] = _X['n_rooms'].apply(lambda x: x > 1)
    del _X["checkIn_date"]
    del _X["checkOut_date"]
    del _X["search_date"]


# In[399]:


from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.neural_network import MLPClassifier
import xgboost as xgb


# model = BernoulliNB()
model = MultinomialNB()
# model = RandomForestClassifier()
# model = SGDClassifier(loss="modified_huber")
# model = DecisionTreeClassifier()
# model = LatentDirichletAllocation(n_components=2)
# model = MLPClassifier(verbose=True, max_iter=5)
# model = PassiveAggressiveClassifier()


# In[428]:


i = 0
for train_test_df in pd.read_csv("train_random.csv", chunksize=1024 * 8, iterator=True):
    if i <= 80:
        i += 1
        continue
    X_test = train_test_df[features_columns]
    add_new_columns(X_test)
    y_test = train_test_df["is_booking"]

    X_test = pd.get_dummies(X_test, columns=["channel", "hotel_category"])
    columns_set = set(X_test.columns)
    to_be_added = list(data_columns_set - columns_set)
    X_test = pd.concat(
    [
        X_test,
        pd.DataFrame(
            [[0 for i in range(len(to_be_added))]], 
            index=X_test.index,
            columns=to_be_added
        )
    ], axis=1
)
    xg_test = xgb.DMatrix(X_test[data_columns], label=y_test)
    break


# In[ ]:





# In[434]:


from imblearn.datasets import make_imbalance
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

single = True
xg = True
for epoch in range(1 if single else 5):
    i = 0
    for train_df in pd.read_csv("train_random.csv", chunksize=1024 * 64, iterator=True):
    #     print(len(train_df[train_df["is_booking"] == True]) / len(train_df))
    #     break
        if i <= -1:
            i += 1
            continue
        X = train_df[features_columns]
        add_new_columns(X)
        y = train_df["is_booking"]
        X = pd.get_dummies(X, columns=["channel", "hotel_category"])
        columns_set = set(X.columns)
        to_be_added = list(data_columns_set - columns_set)
        print("enter", i)

        X = pd.concat(
    [
        X,
        pd.DataFrame(
            [[0 for i in range(len(to_be_added))]], 
            index=X.index,
            columns=to_be_added
        )
    ], axis=1
)
        print("enter_train", i)

#         X, y = SMOTE().fit_resample(X, y)

        if not xg:
            if single:
                model.fit(X[data_columns], y)
                break
            if i == 0:
                try:
                    model.partial_fit(X[data_columns], y, classes=np.unique(y))
                except:
                    model.partial_fit(X[data_columns], y)
            else:
                model.partial_fit(X[data_columns], y)
        else:
            params = {
              'colsample_bynode': 0.8,
              'learning_rate': 1,
              'max_depth': 5,
              'num_parallel_tree': 100,
              'objective': 'binary:logistic',
              'subsample': 0.8,
              'tree_method': 'gpu_hist'
            }
            
            if i == 0:
                model = xgb.train(params, xgb.DMatrix(X[data_columns], label=y))
                model.save_model('model.model')
            else:
                model = xgb.train(params, xgb.DMatrix(X[data_columns], label=y), xgb_model='model.model')
                model.save_model('model.model')
                
#             _predict = model.predict(xg_test)
#             print(mse(_predict, y_test))
        if i == 5:
            # print(model.score(X_test[data_columns], y_test))
            break
        i += 1


# In[430]:


from sklearn.metrics import roc_auc_score, mean_squared_error as mse
_predict = model.predict(xg_test)
print(mse(_predict, y_test))
print(roc_auc_score(y_test, _predict))
# print(X['n_rooms'].unique())
# print(X['rich'].unique())
# print(X['n_rooms'].apply(lambda x: x > 1).unique())
# print(X['diff'].unique())


# In[431]:


ax = xgb.plot_importance(model, max_num_features=50)
fig = ax.figure
fig.set_size_inches(16, 9)


# In[414]:


# i = 0
# for train_test_df in pd.read_csv("train_random.csv", chunksize=1024 * 32, iterator=True):
#     if i <= 30:
#         i += 1
#         continue
#     y_test = train_test_df["is_booking"]
#     X_test = train_test_df[features_columns]
#     # X_test["diff"] = (pd.to_datetime(train_test_df["checkIn_date"]) - pd.to_datetime(train_test_df["search_date"])).dt.days
#     X_test = pd.get_dummies(X_test, columns=["channel", "destination", "hotel_category"])
#     columns_set = set(X_test.columns)
#     to_be_added = list(data_columns_set - columns_set)
#     X_test = pd.concat(
#     [
#         X_test,
#         pd.DataFrame(
#             [[0 for i in range(len(to_be_added))]], 
#             index=X_test.index,
#             columns=to_be_added
#         )
#     ], axis=1)

#     break
# model.score(X_test[data_columns], y_test)


# In[432]:


res = []
i = 0
for test_df in pd.read_csv("test.csv", chunksize=1024 * 32, iterator=True):
    X = test_df[features_columns]
    add_new_columns(X)
    X = pd.get_dummies(X, columns=["channel", "hotel_category"])
    columns_set = set(X.columns)
    to_be_added = data_columns_set - columns_set
    X = pd.concat(
    [
        X,
        pd.DataFrame(
            [[0 for i in range(len(to_be_added))]], 
            index=X.index,
            columns=to_be_added
        )
    ], axis=1)

    xg_res = xgb.DMatrix(X[data_columns])
    res = res + list(model.predict(xg_res))

#     if hasattr(model, 'predict_proba'):
#         res = res + list(model.predict_proba(X[data_columns])[:, 1])
#     else:
#         res = res + list(model.transform(X[data_columns])[:, 1])
    print(i)
    i += 1


# In[433]:


pd.DataFrame({"prediction": res}).to_csv("output.csv", index=False)


# In[ ]:





# In[ ]:




