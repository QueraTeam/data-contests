# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 16:38:01 2021

@author: Amirali
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.utils import class_weight

df = pd.read_csv('train.csv')
###number of labels 
values  = df["TravelInsurance"].value_counts()
n_pos = values["Yes"]
n_neg = values["No"]

###pre proccessing
df_enc = pd.get_dummies(df.iloc[:,1 :9])


###datasets
y = df["TravelInsurance"]
X = df_enc

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.001, random_state=1225)

###models
model = RandomForestClassifier(class_weight="balanced",max_features = "log2",max_depth = 5,n_estimators = 300)
model.fit(X_train, y_train)
#y_proba = model.predict_proba(X_test)
#print(roc_auc_score(y_test, y_proba[:, 1]))



###prediction
df = pd.read_csv('test.csv')
ids = df["Customer Id"]
X = pd.get_dummies(df.iloc[:,1 :9])
pred = model.predict_proba(X)
new_df = df.drop(df.columns[1:9], axis=1)
new_df["prediction"]= pred[:, 1]
new_df.set_index('Customer Id', inplace=True)
new_df.to_csv("output.csv", sep=',')