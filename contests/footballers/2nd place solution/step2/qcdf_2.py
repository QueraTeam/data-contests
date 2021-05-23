# -*- coding: utf-8 -*-
"""QCDF-2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/199DwhkmWP1mci4edY2D1cfoBx3UKIxIp
"""

!unzip "drive/MyDrive/Captain Tsubasa.zip"

import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
test

X_train = []
y_train = []
for i, row in train.iterrows():
  X_train.append([row['minute'], row['second'], abs(row['x']), abs(row['y']),
                  list(set(train['playType'].tolist())).index(row['playType']),
                  list(set(train['bodyPart'].tolist())).index(row['bodyPart']),
                  row['interveningOpponents'],
                  row['interveningTeammates'],
                  list(set(train['interferenceOnShooter'].tolist())).index(row['interferenceOnShooter'])])
  if 'گُل' in row['outcome']:
    y_train.append(1)
  else:
    y_train.append(0)

X_test = []
for i, row in test.iterrows():
  X_test.append([row['minute'], row['second'], abs(row['x']), abs(row['y']),
                 list(set(train['playType'].tolist())).index(row['playType']),
                 list(set(train['bodyPart'].tolist())).index(row['bodyPart']),
                 row['interveningOpponents'],
                 row['interveningTeammates'],
                 list(set(train['interferenceOnShooter'].tolist())).index(row['interferenceOnShooter'])])

from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.models import Sequential
import numpy as np

model = Sequential()
model.add(BatchNormalization())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.1))
# model.add(Dense(16, activation='relu'))
# model.add(Dropout(0.1))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(np.array(X_train), np.array(y_train), batch_size=16, epochs=63)

res = model.predict(X_test)

df = pd.DataFrame(res, columns=['prediction'])
df.to_csv('output.csv', index=False)
df.head()

model.save('drive/MyDrive/chimigan-model-2.h5')