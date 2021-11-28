import pandas as pd
import numpy as np

df = pd.read_csv('train.csv')
df.head()

df['ET'] = (df['Employment Type'] == 'Private Sector/Self Employed').astype(int)
df['GON'] = (df['GraduateOrNot'] == 'Yes').astype(int)
df['FF'] = (df['FrequentFlyer'] == 'Yes').astype(int)
df['ETA'] = (df['EverTravelledAbroad'] == 'Yes').astype(int)
df['TI'] = (df['TravelInsurance'] == 'Yes').astype(int)

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression

xdf = df[['Age', 'ET', 'GON', 'AnnualIncome', 'FamilyMembers', 'ChronicDiseases', 'FF', 'ETA']]
sc = MinMaxScaler()
nxdf = sc.fit_transform(xdf)
X_train, X_test, y_train, y_test = train_test_split(nxdf, df['TI'], test_size=0.3, random_state=0)

#mdl = AdaBoostClassifier(n_estimators=200, learning_rate=0.048).fit(X_train, y_train) # 7956
mdl = GradientBoostingClassifier(n_estimators=200, learning_rate=0.045, max_depth=3).fit(X_train, y_train) # 8111
#mdl = LogisticRegression(max_iter=400, C=3).fit(X_train, y_train) # 7608

roc_auc_score(y_test, mdl.predict_proba(X_test)[:,1])

mdl = GradientBoostingClassifier(n_estimators=200, learning_rate=0.045, max_depth=3).fit(nxdf, df['TI'])

dfp = pd.read_csv('test.csv')
dfp['ET'] = (dfp['Employment Type'] == 'Private Sector/Self Employed').astype(int)
dfp['GON'] = (dfp['GraduateOrNot'] == 'Yes').astype(int)
dfp['FF'] = (dfp['FrequentFlyer'] == 'Yes').astype(int)
dfp['ETA'] = (dfp['EverTravelledAbroad'] == 'Yes').astype(int)

xdfp = dfp[['Age', 'ET', 'GON', 'AnnualIncome', 'FamilyMembers', 'ChronicDiseases', 'FF', 'ETA']]
nxdfp = sc.transform(xdfp)

dfp['prediction'] = mdl.predict_proba(nxdfp)[:,1]

dfp[['Customer Id', 'prediction']].to_csv('output.csv', index=False)
