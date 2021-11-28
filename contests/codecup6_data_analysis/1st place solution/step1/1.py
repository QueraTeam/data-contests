import pandas as pd


df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

print(len(df_train), len(df_train.columns))

print(df_train['AnnualIncome'].mean())

print(len(df_train[df_train['EverTravelledAbroad'] == 'Yes']))

print(df_train.groupby(['Employment Type']).agg({'AnnualIncome': 'count'}))


a = len(df_train[(df_train['ChronicDiseases'] == 1) & (df_train['TravelInsurance'] == "Yes")]) 
b = len(df_train[df_train['ChronicDiseases'] == 1])
print(a, a / b)
