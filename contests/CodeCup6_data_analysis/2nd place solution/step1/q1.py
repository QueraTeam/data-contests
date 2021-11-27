import pandas as pd
import numpy as np

df = pd.read_csv('train.csv')
df.head()

print(df.shape)
print(df.AnnualIncome.mean())
print(df[df.EverTravelledAbroad == 'Yes'].shape[0])
print(df[df['Employment Type'] == 'Private Sector/Self Employed'].shape[0])
print(df[df['Employment Type'] == 'Government Sector'].shape[0])
print(1155*100/1590)
print(100 * df[(df['ChronicDiseases'] == 1) & (df['TravelInsurance'] == 'Yes')].shape[0] / df[df['ChronicDiseases'] == 1].shape[0])
