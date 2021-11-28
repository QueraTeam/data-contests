import numpy as np
import pandas as pd


df = pd.read_csv('travel_insurance/train.csv')

print(f'{df.shape[0]} {df.shape[1]}')

print(int(df.AnnualIncome.mean()))

print((df.EverTravelledAbroad=='Yes').sum())

vc = df['Employment Type'].value_counts()
print(f'{vc.index[0]} {100 * vc[0] / vc.sum() :.2f}')

print(f"{(df[df.ChronicDiseases==1].TravelInsurance=='Yes').mean() * 100 :.2f}")
