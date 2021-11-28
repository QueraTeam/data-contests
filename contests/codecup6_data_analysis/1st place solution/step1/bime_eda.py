# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 16:02:17 2021

@author: Amirali
"""

import pandas as pd
df = pd.read_csv('train.csv')


###shape
n_rows = df.shape[0]
n_col = df.shape[1]


###average incomve
avg_income  = int(df["AnnualIncome"].mean()//1)



###travel abroad
n_abroad  = df["EverTravelledAbroad"].value_counts()["Yes"]



###employment
n_gov = df["Employment Type"].value_counts()["Government Sector"]
n_pv =  df["Employment Type"].value_counts()["Private Sector/Self Employed"]

if n_gov > n_pv:
    share = n_gov/(n_gov+n_pv)*100
    high_emp = "Government Sector "+str(round(share,2))
else:
    share = n_pv/(n_gov+n_pv)*100
    high_emp = "Private Sector/Self Employed "+str(round(share,2))


####diseases
n_dis = ((df['ChronicDiseases'] == 1) ).sum()
inc_dis =  ((df['ChronicDiseases'] == 1) & 
             (df['TravelInsurance'] == 'Yes')).sum()/n_dis*100
inc_dis = round(inc_dis,2)


string  = str(n_rows)+" "+str(n_col)+"\n"+str(avg_income)+"\n"+str(n_abroad)+"\n"+str(high_emp)+"\n"+str(inc_dis)

f = open("output.txt", "w")
f.write(string)
f.close()