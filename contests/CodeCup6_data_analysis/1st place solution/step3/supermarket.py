# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 17:20:57 2021

@author: Amirali
"""
import pandas as pd
import statistics
from datetime import date
import calendar
import datetime
df = pd.read_csv('supermarket.csv')

###numberf of products
values  = df["Product"].value_counts()
n_pro = len(values)

###average sell
values  = df["Date"].value_counts()
n_date = len(values)
n_rows = df.shape[0]
mean_sell = round(n_rows/n_date,2)

###five products
values  = df["Product"].value_counts()
v = dict(values)
low_pro = sorted(v , key=v .get, reverse=False)[:4]

###customers
d_baskets = dict()
d_customers = dict()
for index, row in df.iterrows():
    d_baskets[row["Customer Id"]]=[]
    
for index, row in df.iterrows():
    b =  d_baskets[row["Customer Id"]]
    if row["Date"] not in b and "2020" in row["Date"]:
        b.append(row["Date"])
    d_baskets[row["Customer Id"]] = b
    
for index, row in df.iterrows():
    d_customers[row["Customer Id"]] = len(d_baskets[row["Customer Id"]])

    
high_cus = sorted(d_customers , key=d_customers .get, reverse=True)[:5]


###day of the week
Saturday = 0
Sunday = 0
Monday = 0
Tuesday = 0
Wednesday = 0
Thursday = 0
Friday = 0

date_list = list(df["Date"])

for d in date_list:
    year,month,day = (int(x) for x in d.split('-'))    
    ans = datetime.date(year, month, day)
    if ans.strftime("%A")=="Friday":
        Friday +=1
    if ans.strftime("%A")=="Thursday":
        Thursday +=1
    if ans.strftime("%A")=="Wednesday":
        Wednesday +=1
    if ans.strftime("%A")=="Tuesday":
        Tuesday +=1
    if ans.strftime("%A")=="Monday":
        Monday +=1
    if ans.strftime("%A")=="Sunday":
        Sunday +=1
    if ans.strftime("%A")=="Saturday":
        Saturday +=1


##association
d_baskets = dict()
for index, row in df.iterrows():
    d_baskets[row["Customer Id"]+row["Date"]]=[]
    
for index, row in df.iterrows():
    b = d_baskets[row["Customer Id"]+row["Date"]]
    b.append(row["Product"])
    d_baskets[row["Customer Id"]+row["Date"]] = b
    
values  = df["Product"].value_counts()
v = dict(values)
products = list(v.keys())

d_products = dict()
for p in products:
    d_products[p]=0

for p in products:
    for k in list(d_baskets.keys()):
        b = d_baskets[k]
        if p in b:
            m = d_products[p]
            d_products[p] =m+1

high_pro_b = sorted(d_products , key=d_products .get, reverse=True)[:5]

    

    
dict_values_day = {Friday:"Friday",Thursday:"Thursday",Wednesday:"Wednesday",Tuesday:"Tuesday",Monday:"Monday",Sunday:"Sunday",Saturday:"Saturday"}
high_day = dict_values_day.get(max(dict_values_day))
    
string  = str(n_pro)+"\n"+str(mean_sell)+"\n"+str(low_pro)[1:-1].replace("'","").replace(", ",",")+"\n"+str(high_cus)[1:-1].replace("'","").replace(", ",",")+"\n"+high_day+"\n"+str(high_pro_b)[1:-1].replace("'","").replace(", ",",")+"\n"+"-1"             

f = open("output.txt", "w")
f.write(string)
f.close()
    