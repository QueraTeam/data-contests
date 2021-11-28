import pandas as pd
import numpy as np

df = pd.read_csv('supermarket.csv')
df.head()

#1
df.Product.unique().size

#2
df.groupby('Date').count().Product.mean()

#3
df.groupby('Product').count().sort_values('Date').head(4)

#4
df[df.apply(lambda x: x['Date'][:4] == '2020', axis=1)].groupby(['Customer Id', 'Date']).count().groupby('Customer Id').count().sort_values('Product').tail(5)

#5
df['date'] = pd.to_datetime(df['Date'])
df['dow'] = df['date'].dt.day_name()
df.groupby('dow').count()

#6
dff = df.groupby(['Customer Id', 'Date']).apply(lambda x: tuple(x.Product)).reset_index()

a = dff[0].values
b = []
for i in a:
    for j in i:
        b.append(j)
c = df.Product.unique()

d = []
for i in c:
    d.append(b.count(i))
    
print(c[np.argsort(d)[::-1][:5]])
