import numpy as np
import pandas as pd
from datetime import date


df = pd.read_csv('supermarket.csv')

print(df.Product.drop_duplicates().size)

avg_sale = df.groupby('Date').Product.count().mean()
print(f'{avg_sale :.2f}')

print(','.join(df.Product.value_counts()[-4:].index.to_list()))

print(','.join(df[df.Date.str.startswith('2020')][['Customer Id', 'Date']].drop_duplicates()['Customer Id'].value_counts()[:5].index.to_list()))


weekday_ix = pd.to_datetime(df.Date).dt.weekday.value_counts().index[0]
weekdays = 'Monday Tuesday Wednesday Thursday Friday Saturday Sunday'.split()
print(weekdays[weekday_ix])




print(','.join(df.Product.value_counts()[:5].index.to_list()))
# print (/-1)

# from mlxtend.frequent_patterns import apriori
# from mlxtend.frequent_patterns import association_rules

wide = df.assign(cnt=1).pivot(index=['Customer Id', 'Date'], columns='Product', values='cnt').fillna(0).reset_index(drop=True).astype(int)


# frequent_itemsets = apriori(wide, min_support=0.001, use_colnames=True)


# rules = association_rules(frequent_itemsets, metric="confidence")
# rules.head()



from apriori_python import apriori

wide
itemSetList = wide.apply(lambda r: (np.where(r.values==1)[0].tolist()), axis=1).to_list()


freqItemSet, rules = apriori(itemSetList, minSup=0.01, minConf=0,)

rules.sort(key=lambda x: x[2], reverse=True)

def format_rule(r, l):
    l = [f'"{wide.columns.to_list()[x]}"' for x in l]
    r = [f'"{wide.columns.to_list()[x]}"'  for x in r]
    return (f"({','.join(l)})->({','.join(r)})")
    

print('|'.join(format_rule(r, l) for l, r, _ in rules[:2]))
    



