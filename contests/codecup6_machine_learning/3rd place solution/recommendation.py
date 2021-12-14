#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install scikit-surprise')
import pandas as pd
from surprise import SVD
from surprise import Dataset
from surprise.model_selection import cross_validate


# In[ ]:


df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("train.csv")


# In[ ]:


from surprise.model_selection import train_test_split


trainset, testset = train_test_split(data, test_size=.25)


algo = SVD()


algo.fit(trainset)
predictions = algo.test(testset)

# Then compute RMSE
accuracy.rmse(predictions)

