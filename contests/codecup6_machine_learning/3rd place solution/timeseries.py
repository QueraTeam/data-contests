#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


# In[2]:


df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")


# In[3]:


df_train.head()


# In[4]:


print(len(df_train))


# In[5]:


df_train['time'] =  pd.to_datetime(df_train['time'])


# In[6]:


plt.figure(figsize=(16,9))
plt.plot(df_train['time'], df_train['y'])
plt.show()


# In[7]:


K = 3000
plt.figure(figsize=(16,9))
plt.plot(df_train['time'][:K], df_train['y'][:K])
plt.show()


# In[8]:


result = savgol_filter(df_train['y'], 203, 4)
plt.figure(figsize=(16,9))
plt.plot(df_train['time'][:K], result[:K])
plt.show()


# In[9]:


df_train['new_y'] = result
print(len(result))


# In[10]:


plt.figure(figsize=(16,9))
plt.plot(df_train['time'], result)
plt.show()


# In[11]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

plt.figure(figsize=(16,9))
plot_acf(df_train.new_y.squeeze(), lags=2000)
plt.show()


# In[12]:


plt.figure(figsize=(16,9))
plt.plot(df_train.new_y.diff()); 
plt.show()


# In[13]:


plt.figure(figsize=(16,9))
plot_acf(df_train.new_y.diff().dropna().squeeze()[800:], lags=3000)
plt.show()


# In[14]:


plt.figure(figsize=(16,9))
plot_acf(df_train.new_y.diff().dropna().squeeze(), lags=40, method="ols")
plt.show()


# In[ ]:


plt.figure(figsize=(16,9))
plt.plot(df_train.new_y.diff().diff()); 
plt.show()


# In[ ]:


plt.figure(figsize=(16,9))
plot_acf(df_train.new_y.diff().diff().dropna().squeeze(), lags=2000)
plt.show()


# In[ ]:


from pmdarima.arima.utils import ndiffs
y = df_train.new_y


print(ndiffs(y, test='adf'))
print(ndiffs(y, test='kpss'))
print(ndiffs(y, test='pp'))


# In[ ]:


plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})

plot_pacf(df_train.new_y.diff().dropna(), method="ywm", lags=10)

plt.show()


# In[ ]:


plt.rcParams.update({'figure.figsize':(9,3), 'figure.dpi':120})

plot_pacf(df_train.new_y.diff().diff().dropna())

plt.show()


# In[ ]:


plt.plot(df_train.new_y.diff().diff(24 * 60), label='Seasonal Differencing', color='green')
plt.show()


# In[ ]:


plt.plot(df_train.new_y.diff().diff(24 * 60 * 30), label='Seasonal Differencing', color='green')
plt.show()


# In[ ]:


fig, axes = plt.subplots(1, 1, figsize=(10,5), dpi=100, sharex=True)

# Usual Differencing
axes.plot(df_train.new_y[:].diff())
axes.plot(df_train.new_y[:].diff(24 * 60 * 30))
axes.set_title('Usual Differencing')
axes.legend(loc='upper left', fontsize=10)

plt.show()


# In[ ]:


import pmdarima as pm
import statsmodels.api as sm
data = pd.read_csv("train.csv", parse_dates=['time'], index_col='time')
data['y'] = df_train['new_y'].tolist()
model=sm.tsa.statespace.SARIMAX(data[:],order=(1, 1, 1),seasonal_order=(1,1,1,60 * 24))
results=model.fit()


# In[ ]:


from prophet import Prophet
data = pd.read_csv("train.csv", parse_dates=['time'])
data['y'] = df_train['new_y'].tolist()
data = data.rename(index={"time": "DS"})


# In[ ]:




