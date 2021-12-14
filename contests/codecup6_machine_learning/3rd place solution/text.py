#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


df_train = pd.read_csv("train.csv").astype(str)
df_test = pd.read_csv("test.csv").astype(str)


# In[3]:


from PersianStemmer import PersianStemmer 
ps = PersianStemmer()
train_stemmed = list(map(lambda x: ps.run(x), df_train["comment"]))
df_train["text"] = train_stemmed


# In[4]:


train_stemmed = list(map(lambda x: ps.run(x), df_test["comment"]))
df_test["text"] = train_stemmed


# In[5]:


df_train.head()


# In[6]:


Y_test = df_test.comment.values


# In[7]:


df_train[["text"]].sample(20)


# In[8]:


from snorkel.labeling import labeling_function

ABSTAIN = -1
HAM = 0
SPAM = 1

@labeling_function()
def nice_falvor(x):
    return HAM if "لذیذ" in x.text.lower() or  "لذت" in x.text.lower() else ABSTAIN

@labeling_function()
def bad_falvor(x):
    return SPAM if "شور" in x.text.lower() or  "چرب" in x.text.lower()  or  "سوخت" in x.text.lower() or  "پایین" in x.text.lower() else ABSTAIN

@labeling_function()
def good(x):
    return HAM if "عالی" in x.text.lower() or "بهتر" in x.text.lower() else ABSTAIN

@labeling_function()
def bad(x):
    return SPAM if "بد" in x.text.lower() or "افتضاخ" in x.text.lower() or "نپخته" in x.text.lower() or "هیچ" in x.text.lower()  or "!" in x.text.lower() or "?" in x.text.lower() or "حیف" in x.text.lower() or "خام" in x.text.lower()  or "انقظا" in x.text.lower() or "گران" in x.text.lower() else ABSTAIN

@labeling_function()
def normal(x):
    return SPAM if "معمولی" in x.text.lower() else ABSTAIN

@labeling_function()
def warm(x):
    return HAM if "گرم" in x.text.lower()  or "داغ" in x.text.lower() else ABSTAIN

@labeling_function()
def cold(x):
    return SPAM if "سرد" in x.text.lower() else ABSTAIN

@labeling_function()
def but(x):
    return SPAM if "ولی" in x.text.lower() or "عوض" in x.text.lower() or "اشتباه" in x.text.lower() or "جا" in x.text.lower() else ABSTAIN

@labeling_function()
def tnx(x):
    return HAM if "ممنون" in x.text.lower() else ABSTAIN

@labeling_function()
def bad_delivery(x):
    return SPAM if "دیر" in x.text.lower() or "طول" in x.text.lower() or "ساعت" in x.text.lower() else ABSTAIN

@labeling_function()
def volume(x):
    return SPAM if "کم" in x.text.lower() else ABSTAIN

@labeling_function()
def long(x):
    return SPAM if len(x.text.lower().split()) > 10 else ABSTAIN


# In[9]:


from snorkel.labeling import PandasLFApplier

lfs = [nice_falvor, bad_falvor, good, bad, normal, warm, cold, but, tnx, bad_delivery, volume, long]

applier = PandasLFApplier(lfs=lfs)
L_train = applier.apply(df=df_train)
L_test = applier.apply(df=df_test)


# In[10]:


L_train


# In[11]:


stats = (L_train != ABSTAIN).mean(axis=0)
for i, j in zip(lfs, stats):
    print(i, j)


# In[12]:


from snorkel.labeling import LFAnalysis

LFAnalysis(L=L_train, lfs=lfs).lf_summary()


# In[13]:


from snorkel.labeling.model import MajorityLabelVoter

majority_model = MajorityLabelVoter()
preds_train = majority_model.predict(L=L_train)


# In[14]:


preds_train


# In[15]:


from snorkel.labeling.model import LabelModel

label_model = LabelModel(cardinality=2, verbose=True)
label_model.fit(L_train=L_train, n_epochs=500, log_freq=100, seed=123)


# In[16]:


from snorkel.labeling import filter_unlabeled_dataframe

probs_train = label_model.predict_proba(L=L_train)
df_train_filtered, probs_train_filtered = filter_unlabeled_dataframe(
    X=df_train, y=probs_train, L=L_train
)


# In[17]:


from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(ngram_range=(1, 5))
X_train = vectorizer.fit_transform(df_train_filtered.text.tolist())
X_test = vectorizer.transform(df_test.text.tolist())


# In[18]:


from snorkel.utils import probs_to_preds

preds_train_filtered = probs_to_preds(probs=probs_train_filtered)


# In[19]:


from sklearn.linear_model import LogisticRegression

sklearn_model = LogisticRegression(C=1e3)
sklearn_model.fit(X=X_train, y=preds_train_filtered)


# from sklearn.ensemble import RandomForestClassifier
# sklearn_model = RandomForestClassifier(max_depth=12, random_state=0, n_estimators=200)
# sklearn_model.fit(X=X_train, y=preds_train_filtered)


# In[20]:


# import xgboost as xgb
# reg = xgb.XGBRFClassifier()
# reg.fit(X_train, preds_train_filtered, eval_set=[(X_train, preds_train_filtered)])


# In[22]:


preds = sklearn_model.predict_proba(X_test)[:, 0]


# In[23]:


pd.DataFrame({
    "prediction": map(lambda x: "{:.4f}".format(x), preds)
}).to_csv("output.csv", index=False)


# In[ ]:





# In[ ]:




