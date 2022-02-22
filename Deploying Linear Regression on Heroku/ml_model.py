#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd 
import numpy as np 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression


# In[13]:


def logistic_model():
    df = pd.read_csv("WineQT.csv")
    
    X = df.drop(["quality", "Id"], axis=1)
    y = df["quality"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)
    
    label_encoder = LabelEncoder()
    y_train =  label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)
    
    #logistic regression


    lr = LogisticRegression(solver='newton-cg',random_state=1,fit_intercept=False)
    model  = lr.fit(X_train,y_train)
    
    y_pred = model.predict(X_test)
    
    return y_pred


# In[14]:


logistic_model()


# In[9]:


df = pd.read_csv("WineQT.csv")
    
X = df.drop(["quality", "Id"], axis=1)
y = df["quality"]
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

label_encoder = LabelEncoder()
y_train =  label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)
    
    #logistic regression


lr = LogisticRegression(solver='newton-cg',random_state=1,fit_intercept=False)
model  = lr.fit(X_train,y_train)
    
y_pred = model.predict(X_test)


# In[15]:


import pickle

with open ('/Users/gunjantoora/Documents/gunjan-home/uic/spring22/MLops/project/ml models/ml_model.pkl', 'wb') as ml_model:
    pickle.dump(lr, ml_model)


# In[ ]:




