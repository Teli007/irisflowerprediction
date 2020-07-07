#!/usr/bin/env python
# coding: utf-8

# In[105]:


import pandas as pd
import flask
import pickle

A=pd.read_csv("iris.csv")
A=A.drop(columns=["Unnamed: 0"])
x=A.drop(columns=["Species"])
y=A[["Species"]]

from sklearn.neighbors import KNeighborsClassifier
knc=KNeighborsClassifier(n_neighbors=9)

classifier=knc.fit(x,y)

filename = 'iris.pkl'
pickle.dump(classifier, open(filename, 'wb'))


# In[ ]:




