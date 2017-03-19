# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 15:14:12 2017

@author: czy
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import matplotlib.pyplot as plt

df=pd.read_csv('C:\\Users\\czy\\Documents\\Python Scripts\\winequality-red.csv',sep=';')

x=df.values[:,0:11]
y=df.values[:,11]

lr = LogisticRegression()
lr.fit(x, y)
predicted=lr.predict(x)
RSS_lr=sum((y-predicted)*(y-predicted))
plt.plot(y-predicted)

_svm=svm.SVC()
_svm.fit(x,y)
predicted=_svm.predict(x)
RSS_svm=sum((y-predicted)*(y-predicted))
plt.plot(y-predicted)