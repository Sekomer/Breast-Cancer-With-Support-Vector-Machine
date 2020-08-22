#!/usr/bin/env python
# coding: utf-8

import numpy as np 
import pandas as pd 
import seaborn as sb 
import matplotlib.pyplot as plt 

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV as GSCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC

raw = load_breast_cancer()
# dataset

df = pd.DataFrame(np.c_[raw['data'], raw['target']], columns=np.append(raw['feature_names'], ['target']))
# creating a dataset from features and labels

model = SVC()

X = df.drop(['target'], axis = 1)
# dropping labels from futures

y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state= 42)
# splitting train and test data

sc = MinMaxScaler() # minmax performs better than >> StandardScalar(copy=True, with_mean=True, with_std=True)
sc.fit(X_train)
X_train_scaled= sc.transform(X_train)
X_test_scaled= sc.transform(X_test)

params = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}  
grid = GSCV(SVC(), params, refit=True, verbose=3)
# grid searching for finding best parameters

y_pred = grid.predict(X_test_scaled)

cm = confusion_matrix(y_pred, y_test)

sb.heatmap(cm, annot=True, fmt="d")
# results with tuning

classification_report(y_test, y_pred)
