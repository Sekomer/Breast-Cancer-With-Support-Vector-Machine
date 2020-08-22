#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


raw = load_breast_cancer()
# dataset


# In[3]:


raw.keys()


# In[4]:


df = pd.DataFrame(np.c_[raw['data'], raw['target']], columns=np.append(raw['feature_names'], ['target']))
# creating a dataset from features and labels


# In[5]:


df.tail()


# In[6]:


sb.pairplot(df, hue='target', vars = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness'])


# In[7]:


plt.figure(figsize=(17,17))
sb.heatmap(df.corr(), annot=True)


# In[8]:


model = SVC()


# In[9]:


X = df.drop(['target'], axis = 1)
# dropping labels from futures


# In[10]:


y = df['target']


# In[11]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state= 42)
# splitting train and test data


# In[12]:


sc = MinMaxScaler() # minmax performs better than >> StandardScalar(copy=True, with_mean=True, with_std=True)
sc.fit(X_train)
X_train_scaled= sc.transform(X_train)
X_test_scaled= sc.transform(X_test)


# In[13]:


model.fit(X_train_scaled, y_train)


# In[14]:


y_pred = model.predict(X_test_scaled)
cm = confusion_matrix(y_pred, y_test)


# In[15]:


sb.heatmap(cm, annot=True)
# results without hypertuning


# In[16]:


cm


# In[17]:


params = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}  
grid = GSCV(SVC(), params, refit=True, verbose=3)
# grid searching for finding best parameters


# In[18]:


get_ipython().run_cell_magic('capture', ' ', "## i deleted the output cuz it's too long ##\ngrid.fit(X_train_scaled, y_train);")


# In[19]:


grid.best_params_


# In[20]:


y_pred = grid.predict(X_test_scaled)


# In[21]:


cm = confusion_matrix(y_pred, y_test)


# In[22]:


sb.heatmap(cm, annot=True, fmt="d")
# results with tuning


# In[23]:


cm


# In[24]:


classification_report(y_test, y_pred)

