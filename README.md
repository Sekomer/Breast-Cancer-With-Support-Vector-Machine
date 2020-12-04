# My SVM implementation for the Breast Cancer dataset for educational purposes

```python
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
```


```python
raw = load_breast_cancer()
# dataset
```


```python
raw.keys()
```




    >>>dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])




```python
df = pd.DataFrame(np.c_[raw['data'], raw['target']], columns=np.append(raw['feature_names'], ['target']))
# creating a dataset from features and labels
```


```python
df.tail()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean radius</th>
      <th>mean texture</th>
      <th>mean perimeter</th>
      <th>mean area</th>
      <th>mean smoothness</th>
      <th>mean compactness</th>
      <th>mean concavity</th>
      <th>mean concave points</th>
      <th>mean symmetry</th>
      <th>mean fractal dimension</th>
      <th>...</th>
      <th>worst texture</th>
      <th>worst perimeter</th>
      <th>worst area</th>
      <th>worst smoothness</th>
      <th>worst compactness</th>
      <th>worst concavity</th>
      <th>worst concave points</th>
      <th>worst symmetry</th>
      <th>worst fractal dimension</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>564</th>
      <td>21.56</td>
      <td>22.39</td>
      <td>142.00</td>
      <td>1479.0</td>
      <td>0.11100</td>
      <td>0.11590</td>
      <td>0.24390</td>
      <td>0.13890</td>
      <td>0.1726</td>
      <td>0.05623</td>
      <td>...</td>
      <td>26.40</td>
      <td>166.10</td>
      <td>2027.0</td>
      <td>0.14100</td>
      <td>0.21130</td>
      <td>0.4107</td>
      <td>0.2216</td>
      <td>0.2060</td>
      <td>0.07115</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>565</th>
      <td>20.13</td>
      <td>28.25</td>
      <td>131.20</td>
      <td>1261.0</td>
      <td>0.09780</td>
      <td>0.10340</td>
      <td>0.14400</td>
      <td>0.09791</td>
      <td>0.1752</td>
      <td>0.05533</td>
      <td>...</td>
      <td>38.25</td>
      <td>155.00</td>
      <td>1731.0</td>
      <td>0.11660</td>
      <td>0.19220</td>
      <td>0.3215</td>
      <td>0.1628</td>
      <td>0.2572</td>
      <td>0.06637</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>566</th>
      <td>16.60</td>
      <td>28.08</td>
      <td>108.30</td>
      <td>858.1</td>
      <td>0.08455</td>
      <td>0.10230</td>
      <td>0.09251</td>
      <td>0.05302</td>
      <td>0.1590</td>
      <td>0.05648</td>
      <td>...</td>
      <td>34.12</td>
      <td>126.70</td>
      <td>1124.0</td>
      <td>0.11390</td>
      <td>0.30940</td>
      <td>0.3403</td>
      <td>0.1418</td>
      <td>0.2218</td>
      <td>0.07820</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>567</th>
      <td>20.60</td>
      <td>29.33</td>
      <td>140.10</td>
      <td>1265.0</td>
      <td>0.11780</td>
      <td>0.27700</td>
      <td>0.35140</td>
      <td>0.15200</td>
      <td>0.2397</td>
      <td>0.07016</td>
      <td>...</td>
      <td>39.42</td>
      <td>184.60</td>
      <td>1821.0</td>
      <td>0.16500</td>
      <td>0.86810</td>
      <td>0.9387</td>
      <td>0.2650</td>
      <td>0.4087</td>
      <td>0.12400</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>568</th>
      <td>7.76</td>
      <td>24.54</td>
      <td>47.92</td>
      <td>181.0</td>
      <td>0.05263</td>
      <td>0.04362</td>
      <td>0.00000</td>
      <td>0.00000</td>
      <td>0.1587</td>
      <td>0.05884</td>
      <td>...</td>
      <td>30.37</td>
      <td>59.16</td>
      <td>268.6</td>
      <td>0.08996</td>
      <td>0.06444</td>
      <td>0.0000</td>
      <td>0.0000</td>
      <td>0.2871</td>
      <td>0.07039</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>




```python
sb.pairplot(df, hue='target', vars = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness'])
```




    <seaborn.axisgrid.PairGrid at 0x1eba5bb1040>




![png](https://github.com/Sekomer/Breast-Cancer-With-Support-Vector-Machine/blob/master/photos/output_5_1.png)



```python
plt.figure(figsize=(17,17))
sb.heatmap(df.corr(), annot=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1eba8074580>




![png](https://github.com/Sekomer/Breast-Cancer-With-Support-Vector-Machine/blob/master/photos/output_6_1.png)



```python
model = SVC()
```


```python
X = df.drop(['target'], axis = 1)
# dropping labels from futures
```


```python
y = df['target']
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state= 42)
# splitting train and test data
```


```python

```


```python
model.fit(X_train, y_train)
```




    >>>SVC()




```python
y_pred = model.predict(X_test)
cm = confusion_matrix(y_pred, y_test)
```


```python
sb.heatmap(cm, annot=True)
# results without hypertuning
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1eba993a790>




![png](https://github.com/Sekomer/Breast-Cancer-With-Support-Vector-Machine/blob/master/photos/output_14_1.png)



```python
cm
```




    >>>array([[ 52,   0],
           [ 11, 108]], dtype=int64)




```python
# difference of scaling and grid search #
```


```python
sc = MinMaxScaler() # minmax performs better than >> StandardScalar(copy=True, with_mean=True, with_std=True)
sc.fit(X_train)
X_train_scaled= sc.transform(X_train)
X_test_scaled= sc.transform(X_test)
```


```python
params = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
              'kernel': ['rbf']}  
grid = GSCV(SVC(), params, refit=True, verbose=3)
# grid searching for finding best parameters
```


```python
%%capture  
## i deleted the output cuz it's too long ##
grid.fit(X_train_scaled, y_train);
```


```python
grid.best_params_
```




    >>>{'C': 10, 'gamma': 0.1, 'kernel': 'rbf'}




```python
y_pred = grid.predict(X_test_scaled)
```


```python
cm = confusion_matrix(y_pred, y_test)
```


```python
sb.heatmap(cm, annot=True, fmt="d")
# results with tuning
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1eba9994e50>




![png](https://github.com/Sekomer/Breast-Cancer-With-Support-Vector-Machine/blob/master/photos/output_21_1.png)



```python
cm
```




    >>>array([[ 61,   0],
           [  2, 108]], dtype=int64)




```python
classification_report(y_test, y_pred)
```




                    precision   recall   f1-score   support
             0.0       1.00      0.97      0.98        63
             1.0       0.98      1.00      0.99       108
             
        accuracy                           0.99       171               
        macro avg       0.99      0.98     0.99       171
        weighted avg    0.99      0.99     0.99       171
             




```python
