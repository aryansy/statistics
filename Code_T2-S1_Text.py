#!/usr/bin/env python
# coding: utf-8

# # Importing Necessary Libraries

# In[ ]:


import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns


# In[156]:


df_train = pd.read_csv("train_data.csv")
df_train.head()


# In[157]:


df_test = pd.read_csv("test_data.csv")
df_test.columns


# In[158]:


df_test = df_test[['College_T1', 'College_T2',
       'Role_Manager', 'City_Metro', 'previous CTC', 'previous job changes',
       'Graduation marks', 'Exp', 'Actual CTC', 'Predicted CTC']]
df_test = df_test.drop(df_test.index[1338])


# In[159]:


df_test.head(2)


# # Categorical to Numerical Columns

# In[160]:


df_train['Previous CTC'] = df_train['Previous CTC'].str.replace(',','').astype(float)
df_train['CTC'] = df_train['CTC'].str.replace(',','').astype(float)


# In[161]:


#df_train = df_train.drop("College", axis=1)
df_train =pd.get_dummies(df_train, drop_first=True)


# In[162]:


df_train.head(1)


# # Data Description & Summary

# In[163]:


df_train.info()


# In[62]:


#df_train['Previous CTC'] = df_train['Previous CTC'].str.replace(',','').astype(float)
#df_train['CTC'] = df_train['CTC'].str.replace(',','').astype(float)


# In[164]:


df_train.describe()


# In[165]:


df_train.isna().sum()


# # Visualization

# In[33]:


#   "" Linear Regression assumptions:  ""
# 1. No multicollinearity - obs indp. of each other
# 2. Multivariate normality
# 3. Linear Relationship between X & Y
# 4. Homoscedasticity - var of residual is constant


# In[166]:


sns.pairplot(df_train[['Previous CTC','Previous job changes', 'Graduation marks', 'Exp (Months)', 'CTC']])


# # Matching the Training & Test column names

# In[175]:


df_train.columns


# In[178]:


df_train = df_train[[ 'College_Tier 2', 'College_Tier 3', 'Role_Manager', 'City type_Non-Metro',
                     'Previous CTC', 'Previous job changes', 'Graduation marks', 'Exp (Months)', 'CTC']]
df_train.head(1)


# In[183]:


df_train.rename(columns = {'College_Tier 2':'College_T1', 'College_Tier 3':'College_T2', 'City type_Non-Metro':'City_Metro',
                          'Previous CTC':'previous CTC', 'Previous job changes': 'previous job changes','Exp (Months)':'Exp',
                          'CTC':'Actual CTC'}, inplace=True)
df_train.head(1)


# In[189]:


df_test = df_test.drop('Predicted CTC', axis=1)
df_test.head()


# In[190]:


df_test.columns


# In[191]:


df_train.columns


# # Splitting into Train & Test for Modelling the Train Data

# In[192]:


X_train = df_train.drop('Actual CTC', axis = 1)
y_train = df_train['Actual CTC']


# In[193]:


X_test = df_test.drop('Actual CTC', axis = 1)
y_test = df_test['Actual CTC']


# In[194]:


y_train.head()


# # Scaling Train & Test data

# In[195]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[196]:


X_train_scaled = scaler.fit_transform(X_train)


# In[197]:


X_test_scaled = scaler.fit_transform(X_test)


# # Fitting a LRM on Train data

# In[198]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X_train_scaled,y_train)


# In[199]:


reg.coef_  # weight vector


# In[202]:


X_test.head()


# # Prediction on Test data

# In[203]:


y_test_predict = reg.predict(X_test_scaled)


# In[208]:


pd.DataFrame({'test_Actual': y_test, 'test_Predicted':y_test_predict})


# # Evaluation

# In[218]:


import numpy as np
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse

print('mean_abs_error:', mae(y_test,y_test_predict))
print('root_mean_sq_error:', np.sqrt(mse(y_test,y_test_predict)))


# In[ ]:




