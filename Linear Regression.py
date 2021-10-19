#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


BostonHousing = pd.read_csv("BostonHousing.csv")
BostonHousing


# In[4]:



Y = BostonHousing.medv
Y


# In[5]:


X = BostonHousing.drop(['medv'], axis=1)
X


# In[6]:


from sklearn.model_selection import train_test_split


# In[7]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


# In[8]:


X_train.shape, Y_train.shape


# In[9]:


X_test.shape, Y_test.shape


# In[10]:


from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


# In[11]:


model = linear_model.LinearRegression()


# In[12]:


model.fit(X_train, Y_train)


# In[13]:


Y_pred = model.predict(X_test)


# In[14]:


print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('Mean squared error (MSE): %.2f'
      % mean_squared_error(Y_test, Y_pred))
print('Coefficient of determination (R^2): %.2f'
      % r2_score(Y_test, Y_pred))


# In[15]:


r2_score(Y_test, Y_pred)


# In[16]:


r2_score(Y_test, Y_pred).dtype


# In[17]:


'%f' % 0.523810833536016


# In[18]:


'%.3f' % 0.523810833536016


# In[19]:


'%.2f' % 0.523810833536016


# In[20]:


import seaborn as sns


# In[21]:


Y_test


# In[22]:


import numpy as np
np.array(Y_test)


# In[23]:


Y_pred


# In[24]:


sns.scatterplot(Y_test, Y_pred)


# In[25]:


sns.scatterplot(Y_test, Y_pred, marker="+")


# In[26]:


sns.scatterplot(Y_test, Y_pred, alpha=0.5)


# In[ ]:




