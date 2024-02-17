#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
from sklearn.model_selection import train_test_split


# In[3]:


df = pd.read_csv('Telecom Churn Prediction.csv')


# In[4]:


df.head


# In[6]:


df.shape


# In[7]:


df.mean


# In[9]:


df.info()


# In[10]:


df['Churn'].value_counts()


# In[14]:


df['Churn'].value_counts().plot(kind = 'pie')


# In[16]:


print(df.isnull().sum())


# In[ ]:




