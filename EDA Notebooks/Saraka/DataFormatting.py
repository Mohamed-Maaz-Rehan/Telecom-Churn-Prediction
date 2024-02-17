#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv('Telecom Churn Prediction.csv')


# In[3]:


df.head


# In[4]:


df.shape


# In[5]:


df.mean


# In[6]:


df.info()


# In[7]:


df['Churn'].value_counts()


# In[21]:


a = df['Churn'].value_counts().plot(kind = 'pie', y='Churn', autopct='%1.0f%%')
plt.show()


# In[9]:


print(df.isnull().sum())


# In[10]:


df['Partner']=df['Partner'].replace({'No': 0, 'Yes': 1})


# In[11]:


df['Dependents']=df['Dependents'].replace({'No': 0, 'Yes': 1})


# In[12]:


df['Churn']=df['Churn'].replace({'No': 0, 'Yes': 1})


# In[22]:


df2 = df.drop(df.columns.difference(['Partner','Dependents','tenure','Churn']), axis=1)


# In[23]:


df2


# In[24]:


df2.corr()


# In[25]:


dataplot = sns.heatmap(df2.corr(), annot=True) 
plt.show()

