#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler


# In[2]:


df = pd.read_csv('Telecom Churn Prediction.csv')


# In[3]:


df.shape


# In[4]:


df.describe(include='all')


# In[5]:


df = df.drop((['customerID']), axis=1)


# In[6]:


df['TotalCharges']=df['TotalCharges'].replace({' ': 0})


# In[7]:


df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])


# In[8]:


df.info()


# In[9]:


plt.figure(figsize=(20, 5))

features = ['tenure']
target = df['TotalCharges']

for i, col in enumerate(features):
    plt.subplot(1, len(features) , i+1)
    y = df[col]
    x = target
    plt.scatter(x, y, marker='o')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('TotalCharges')


# In[10]:


for i in df:
    print(df[i].value_counts())


# In[11]:


df.info()


# In[12]:


df['Churn']=df['Churn'].replace({'No': 0,'Yes':1})


# In[13]:


numeric_cols = df._get_numeric_data().columns
print(numeric_cols)
categ_cols = list(set(df.columns) - set(numeric_cols))
categ_cols


# In[14]:


for i in categ_cols:
    print(df[i].value_counts())
    print('\n')


# In[15]:


from sklearn.preprocessing import LabelEncoder

lb = LabelEncoder() 
for i in categ_cols:
    df[i] = lb.fit_transform(df[i])


# In[16]:


df.head()


# In[17]:


plt.figure(figsize = (9,8))
dataplot = sns.heatmap(df.corr().round(1), annot=True) 
plt.show()


# In[18]:


df2 = df.drop((['gender','PhoneService','MultipleLines','InternetService','StreamingTV','StreamingMovies']), axis=1)


# In[19]:


plt.figure(figsize = (9,8))
dataplot = sns.heatmap(df2.corr().round(1), annot=True) 
plt.show()


# In[20]:


#Tenure Vs. Total Charges 
#Tenure Vs. Contract


# In[21]:


df2.corr()


# In[22]:


for i in df2:
    print(i)


# In[23]:


plt.figure(figsize=(20, 5))

features = ['tenure','TotalCharges']
target = df2['Churn']

for i, col in enumerate(features):
    plt.subplot(1, len(features) , i+1)
    y = df2[col]
    x = target
    plt.scatter(x, y, marker='o')
    plt.title(col)
    plt.xlabel('Churn')
    plt.ylabel(col)


# In[24]:


from imblearn.over_sampling import RandomOverSampler

