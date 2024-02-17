#!/usr/bin/env python
# coding: utf-8

# # Telecom Churn Prediction 

# In[127]:


## supress warnings

import warnings
warnings.filterwarnings('ignore')


# In[128]:


# Importing necessary libraries 

import numpy as np
import pandas as pd 

import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler


# In[129]:


## Importing a dataset 

# There are two ways to import dataset 
# 1. Path 
# 2. filename

df = pd.read_csv('telecom.csv')


# In[130]:


# To check the top 5 rows in the dataset 

df.head()


# In[131]:


# To check the bottom 5 rows in the dataset 

df.tail()


# In[132]:


# To check the shape of the dataset 

df.shape


# In[133]:


# To get the statistical info of the dataset 

df.describe()


# In[134]:


## To check the info of the dataset 

df.info()


# In[135]:


# two libraries for data visualization

## seaborn 
## matplotlib 


# In[136]:


df['Churn'].value_counts()


# In[139]:


missing_values = df.isnull().sum()


# In[140]:


missing_values


# In[141]:


duplicates = df[df.duplicated()]


# In[142]:


duplicates


# In[143]:


column_data_types = df.dtypes


# In[144]:


num_cols = df.select_dtypes(include=['float64', 'int64']).columns


# In[145]:


z_scores = np.abs((df[num_cols] - df[num_cols].mean()) / df[num_cols].std())


# In[146]:


threshold = 3


# In[147]:


outliers = df[(z_scores > threshold).any(axis=1)]


# In[148]:


outliers


# In[149]:


#check for inconsistencies
for column in df.columns:
    unique_values = df[column].unique()
    if len(unique_values) > 1:
        print(f"Inconsistency found in column '{column}': {unique_values}")


# In[150]:


string_columns = df.select_dtypes(include=['object']).columns
for column in string_columns:
    unique_values = df[column].unique()
    print(f"Column '{column}': {unique_values}")


# In[151]:


numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
for column in numerical_columns:
    unique_values = df[column].unique()
    print(f"Column '{column}': {unique_values}")


# In[152]:


# Check for class imbalances
target_column = 'Churn'
class_distribution = df[target_column].value_counts(normalize=True)


# In[153]:


class_distribution


# In[154]:


categorical_columns = df.select_dtypes(include=['object']).columns
for column in categorical_columns:
    if column != target_column:
        category_distribution = df[column].value_counts(normalize=True)
        print(f"\nDistribution of '{column}':")
        print(category_distribution)


# In[ ]:




