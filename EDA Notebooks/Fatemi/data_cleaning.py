#!/usr/bin/env python
# coding: utf-8

# In[33]:


# Importing libraries 

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt


# In[6]:


#importing the dataset by reading the csv file
df = pd.read_csv('Telecom Churn Prediction.csv')

#displaying the first five rows of dataset 
print(df.head())


# In[26]:


#keeping customerID, gender, SeniorCitizen and churn columns and dropping others
columns_to_drop = ['Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges']
new_df = df.drop(columns=columns_to_drop)

# Save the new dataset to a new CSV file
new_df.to_csv("new_telecom.csv", index=False)


# In[27]:


print(new_df.head())


# In[16]:


#find in the missing data in the dataset
new_df.isnull()


# In[17]:


#find in the missing data in the dataset with other method
new_df.isna().any()


# In[19]:


#check for duplicate data
new_df.duplicated()


# In[21]:


# check the sum of duplicates in the data set

new_df.duplicated().sum()


# In[28]:


#convert seniorcitizen column values to yes no from 0 and 1
new_df['SeniorCitizen'] = new_df['SeniorCitizen'].map({1: 'Yes', 0: 'No'})


# In[29]:


new_df.head()


# In[30]:


new_df.info()


# In[31]:


for col in new_df.columns:
    unique_val = df[col].unique()
    if len(unique_val) > 1:
        print(f"Column Inconsistency '{col}': {unique_val}")


# In[32]:


new_df['Churn'].value_counts()


# In[35]:


bar_graph = sns.countplot(df['SeniorCitizen'])
bar_graph.bar_label(bar_graph.containers[0])
plt.xticks(rotation=45)
plt.title("Bar Plot of SeniorCitizen")
plt.xlabel("SeniorCitizen")
plt.ylabel("Count")
plt.show()


# In[ ]:




