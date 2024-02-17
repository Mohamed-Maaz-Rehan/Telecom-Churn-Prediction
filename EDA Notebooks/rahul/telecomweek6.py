#!/usr/bin/env python
# coding: utf-8

# # Telecom Churn Prediction 

# In[216]:


## supress warnings

import warnings
warnings.filterwarnings('ignore')


# In[217]:


# Importing necessary libraries 

import numpy as np
import pandas as pd 

import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler


# In[218]:


## Importing a dataset 

# There are two ways to import dataset 
# 1. Path 
# 2. filename

df = pd.read_csv('telecom.csv')


# In[219]:


df


# In[220]:


# To check the top 5 rows in the dataset 

df.head()


# In[221]:


# To check the bottom 5 rows in the dataset 

df.tail()


# In[222]:


# To check the shape of the dataset 

df.shape


# In[223]:


# To get the statistical info of the dataset 

df.describe()


# In[224]:


## To check the info of the dataset 

df.info()


# In[225]:


# two libraries for data visualization

## seaborn 
## matplotlib 


# In[226]:


df['Churn'].value_counts()


# In[227]:


missing_values = df.isnull().sum()


# In[228]:


missing_values


# In[229]:


duplicates = df[df.duplicated()]


# In[230]:


duplicates


# In[231]:


column_data_types = df.dtypes


# In[232]:


num_cols = df.select_dtypes(include=['float64', 'int64']).columns


# In[233]:


z_scores = np.abs((df[num_cols] - df[num_cols].mean()) / df[num_cols].std())


# In[234]:


threshold = 3


# In[235]:


outliers = df[(z_scores > threshold).any(axis=1)]


# In[236]:


outliers


# In[237]:


#check for inconsistencies
for column in df.columns:
    unique_values = df[column].unique()
    if len(unique_values) > 1:
        print(f"Inconsistency found in column '{column}': {unique_values}")


# In[238]:


string_columns = df.select_dtypes(include=['object']).columns
for column in string_columns:
    unique_values = df[column].unique()
    print(f"Column '{column}': {unique_values}")


# In[239]:


df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Fill NaN values with 0 (or any other appropriate value)
df['TotalCharges'] = df['TotalCharges'].fillna(0).astype(int)


# In[240]:


df.drop(columns='customerID', inplace=True)


# In[241]:


df.info()


# In[242]:


numerical_columns = df.select_dtypes(include=['float64', 'int64','int32']).columns
for column in numerical_columns:
    unique_values = df[column].unique()
    print(f"Column '{column}': {unique_values}")


# In[243]:


df.describe()


# In[244]:


corr_matrix = df[numerical_columns].corr()

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix of Telecom Churn Dataset')
plt.show()


# In[245]:


from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)


# In[265]:


train_df.shape


# In[266]:


test_df.shape


# In[267]:


train_df


# In[268]:


train_df.info()


# In[ ]:





# In[269]:


train_df.head()


# In[270]:


categorical_columns = list(df.select_dtypes(include=['object']).columns)


# In[271]:


from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the categorical column
for i in categorical_columns:
    train_df[i] = label_encoder.fit_transform(train_df[i])


# In[272]:


train_df


# In[273]:


from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform the categorical column
for i in categorical_columns:
    test_df[i] = label_encoder.fit_transform(test_df[i])


# In[274]:


test_df


# In[275]:


X_train = train_df.drop(columns=['Churn'])
y_train = train_df['Churn']
X_test = test_df.drop(columns=['Churn'])
y_test = test_df['Churn']

# Print the shapes of the resulting datasets
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)


# In[278]:


from imblearn.over_sampling import SMOTE

# Create SMOTE object
smote = SMOTE(random_state=42)

# Resample the data
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)


# In[ ]:





# In[ ]:





# In[ ]:




