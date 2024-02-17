#!/usr/bin/env python
# coding: utf-8

# # Importing necessary libraries 

# In[1]:


# Importing for numerical computing
import numpy as np

# Importing for data manipulation
import pandas as pd

# Importing for statistical data visualization
import seaborn as sns

# Importing for creating plots and visualizations
import matplotlib.pyplot as plt

# Importing to transform categorical variables into numerical
from sklearn.preprocessing import LabelEncoder

# Importing for splitting data into training and testing sets.
from sklearn.model_selection import train_test_split


# In[2]:


# Suppress unnecessary warnings during code execution
import warnings
warnings.filterwarnings('ignore')


# In[3]:


# Reading CSV file using pandas
df = pd.read_csv('Telecom Churn Prediction.csv')


# In[4]:


# Printing first 5 rows of the dataset
df.head()


# In[5]:


# Determine the shape of the dataframe
df.shape


# In[6]:


# Get the datatypes of each column
df.info()


# In[7]:


# Print descriptive statistics for the dataFrame
df.describe()


# In[8]:


# Print descriptive statistics for the dataFrame including all columns
df.describe(include="all")


# In[9]:


# list down all the features (columns) in a dataFrame
feature_list = list(df.keys())
print(feature_list)


# In[10]:


# Checking the count of target column(variable) - Churn
df['Churn'].value_counts()


# In[11]:


# Checking for duplicate values for all columns in a DataFrame
duplicate = df[df.duplicated()]

if not duplicate.empty:
    print("Duplicate Rows Exist:")
    print(duplicate)
else:
    print("No duplicate rows")


# In[12]:


#  Checking for the duplicate values for 'customerID' column in a dataFrame
duplicate = df[df.duplicated('customerID')]

if not duplicate.empty:
    print("Duplicate Rows Exist:")
    print(duplicate)
else:
    print("No duplicate customer ID exist")


# In[13]:


# Check for null values in the DataFrame
null_values = df.isnull()
print(null_values)


# In[14]:


# The sum of null values in each column
null_counts = null_values.sum()
print(null_counts)


# In[15]:


# Define a dictionary to map categories to numeric values
gender_mapping = {'Male': 0, 'Female': 1}
mapping_values = {'No': 0, 'Yes': 1}

col_mapping_dict = {
    'gender': gender_mapping, 
    'Partner': mapping_values, 
    'Dependents': mapping_values, 
    'PhoneService': mapping_values,
    'PaperlessBilling': mapping_values,
    'Churn': mapping_values
    
}

for key,values in col_mapping_dict.items():
    df[key] = df[key].map(col_mapping_dict[key])


# In[16]:


df.head()


# In[17]:


numeric_data = df.select_dtypes(include=[np.number])
corelation = numeric_data.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corelation, xticklabels=corelation.columns, yticklabels=corelation.columns,annot=True)


# In[18]:


sns.boxplot(x=df['gender'], y=df['tenure'], hue=df['Churn'])


# In[19]:


# Define a dictionary to map categories to numeric values
multi_line_mapping = {'No': 0, 'Yes': 1, 'No phone service': 2}
internet_service_mapping = {'No': 0, 'Fiber optic': 1, 'DSL': 2}
mapping_values = {'No': 0, 'Yes': 1, 'No internet service': 2}
contract_mapping = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
payment_mapping = {'Electronic check': 0, 'Mailed check': 1, 'Bank transfer (automatic)': 2, 'Credit card (automatic)':3}

col_mapping_dict = {
    'MultipleLines': multi_line_mapping, 
    'InternetService': internet_service_mapping, 
    'OnlineSecurity': mapping_values, 
    'OnlineBackup': mapping_values,
    'DeviceProtection': mapping_values,
    'TechSupport': mapping_values,
    'StreamingTV': mapping_values,
    'StreamingMovies': mapping_values,
    'Contract': contract_mapping,
    'PaymentMethod': payment_mapping
}

for key,values in col_mapping_dict.items():
    df[key] = df[key].map(col_mapping_dict[key])


# In[20]:


df.head()


# In[21]:


unique_values = df['TotalCharges'].unique()
print(sorted(unique_values))


# In[22]:


df['TotalCharges'] = df['TotalCharges'].replace({' ': 0})
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])


# In[23]:


selected_columns = ['tenure', 'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       'PaymentMethod', 'MonthlyCharges', 'TotalCharges', 'Churn']

# Calculate the correlation matrix for selected columns
correlation = df[selected_columns].corr()

# Plot the correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".1f")
plt.title('Correlation Heatmap for Selected Columns')
plt.show()


# In[24]:


a = df['Churn'].value_counts().plot(kind = 'pie', y='Churn', autopct='%1.0f%%')
plt.show()


# In[28]:


# Dividing Target value and other features in Y and X respectively.
X = df[['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents',
       'tenure', 'PhoneService', 'MultipleLines', 'InternetService',
       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
       'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
       'PaymentMethod', 'MonthlyCharges', 'TotalCharges']]
y = df['Churn']


# In[29]:


# Spliting the data into training and testing sets with a 70(train)-30(test) split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[31]:


# pip install imbalanced-learn
from imblearn.under_sampling import RandomUnderSampler

# Define the RandomUnderSampler
under_sampler = RandomUnderSampler(random_state=42)

# Resample the dataset
X_resampled, y_resampled = under_sampler.fit_resample(X_train, y_train)


# In[ ]:




