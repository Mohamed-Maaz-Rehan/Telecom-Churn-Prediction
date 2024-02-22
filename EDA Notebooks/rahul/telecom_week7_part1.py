#!/usr/bin/env python
# coding: utf-8

# In[1]:


## supress warnings

import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Importing necessary libraries 

import numpy as np
import pandas as pd 

import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler


# In[3]:


df = pd.read_csv('telecom.csv')


# In[4]:


cat_cols = list(df.select_dtypes(include=['object']).columns)


# In[5]:


df_encoded = pd.get_dummies(df, columns=cat_cols)

def map_to_01(value):
    return 1 if value else 0

df_encoded_01 = df_encoded.applymap(map_to_01)

print(df_encoded_01.head())


# In[10]:


from sklearn.model_selection import train_test_split

# Split the data into features (X) and target variable (y)
X = df_encoded_01.drop(columns=['Churn_Yes','Churn_No'])  # Features
y = df_encoded_01['Churn_Yes']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Print the shapes of the resulting datasets
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)


# In[15]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Initialize the logistic regression model
log_reg = LogisticRegression()

# Fit the model on the training data
log_reg.fit(X_train, y_train)

# Predict on the training data
y_train_pred = log_reg.predict(X_train)

# Predict on the testing data
y_test_pred = log_reg.predict(X_test)

# Calculate the training accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
print("Training Accuracy:", train_accuracy)

# Calculate the testing accuracy
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Testing Accuracy:", test_accuracy)


# In[16]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Initialize the decision tree classifier
tree_clf = DecisionTreeClassifier()

# Fit the classifier on the training data
tree_clf.fit(X_train, y_train)

# Predict on the training data
y_train_pred = tree_clf.predict(X_train)

# Predict on the testing data
y_test_pred = tree_clf.predict(X_test)

# Calculate the training accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
print("Training Accuracy:", train_accuracy)

# Calculate the testing accuracy
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Testing Accuracy:", test_accuracy)



# In[ ]:




