#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[2]:


import numpy as np
import pandas as pd 

import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler


# In[3]:


df = pd.read_csv('telecom.csv')


# In[4]:


df.drop(columns=['customerID'], inplace=True)


# In[5]:


cat_cols = list(df.select_dtypes(include=['object']).columns)


# In[6]:


def map_to_01(value):
    return 1 if value else 0

# Specify the column to exclude
exclude_column = 'Churn'

# Apply the map_to_01 function to all columns except the exclude_column
columns_to_map = df.columns[df.columns != exclude_column]
df_encoded_01 = df.copy()
df_encoded_01[columns_to_map] = df_encoded_01[columns_to_map].applymap(map_to_01)

print(df_encoded_01.head())


# In[7]:


df_encoded_01.dtypes


# In[8]:


df_encoded_01['Churn'] = df_encoded_01['Churn'].map({'Yes': 1, 'No': 0})


# In[9]:


# Assuming df is your DataFrame containing the data
# Replace df with the actual name of your DataFrame

# Specify the values you want to filter for
desired_values = {0, 1, 0.0, 1.0}  # You can modify this set to include other values if needed

# Create a list to store column names meeting the condition
columns_with_mixed_values = []

# Iterate over the columns and check if they contain a mix of desired values
for column in df_encoded_01.columns:
    unique_values = df_encoded_01[column].dropna().unique()  # Drop NaN values and get unique values
    if set(unique_values) <= desired_values:
        columns_with_mixed_values.append(column)

# Print the column names meeting the condition
print("Column(s) containing a mix of 1s, 0s, and possibly other values:")
print(columns_with_mixed_values)


# In[10]:


import pandas as pd

# Assuming df is your DataFrame containing the data
# Replace df with the actual name of your DataFrame

# Create a list to store column names meeting the condition
columns_with_other_values = []

# Iterate over the columns and check if they contain values other than 0 and 1
for column in df_encoded_01.columns:
    unique_values = df_encoded_01[column].dropna().unique()  # Drop NaN values and get unique values
    if set(unique_values) - {0, 1}:  # Check if there are values other than 0 and 1
        columns_with_other_values.append(column)

# Print the column names meeting the condition
print("Column(s) containing values other than 0 and 1:")
print(columns_with_other_values)


# In[11]:


df_encoded_01.dtypes


# In[12]:


df_encoded_01


# In[13]:


from imblearn.over_sampling import SMOTE
from collections import Counter


# In[14]:


# Original class distribution
print("Original class distribution:", df['Churn'].value_counts())

# Separate features (X) and target variable (y)
X = df_encoded_01.drop('Churn', axis=1)
y = df_encoded_01['Churn']

# Apply SMOTE to generate synthetic samples
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# New class distribution after applying SMOTE
print("New class distribution after SMOTE:", Counter(y_resampled))


# In[15]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)


# In[21]:


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# Step 2: Define the hyperparameters to tune
param_grid = {
    'penalty': ['l1', 'l2'],
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear']
}

# Step 3: Perform Grid Search with 5-fold cross-validation
logistic_classifier = LogisticRegression()
grid_search = GridSearchCV(estimator=logistic_classifier, param_grid=param_grid, cv=5, scoring='f1')
grid_search.fit(X_train, y_train)

# Step 4: Get the best model and evaluate its performance
best_logistic_classifier = grid_search.best_estimator_

# Training set predictions
y_train_pred = best_logistic_classifier.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)

# Testing set predictions
y_test_pred = best_logistic_classifier.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)

print("Training Accuracy:", train_accuracy)
print("Training F1 Score:", train_f1)
print("Testing Accuracy:", test_accuracy)
print("Testing F1 Score:", test_f1)


# In[22]:


from sklearn.tree import DecisionTreeClassifier

# Step 2: Define the hyperparameters to tune
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Step 3: Perform Grid Search with 5-fold cross-validation
dt_classifier = DecisionTreeClassifier()
grid_search = GridSearchCV(estimator=dt_classifier, param_grid=param_grid, cv=5, scoring='f1')
grid_search.fit(X_train, y_train)

# Step 4: Get the best model and evaluate its performance
best_dt_classifier = grid_search.best_estimator_

# Training set predictions
y_train_pred = best_dt_classifier.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)

# Testing set predictions
y_test_pred = best_dt_classifier.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)

print("Training Accuracy:", train_accuracy)
print("Training F1 Score:", train_f1)
print("Testing Accuracy:", test_accuracy)
print("Testing F1 Score:", test_f1)


# In[23]:


from sklearn.ensemble import RandomForestClassifier

# Step 2: Define the hyperparameters to tune
param_grid = {
    'n_estimators': [100, 200, 300],
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Step 3: Perform Grid Search with 5-fold cross-validation
rf_classifier = RandomForestClassifier()
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, scoring='f1')
grid_search.fit(X_train, y_train)

# Step 4: Get the best model and evaluate its performance
best_rf_classifier = grid_search.best_estimator_

# Training set predictions
y_train_pred = best_rf_classifier.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)

# Testing set predictions
y_test_pred = best_rf_classifier.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)

print("Training Accuracy:", train_accuracy)
print("Training F1 Score:", train_f1)
print("Testing Accuracy:", test_accuracy)
print("Testing F1 Score:", test_f1)


# In[24]:


from sklearn.neighbors import KNeighborsClassifier

# Step 2: Define the hyperparameters to tune
param_grid = {
    'n_neighbors': [3, 5, 7, 9],  # number of neighbors to consider
    'weights': ['uniform', 'distance'],  # weight function used in prediction
    'p': [1, 2]  # power parameter for Minkowski distance
}

# Step 3: Perform Grid Search with 5-fold cross-validation
knn_classifier = KNeighborsClassifier()
grid_search = GridSearchCV(estimator=knn_classifier, param_grid=param_grid, cv=5, scoring='f1')
grid_search.fit(X_train, y_train)

# Step 4: Get the best model and evaluate its performance
best_knn_classifier = grid_search.best_estimator_

# Training set predictions
y_train_pred = best_knn_classifier.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)

# Testing set predictions
y_test_pred = best_knn_classifier.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)

print("Training Accuracy:", train_accuracy)
print("Training F1 Score:", train_f1)
print("Testing Accuracy:", test_accuracy)
print("Testing F1 Score:", test_f1)


# In[25]:


from sklearn.svm import SVC


# Step 2: Define the hyperparameters to tune
param_grid = {
    'C': [0.1, 1, 10, 100],               # Regularization parameter
    'kernel': ['linear', 'rbf', 'poly'],  # Kernel type
    'gamma': ['scale', 'auto'],           # Kernel coefficient
    'degree': [2, 3, 4],                  # Degree for 'poly' kernel
}

# Step 3: Perform Grid Search with 5-fold cross-validation
svm_classifier = SVC()
grid_search = GridSearchCV(estimator=svm_classifier, param_grid=param_grid, cv=5, scoring='f1')
grid_search.fit(X_train, y_train)

# Step 4: Get the best model and evaluate its performance
best_svm_classifier = grid_search.best_estimator_

# Training set predictions
y_train_pred = best_svm_classifier.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)
train_f1 = f1_score(y_train, y_train_pred)

# Testing set predictions
y_test_pred = best_svm_classifier.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)

print("Training Accuracy:", train_accuracy)
print("Training F1 Score:", train_f1)
print("Testing Accuracy:", test_accuracy)
print("Testing F1 Score:", test_f1)


# In[ ]:




