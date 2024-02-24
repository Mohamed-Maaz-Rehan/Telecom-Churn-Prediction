#!/usr/bin/env python
# coding: utf-8

# In[27]:


## supress warnings

import warnings
warnings.filterwarnings('ignore')


# In[28]:


# Importing necessary libraries 

import numpy as np
import pandas as pd 

import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler


# In[29]:


df = pd.read_csv('telecom.csv')


# In[31]:


df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Fill NaN values with 0 (or any other appropriate value)
df['TotalCharges'] = df['TotalCharges'].fillna(0).astype(int)


# In[32]:


df.drop(columns=['customerID'], inplace=True)


# In[33]:


cat_cols = list(df.select_dtypes(include=['object']).columns)


# In[34]:


df_encoded = pd.get_dummies(df, columns=cat_cols)

def map_to_01(value):
    return 1 if value else 0

df_encoded_01 = df_encoded.applymap(map_to_01)

print(df_encoded_01.head())


# In[35]:


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


# In[36]:


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


# In[37]:


from sklearn.tree import DecisionTreeClassifier

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



# In[38]:


from sklearn.ensemble import RandomForestClassifier

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier()

# Fit the model on the training data
rf_classifier.fit(X_train, y_train)

# Predict on the training data
y_train_pred = rf_classifier.predict(X_train)

# Predict on the testing data
y_test_pred = rf_classifier.predict(X_test)

# Calculate the training accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
print("Training Accuracy:", train_accuracy)

# Calculate the testing accuracy
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Testing Accuracy:", test_accuracy)


# In[39]:


from sklearn.svm import SVC

# Initialize the SVM classifier
svm_classifier = SVC()

# Fit the model on the training data
svm_classifier.fit(X_train, y_train)

# Predict on the training data
y_train_pred = svm_classifier.predict(X_train)

# Predict on the testing data
y_test_pred = svm_classifier.predict(X_test)

# Calculate the training accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
print("Training Accuracy:", train_accuracy)

# Calculate the testing accuracy
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Testing Accuracy:", test_accuracy)


# In[40]:


from sklearn.neighbors import KNeighborsClassifier

# Initialize the kNN classifier
knn_classifier = KNeighborsClassifier()

# Fit the model on the training data
knn_classifier.fit(X_train, y_train)

# Predict on the training data
y_train_pred = knn_classifier.predict(X_train)

# Predict on the testing data
y_test_pred = knn_classifier.predict(X_test)

# Calculate the training accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
print("Training Accuracy:", train_accuracy)

# Calculate the testing accuracy
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Testing Accuracy:", test_accuracy)


# # All models are overfitting here

# In[ ]:




