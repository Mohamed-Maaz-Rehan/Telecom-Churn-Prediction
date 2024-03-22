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


df_encoded_01['Churn'] = df_encoded_01['Churn'].map({'Yes': 1, 'No': 0})


# In[8]:


from sklearn.model_selection import train_test_split

# Split the data into features (X) and target variable (y)
X = df_encoded_01.drop(columns=['Churn'])  # Features
y = df_encoded_01['Churn']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shapes of the resulting datasets
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)


# # Standardization for imbalanced data

# In[9]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Instantiate and train the Logistic Regression model
log_reg_model = LogisticRegression()
log_reg_model.fit(X_train_scaled, y_train)

# Make predictions on the training and testing data
y_train_pred = log_reg_model.predict(X_train_scaled)
y_test_pred = log_reg_model.predict(X_test_scaled)

# Calculate training and testing accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Calculate training and testing F1 score
train_f1_score = f1_score(y_train, y_train_pred)
test_f1_score = f1_score(y_test, y_test_pred)

# Print the results
print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)
print("Training F1 Score:", train_f1_score)
print("Testing F1 Score:", test_f1_score)


# In[10]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

# Instantiate the Decision Tree classifier
tree_clf = DecisionTreeClassifier()

# Train the model
tree_clf.fit(X_train_scaled, y_train)

# Make predictions on the training and testing data
y_train_pred = tree_clf.predict(X_train_scaled)
y_test_pred = tree_clf.predict(X_test_scaled)

# Calculate training and testing accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Calculate training and testing F1 score
train_f1_score = f1_score(y_train, y_train_pred)
test_f1_score = f1_score(y_test, y_test_pred)

# Print the results
print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)
print("Training F1 Score:", train_f1_score)
print("Testing F1 Score:", test_f1_score)


# In[11]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Instantiate the Random Forest classifier
rf_classifier = RandomForestClassifier()

# Train the model
rf_classifier.fit(X_train_scaled, y_train)

# Make predictions on the training and testing data
y_train_pred = rf_classifier.predict(X_train_scaled)
y_test_pred = rf_classifier.predict(X_test_scaled)

# Calculate training and testing accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Calculate training and testing F1 score
train_f1_score = f1_score(y_train, y_train_pred)
test_f1_score = f1_score(y_test, y_test_pred)

# Print the results
print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)
print("Training F1 Score:", train_f1_score)
print("Testing F1 Score:", test_f1_score)


# In[12]:


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

# Instantiate the SVM classifier
svm_classifier = SVC()

# Train the model
svm_classifier.fit(X_train_scaled, y_train)

# Make predictions on the training and testing data
y_train_pred = svm_classifier.predict(X_train_scaled)
y_test_pred = svm_classifier.predict(X_test_scaled)

# Calculate training and testing accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Calculate training and testing F1 score
train_f1_score = f1_score(y_train, y_train_pred)
test_f1_score = f1_score(y_test, y_test_pred)

# Print the results
print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)
print("Training F1 Score:", train_f1_score)
print("Testing F1 Score:", test_f1_score)


# In[13]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

# Instantiate the kNN classifier
knn_classifier = KNeighborsClassifier()

# Train the model
knn_classifier.fit(X_train_scaled, y_train)

# Make predictions on the training and testing data
y_train_pred = knn_classifier.predict(X_train_scaled)
y_test_pred = knn_classifier.predict(X_test_scaled)

# Calculate training and testing accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Calculate training and testing F1 score
train_f1_score = f1_score(y_train, y_train_pred)
test_f1_score = f1_score(y_test, y_test_pred)

# Print the results
print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)
print("Training F1 Score:", train_f1_score)
print("Testing F1 Score:", test_f1_score)


# # Normalisation for imbalanced data

# In[14]:


from sklearn.preprocessing import MinMaxScaler

# Initialize the MinMaxScaler
scaler = MinMaxScaler()

# Fit and transform the training data
X_train_normalized = scaler.fit_transform(X_train)

# Transform the testing data
X_test_normalized = scaler.transform(X_test)


# In[15]:


# Instantiate the logistic regression model
log_reg_model = LogisticRegression()

# Train the model on the normalized training data
log_reg_model.fit(X_train_normalized, y_train)

# Make predictions on the normalized training and testing data
y_train_pred = log_reg_model.predict(X_train_normalized)
y_test_pred = log_reg_model.predict(X_test_normalized)

# Calculate training and testing accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Calculate training and testing F1 score
train_f1_score = f1_score(y_train, y_train_pred)
test_f1_score = f1_score(y_test, y_test_pred)

# Print the results
print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)
print("Training F1 Score:", train_f1_score)
print("Testing F1 Score:", test_f1_score)


# In[16]:


# Instantiate the Decision Tree classifier
tree_clf = DecisionTreeClassifier()

# Train the model on the normalized training data
tree_clf.fit(X_train_normalized, y_train)

# Make predictions on the normalized training and testing data
y_train_pred = tree_clf.predict(X_train_normalized)
y_test_pred = tree_clf.predict(X_test_normalized)

# Calculate training and testing accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Calculate training and testing F1 score
train_f1_score = f1_score(y_train, y_train_pred)
test_f1_score = f1_score(y_test, y_test_pred)

# Print the results
print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)
print("Training F1 Score:", train_f1_score)
print("Testing F1 Score:", test_f1_score)


# In[17]:


# Instantiate the Random Forest classifier
rf_classifier = RandomForestClassifier()

# Train the model on the normalized training data
rf_classifier.fit(X_train_normalized, y_train)

# Make predictions on the normalized training and testing data
y_train_pred = rf_classifier.predict(X_train_normalized)
y_test_pred = rf_classifier.predict(X_test_normalized)

# Calculate training and testing accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Calculate training and testing F1 score
train_f1_score = f1_score(y_train, y_train_pred)
test_f1_score = f1_score(y_test, y_test_pred)

# Print the results
print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)
print("Training F1 Score:", train_f1_score)
print("Testing F1 Score:", test_f1_score)


# In[18]:


# Instantiate the SVM classifier
svm_classifier = SVC()

# Train the model on the normalized training data
svm_classifier.fit(X_train_normalized, y_train)

# Make predictions on the normalized training and testing data
y_train_pred = svm_classifier.predict(X_train_normalized)
y_test_pred = svm_classifier.predict(X_test_normalized)

# Calculate training and testing accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Calculate training and testing F1 score
train_f1_score = f1_score(y_train, y_train_pred)
test_f1_score = f1_score(y_test, y_test_pred)

# Print the results
print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)
print("Training F1 Score:", train_f1_score)
print("Testing F1 Score:", test_f1_score)


# In[19]:


# Instantiate the kNN classifier
knn_classifier = KNeighborsClassifier()

# Train the model on the normalized training data
knn_classifier.fit(X_train_normalized, y_train)

# Make predictions on the normalized training and testing data
y_train_pred = knn_classifier.predict(X_train_normalized)
y_test_pred = knn_classifier.predict(X_test_normalized)

# Calculate training and testing accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Calculate training and testing F1 score
train_f1_score = f1_score(y_train, y_train_pred)
test_f1_score = f1_score(y_test, y_test_pred)

# Print the results
print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)
print("Training F1 Score:", train_f1_score)
print("Testing F1 Score:", test_f1_score)


# In[ ]:




