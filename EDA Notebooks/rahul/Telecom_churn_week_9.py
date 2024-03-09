#!/usr/bin/env python
# coding: utf-8

# In[2]:


## supress warnings

import warnings
warnings.filterwarnings('ignore')


# In[3]:


import numpy as np
import pandas as pd 

import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler


# In[4]:


df = pd.read_csv('telecom.csv')


# In[5]:


df.drop(columns=['customerID'], inplace=True)


# In[6]:


cat_cols = list(df.select_dtypes(include=['object']).columns)


# In[7]:


df_encoded = pd.get_dummies(df, columns=cat_cols)

def map_to_01(value):
    return 1 if value else 0

df_encoded_01 = df_encoded.applymap(map_to_01)

print(df_encoded_01.head())


# In[8]:


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


# In[9]:


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


# In[10]:


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



# In[11]:


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


# In[12]:


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


# In[13]:


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


# # Regularization

# In[16]:


# Initialize the logistic regression model with regularization (L2)
log_reg_l2 = LogisticRegression(C=1.0, penalty='l2', max_iter=1000)

# Fit the model on the training data
log_reg_l2.fit(X_train, y_train)

# Predict on the training data
y_train_pred_l2 = log_reg_l2.predict(X_train)

# Predict on the testing data
y_test_pred_l2 = log_reg_l2.predict(X_test)

# Calculate the accuracy of the model on the training data
train_accuracy_l2 = log_reg_l2.score(X_train, y_train)
print("Training Accuracy with Regularization:", train_accuracy_l2)

# Calculate the accuracy of the model on the testing data
test_accuracy_l2 = log_reg_l2.score(X_test, y_test)
print("Testing Accuracy with Regularization:", test_accuracy_l2)


# In[17]:


# Initialize the logistic regression model with L1 regularization
log_reg_l1 = LogisticRegression(C=1.0, penalty='l1', solver='liblinear', max_iter=1000)

# Fit the model on the training data
log_reg_l1.fit(X_train, y_train)

# Predict on the training data
y_train_pred_l1 = log_reg_l1.predict(X_train)

# Predict on the testing data
y_test_pred_l1 = log_reg_l1.predict(X_test)

# Calculate the training accuracy
train_accuracy_l1 = accuracy_score(y_train, y_train_pred_l1)
print("Training Accuracy with L1 Regularization:", train_accuracy_l1)

# Calculate the testing accuracy
test_accuracy_l1 = accuracy_score(y_test, y_test_pred_l1)
print("Testing Accuracy with L1 Regularization:", test_accuracy_l1)


# In[18]:


# Initialize the logistic regression model with Elastic Net regularization
log_reg_en = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=1000)

# Fit the model on the training data
log_reg_en.fit(X_train, y_train)

# Predict on the training data
y_train_pred_en = log_reg_en.predict(X_train)

# Predict on the testing data
y_test_pred_en = log_reg_en.predict(X_test)

# Calculate the training accuracy
train_accuracy_en = accuracy_score(y_train, y_train_pred_en)
print("Training Accuracy with Elastic Net Regularization:", train_accuracy_en)

# Calculate the testing accuracy
test_accuracy_en = accuracy_score(y_test, y_test_pred_en)
print("Testing Accuracy with Elastic Net Regularization:", test_accuracy_en)


# In[19]:


# Initialize the decision tree classifier with cost complexity pruning
clf = DecisionTreeClassifier(ccp_alpha=0.01)

# Fit the model on the training data
clf.fit(X_train, y_train)

# Predict on the training data
y_train_pred = clf.predict(X_train)

# Predict on the testing data
y_test_pred = clf.predict(X_test)

# Calculate the training accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
print("Training Accuracy with Cost Complexity Pruning:", train_accuracy)

# Calculate the testing accuracy
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Testing Accuracy with Cost Complexity Pruning:", test_accuracy)


# In[20]:


# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42)

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


# In[21]:


# Initialize the SVM classifier with regularization
svm_classifier = SVC(C=1.0, kernel='rbf', gamma='scale', random_state=42)

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


# In[22]:


# Initialize the KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=5)

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


# In[ ]:




