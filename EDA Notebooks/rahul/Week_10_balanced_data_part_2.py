#!/usr/bin/env python
# coding: utf-8

# In[1]:


## supress warnings

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


# In[9]:


from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate SMOTE
smote = SMOTE(random_state=42)

# Apply SMOTE to generate synthetic samples for training data
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Instantiate logistic regression model with L1 regularization
log_reg_l1 = LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)

# Fit the model on the resampled training data
log_reg_l1.fit(X_train_resampled, y_train_resampled)

# Predict on the training data
y_train_pred_l1 = log_reg_l1.predict(X_train_resampled)

# Calculate the accuracy of the model on the training data
accuracy_train_l1 = accuracy_score(y_train_resampled, y_train_pred_l1)
print("Training Accuracy with L1 Regularization:", accuracy_train_l1)

# Calculate the F1 score for training data
train_f1_score_l1 = f1_score(y_train_resampled, y_train_pred_l1, average='weighted')
print("Training F1 Score with L1 Regularization:", train_f1_score_l1)

# Predict on the testing data
y_test_pred_l1 = log_reg_l1.predict(X_test)

# Calculate the accuracy of the model on the testing data
accuracy_test_l1 = accuracy_score(y_test, y_test_pred_l1)
print("Testing Accuracy with L1 Regularization:", accuracy_test_l1)

# Calculate the F1 score for testing data
test_f1_score_l1 = f1_score(y_test, y_test_pred_l1, average='weighted')
print("Testing F1 Score with L1 Regularization:", test_f1_score_l1)


# In[10]:


# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate SMOTE
smote = SMOTE(random_state=42)

# Apply SMOTE to generate synthetic samples for training data
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Instantiate logistic regression model with L2 regularization
log_reg_l2 = LogisticRegression(penalty='l2', solver='liblinear', max_iter=1000)

# Fit the model on the resampled training data
log_reg_l2.fit(X_train_resampled, y_train_resampled)

# Predict on the training data
y_train_pred_l2 = log_reg_l2.predict(X_train_resampled)

# Calculate the accuracy of the model on the training data
accuracy_train_l2 = accuracy_score(y_train_resampled, y_train_pred_l2)
print("Training Accuracy with L2 Regularization:", accuracy_train_l2)

# Calculate the F1 score for training data
train_f1_score_l2 = f1_score(y_train_resampled, y_train_pred_l2, average='weighted')
print("Training F1 Score with L2 Regularization:", train_f1_score_l2)

# Predict on the testing data
y_test_pred_l2 = log_reg_l2.predict(X_test)

# Calculate the accuracy of the model on the testing data
accuracy_test_l2 = accuracy_score(y_test, y_test_pred_l2)
print("Testing Accuracy with L2 Regularization:", accuracy_test_l2)

# Calculate the F1 score for testing data
test_f1_score_l2 = f1_score(y_test, y_test_pred_l2, average='weighted')
print("Testing F1 Score with L2 Regularization:", test_f1_score_l2)


# In[11]:


# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate SMOTE
smote = SMOTE(random_state=42)

# Apply SMOTE to generate synthetic samples for training data
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Instantiate logistic regression model with Elastic Net regularization
log_reg_elastic_net = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=1000)

# Fit the model on the resampled training data
log_reg_elastic_net.fit(X_train_resampled, y_train_resampled)

# Predict on the training data
y_train_pred_elastic_net = log_reg_elastic_net.predict(X_train_resampled)

# Calculate the accuracy of the model on the training data
accuracy_train_elastic_net = accuracy_score(y_train_resampled, y_train_pred_elastic_net)
print("Training Accuracy with Elastic Net Regularization:", accuracy_train_elastic_net)

# Calculate the F1 score for training data
train_f1_score_elastic_net = f1_score(y_train_resampled, y_train_pred_elastic_net, average='weighted')
print("Training F1 Score with Elastic Net Regularization:", train_f1_score_elastic_net)

# Predict on the testing data
y_test_pred_elastic_net = log_reg_elastic_net.predict(X_test)

# Calculate the accuracy of the model on the testing data
accuracy_test_elastic_net = accuracy_score(y_test, y_test_pred_elastic_net)
print("Testing Accuracy with Elastic Net Regularization:", accuracy_test_elastic_net)

# Calculate the F1 score for testing data
test_f1_score_elastic_net = f1_score(y_test, y_test_pred_elastic_net, average='weighted')
print("Testing F1 Score with Elastic Net Regularization:", test_f1_score_elastic_net)


# In[13]:


from sklearn.tree import DecisionTreeClassifier

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree classifier
tree_classifier = DecisionTreeClassifier()

# Fit the model on the training data
tree_classifier.fit(X_train, y_train)

# Predict on the training data
y_train_pred_tree = tree_classifier.predict(X_train)

# Calculate the accuracy of the model on the training data
accuracy_train_tree = accuracy_score(y_train, y_train_pred_tree)
print("Training Accuracy with Decision Trees:", accuracy_train_tree)

# Calculate the F1 score for training data
train_f1_score_tree = f1_score(y_train, y_train_pred_tree, average='weighted')
print("Training F1 Score with Decision Trees:", train_f1_score_tree)

# Predict on the testing data
y_test_pred_tree = tree_classifier.predict(X_test)

# Calculate the accuracy of the model on the testing data
accuracy_test_tree = accuracy_score(y_test, y_test_pred_tree)
print("Testing Accuracy with Decision Trees:", accuracy_test_tree)

# Calculate the F1 score for testing data
test_f1_score_tree = f1_score(y_test, y_test_pred_tree, average='weighted')
print("Testing F1 Score with Decision Trees:", test_f1_score_tree)


# In[14]:


from sklearn.ensemble import RandomForestClassifier

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier()

# Fit the model on the training data
rf_classifier.fit(X_train, y_train)

# Predict on the training data
y_train_pred_rf = rf_classifier.predict(X_train)

# Calculate the accuracy of the model on the training data
accuracy_train_rf = accuracy_score(y_train, y_train_pred_rf)
print("Training Accuracy with Random Forest:", accuracy_train_rf)

# Calculate the F1 score for training data
train_f1_score_rf = f1_score(y_train, y_train_pred_rf, average='weighted')
print("Training F1 Score with Random Forest:", train_f1_score_rf)

# Predict on the testing data
y_test_pred_rf = rf_classifier.predict(X_test)

# Calculate the accuracy of the model on the testing data
accuracy_test_rf = accuracy_score(y_test, y_test_pred_rf)
print("Testing Accuracy with Random Forest:", accuracy_test_rf)

# Calculate the F1 score for testing data
test_f1_score_rf = f1_score(y_test, y_test_pred_rf, average='weighted')
print("Testing F1 Score with Random Forest:", test_f1_score_rf)


# In[15]:


from sklearn.svm import SVC


# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the SVM classifier
svm_classifier = SVC()

# Fit the model on the training data
svm_classifier.fit(X_train, y_train)

# Predict on the training data
y_train_pred_svm = svm_classifier.predict(X_train)

# Calculate the accuracy of the model on the training data
accuracy_train_svm = accuracy_score(y_train, y_train_pred_svm)
print("Training Accuracy with SVM:", accuracy_train_svm)

# Calculate the F1 score for training data
train_f1_score_svm = f1_score(y_train, y_train_pred_svm, average='weighted')
print("Training F1 Score with SVM:", train_f1_score_svm)

# Predict on the testing data
y_test_pred_svm = svm_classifier.predict(X_test)

# Calculate the accuracy of the model on the testing data
accuracy_test_svm = accuracy_score(y_test, y_test_pred_svm)
print("Testing Accuracy with SVM:", accuracy_test_svm)

# Calculate the F1 score for testing data
test_f1_score_svm = f1_score(y_test, y_test_pred_svm, average='weighted')
print("Testing F1 Score with SVM:", test_f1_score_svm)


# In[17]:


from sklearn.neighbors import KNeighborsClassifier

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the KNN classifier
knn_classifier = KNeighborsClassifier()

# Fit the model on the training data
knn_classifier.fit(X_train, y_train)

# Predict on the training data
y_train_pred_knn = knn_classifier.predict(X_train)

# Calculate the accuracy of the model on the training data
accuracy_train_knn = accuracy_score(y_train, y_train_pred_knn)
print("Training Accuracy with KNN:", accuracy_train_knn)

# Calculate the F1 score for training data
train_f1_score_knn = f1_score(y_train, y_train_pred_knn, average='weighted')
print("Training F1 Score with KNN:", train_f1_score_knn)

# Predict on the testing data
y_test_pred_knn = knn_classifier.predict(X_test)

# Calculate the accuracy of the model on the testing data
accuracy_test_knn = accuracy_score(y_test, y_test_pred_knn)
print("Testing Accuracy with KNN:", accuracy_test_knn)

# Calculate the F1 score for testing data
test_f1_score_knn = f1_score(y_test, y_test_pred_knn, average='weighted')
print("Testing F1 Score with KNN:", test_f1_score_knn)


# In[ ]:





# In[ ]:




