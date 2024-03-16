#!/usr/bin/env python
# coding: utf-8

# In[37]:


## supress warnings

import warnings
warnings.filterwarnings('ignore')


# In[38]:


import numpy as np
import pandas as pd 

import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler


# In[39]:


df = pd.read_csv('telecom.csv')


# In[40]:


df.drop(columns=['customerID'], inplace=True)


# In[43]:


cat_cols = list(df.select_dtypes(include=['object']).columns)


# In[47]:


def map_to_01(value):
    return 1 if value else 0

# Specify the column to exclude
exclude_column = 'Churn'

# Apply the map_to_01 function to all columns except the exclude_column
columns_to_map = df.columns[df.columns != exclude_column]
df_encoded_01 = df.copy()
df_encoded_01[columns_to_map] = df_encoded_01[columns_to_map].applymap(map_to_01)

print(df_encoded_01.head())


# In[50]:


df_encoded_01['Churn'] = df_encoded_01['Churn'].map({'Yes': 1, 'No': 0})


# In[51]:


from sklearn.model_selection import train_test_split

X = df_encoded_01.drop(columns='Churn')  # Features
y = df_encoded_01['Churn']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set shape (X_train, y_train):", X_train.shape, y_train.shape)
print("Testing set shape (X_test, y_test):", X_test.shape, y_test.shape)


# In[52]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# Assuming X and y are your feature matrix and target vector
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the logistic regression model
logistic_regression = LogisticRegression()

# Fit the model on the training data
logistic_regression.fit(X_train, y_train)

# Predict on the training data
y_train_pred = logistic_regression.predict(X_train)

# Predict on the testing data
y_test_pred = logistic_regression.predict(X_test)

# Calculate accuracy for training and testing sets
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Calculate F1 score for training and testing sets
train_f1 = f1_score(y_train, y_train_pred)
test_f1 = f1_score(y_test, y_test_pred)

print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)
print("Training F1 Score:", train_f1)
print("Testing F1 Score:", test_f1)


# In[57]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score

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

# Calculate the F1 score for training data
train_f1_score = f1_score(y_train, y_train_pred)
print("Training F1 Score:", train_f1_score)

# Calculate the F1 score for testing data
test_f1_score = f1_score(y_test, y_test_pred)
print("Testing F1 Score:", test_f1_score)




# In[58]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

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

# Calculate the F1 score for training data
train_f1_score = f1_score(y_train, y_train_pred)
print("Training F1 Score:", train_f1_score)

# Calculate the F1 score for testing data
test_f1_score = f1_score(y_test, y_test_pred)
print("Testing F1 Score:", test_f1_score)


# In[60]:


from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

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

# Calculate the F1 score for training data
train_f1_score = f1_score(y_train, y_train_pred)
print("Training F1 Score:", train_f1_score)

# Calculate the F1 score for testing data
test_f1_score = f1_score(y_test, y_test_pred)
print("Testing F1 Score:", test_f1_score)


# In[61]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

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

# Calculate the F1 score for training data
train_f1_score = f1_score(y_train, y_train_pred)
print("Training F1 Score:", train_f1_score)

# Calculate the F1 score for testing data
test_f1_score = f1_score(y_test, y_test_pred)
print("Testing F1 Score:", test_f1_score)


# In[ ]:





# # Regularization

# In[62]:


# Initialize the logistic regression model with regularization (L2)
log_reg_l2 = LogisticRegression(C=1.0, penalty='l2', max_iter=1000)

# Fit the model on the training data
log_reg_l2.fit(X_train, y_train)

# Predict on the training data
y_train_pred_l2 = log_reg_l2.predict(X_train)

# Predict on the testing data
y_test_pred_l2 = log_reg_l2.predict(X_test)

# Calculate the accuracy of the model on the training data
train_accuracy_l2 = accuracy_score(y_train, y_train_pred_l2)
print("Training Accuracy with Regularization:", train_accuracy_l2)

# Calculate the accuracy of the model on the testing data
test_accuracy_l2 = accuracy_score(y_test, y_test_pred_l2)
print("Testing Accuracy with Regularization:", test_accuracy_l2)

# Calculate the F1 score for training data
train_f1_score_l2 = f1_score(y_train, y_train_pred_l2)
print("Training F1 Score with Regularization:", train_f1_score_l2)

# Calculate the F1 score for testing data
test_f1_score_l2 = f1_score(y_test, y_test_pred_l2)
print("Testing F1 Score with Regularization:", test_f1_score_l2)



# In[65]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# Initialize the logistic regression model with regularization (L1)
log_reg_l1 = LogisticRegression(C=1.0, penalty='l1', max_iter=1000, solver='liblinear')

# Fit the model on the training data
log_reg_l1.fit(X_train, y_train)

# Predict on the training data
y_train_pred_l1 = log_reg_l1.predict(X_train)

# Predict on the testing data
y_test_pred_l1 = log_reg_l1.predict(X_test)

# Calculate the accuracy of the model on the training data
train_accuracy_l1 = accuracy_score(y_train, y_train_pred_l1)
print("Training Accuracy with L1 Regularization:", train_accuracy_l1)

# Calculate the accuracy of the model on the testing data
test_accuracy_l1 = accuracy_score(y_test, y_test_pred_l1)
print("Testing Accuracy with L1 Regularization:", test_accuracy_l1)

# Calculate the F1 score for training data
train_f1_score_l1 = f1_score(y_train, y_train_pred_l1)
print("Training F1 Score with L1 Regularization:", train_f1_score_l1)

# Calculate the F1 score for testing data
test_f1_score_l1 = f1_score(y_test, y_test_pred_l1)
print("Testing F1 Score with L1 Regularization:", test_f1_score_l1)



# In[66]:


# Initialize the logistic regression model with Elastic Net regularization
log_reg_elastic_net = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=1000)

# Fit the model on the training data
log_reg_elastic_net.fit(X_train, y_train)

# Predict on the training data
y_train_pred_elastic_net = log_reg_elastic_net.predict(X_train)

# Predict on the testing data
y_test_pred_elastic_net = log_reg_elastic_net.predict(X_test)

# Calculate the accuracy of the model on the training data
train_accuracy_elastic_net = accuracy_score(y_train, y_train_pred_elastic_net)
print("Training Accuracy with Elastic Net Regularization:", train_accuracy_elastic_net)

# Calculate the accuracy of the model on the testing data
test_accuracy_elastic_net = accuracy_score(y_test, y_test_pred_elastic_net)
print("Testing Accuracy with Elastic Net Regularization:", test_accuracy_elastic_net)

# Calculate the F1 score for training data
train_f1_score_elastic_net = f1_score(y_train, y_train_pred_elastic_net)
print("Training F1 Score with Elastic Net Regularization:", train_f1_score_elastic_net)

# Calculate the F1 score for testing data
test_f1_score_elastic_net = f1_score(y_test, y_test_pred_elastic_net)
print("Testing F1 Score with Elastic Net Regularization:", test_f1_score_elastic_net)


# In[67]:


# Initialize the Decision Tree classifier
tree_classifier = DecisionTreeClassifier()

# Fit the model on the training data
tree_classifier.fit(X_train, y_train)

# Predict on the training data
y_train_pred_tree = tree_classifier.predict(X_train)

# Predict on the testing data
y_test_pred_tree = tree_classifier.predict(X_test)

# Calculate the accuracy of the model on the training data
train_accuracy_tree = accuracy_score(y_train, y_train_pred_tree)
print("Training Accuracy with Decision Trees:", train_accuracy_tree)

# Calculate the accuracy of the model on the testing data
test_accuracy_tree = accuracy_score(y_test, y_test_pred_tree)
print("Testing Accuracy with Decision Trees:", test_accuracy_tree)

# Calculate the F1 score for training data
train_f1_score_tree = f1_score(y_train, y_train_pred_tree)
print("Training F1 Score with Decision Trees:", train_f1_score_tree)

# Calculate the F1 score for testing data
test_f1_score_tree = f1_score(y_test, y_test_pred_tree)
print("Testing F1 Score with Decision Trees:", test_f1_score_tree)


# In[68]:


# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier()

# Fit the model on the training data
rf_classifier.fit(X_train, y_train)

# Predict on the training data
y_train_pred_rf = rf_classifier.predict(X_train)

# Predict on the testing data
y_test_pred_rf = rf_classifier.predict(X_test)

# Calculate the accuracy of the model on the training data
train_accuracy_rf = accuracy_score(y_train, y_train_pred_rf)
print("Training Accuracy with Random Forest:", train_accuracy_rf)

# Calculate the accuracy of the model on the testing data
test_accuracy_rf = accuracy_score(y_test, y_test_pred_rf)
print("Testing Accuracy with Random Forest:", test_accuracy_rf)

# Calculate the F1 score for training data
train_f1_score_rf = f1_score(y_train, y_train_pred_rf)
print("Training F1 Score with Random Forest:", train_f1_score_rf)

# Calculate the F1 score for testing data
test_f1_score_rf = f1_score(y_test, y_test_pred_rf)
print("Testing F1 Score with Random Forest:", test_f1_score_rf)



# In[71]:


from sklearn.svm import SVC

# Initialize the SVM classifier
svm_classifier = SVC()

# Fit the model on the training data
svm_classifier.fit(X_train, y_train)

# Predict on the training data
y_train_pred_svm = svm_classifier.predict(X_train)

# Predict on the testing data
y_test_pred_svm = svm_classifier.predict(X_test)

# Calculate the accuracy of the model on the training data
train_accuracy_svm = accuracy_score(y_train, y_train_pred_svm)
print("Training Accuracy with SVM:", train_accuracy_svm)

# Calculate the accuracy of the model on the testing data
test_accuracy_svm = accuracy_score(y_test, y_test_pred_svm)
print("Testing Accuracy with SVM:", test_accuracy_svm)

# Calculate the F1 score for training data
train_f1_score_svm = f1_score(y_train, y_train_pred_svm)
print("Training F1 Score with SVM:", train_f1_score_svm)

# Calculate the F1 score for testing data
test_f1_score_svm = f1_score(y_test, y_test_pred_svm)
print("Testing F1 Score with SVM:", test_f1_score_svm)


# In[70]:


from sklearn.neighbors import KNeighborsClassifier

# Initialize the KNN classifier
knn_classifier = KNeighborsClassifier()

# Fit the model on the training data
knn_classifier.fit(X_train, y_train)

# Predict on the training data
y_train_pred_knn = knn_classifier.predict(X_train)

# Predict on the testing data
y_test_pred_knn = knn_classifier.predict(X_test)

# Calculate the accuracy of the model on the training data
train_accuracy_knn = accuracy_score(y_train, y_train_pred_knn)
print("Training Accuracy with KNN:", train_accuracy_knn)

# Calculate the accuracy of the model on the testing data
test_accuracy_knn = accuracy_score(y_test, y_test_pred_knn)
print("Testing Accuracy with KNN:", test_accuracy_knn)

# Calculate the F1 score for training data
train_f1_score_knn = f1_score(y_train, y_train_pred_knn)
print("Training F1 Score with KNN:", train_f1_score_knn)

# Calculate the F1 score for testing data
test_f1_score_knn = f1_score(y_test, y_test_pred_knn)
print("Testing F1 Score with KNN:", test_f1_score_knn)


# In[ ]:





# In[ ]:




