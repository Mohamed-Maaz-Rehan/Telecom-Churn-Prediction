# Importing necessary libraries 

import numpy as np
import pandas as pd 

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn.neighbors import KNeighborsClassifier


# supress warnings

import warnings
warnings.filterwarnings('ignore')


# Reading csv file
df = pd.read_csv('Telecom Churn Prediction.csv')

# Displaying first 5 rows
df.head()


# Dropping the customerID column
df.drop(['customerID'], axis=1, inplace=True)

# Displaying first 5 rows
df.head()

# Changing the categorical columns to numeric
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})
df['MultipleLines'] = df['MultipleLines'].map({'No': 0, 'Yes': 1, 'No phone service': 2})
df['Contract'] = df['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
df['InternetService'] = df['InternetService'].map({'No': 0, 'DSL': 1, 'Fiber optic': 2})
df['PaymentMethod'] = df['PaymentMethod'].map({'Electronic check': 0, 'Mailed check': 1, 'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3})
df['TotalCharges']=df['TotalCharges'].replace({' ': 0})
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])

for col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']:
    df[col] = df[col].map({'No': 0, 'Yes': 1})
for col in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']:
    df[col] = df[col].map({'No': 0, 'Yes': 1, 'No internet service': 2})

# Displaying first 5 rows
df.head()


# # Splitting data in train and test

# removes the target column from the dataset, leaving only the features
X = df.drop('Churn', axis=1)  
y = df['Churn']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Print the shapes of the resulting datasets
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)


# # Logistic Regression Model

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target variable


# Create a logistic regression model
model = LogisticRegression(max_iter=200)

# Train the model with the training data
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")


# # Decision Tree

# Initialize the Decision Tree model
clf = DecisionTreeClassifier()

# Train the model with the training data
clf.fit(X_train, y_train)

# Make predictions on the test data
predictions = clf.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")


# # Random Forest 

# Initialize the Random Forest model
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)  # 100 trees in the forest

# Train the model with the training data
rf_clf.fit(X_train, y_train)

# Make predictions on the test data
predictions = rf_clf.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")


# # Support Vector Machine (SVM) model

# Initialize the SVM model
svm_model = SVC()

# Train the model with the training data
svm_model.fit(X_train, y_train)

# Make predictions on the test data
predictions = svm_model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy * 100:.2f}%")


# # k-Nearest Neighbors (KNN) model

# # Initialize the KNN model with k=3
# k = 3
# knn_model = KNeighborsClassifier(n_neighbors=k)

# # Train the model with the training data
# knn_model.fit(X_train, y_train)

# # Make predictions on the test data
# predictions = knn_model.predict(X_test)

# # Evaluate the model's performance
# accuracy = accuracy_score(y_test, predictions)
# print(f"Accuracy: {accuracy * 100:.2f}%")



