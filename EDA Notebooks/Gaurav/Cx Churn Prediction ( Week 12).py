#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')


# In[65]:


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import recall_score, confusion_matrix, precision_score, f1_score, accuracy_score, classification_report


# In[66]:


#loading data
df = pd.read_csv('telecom.csv')


# Each row represents a customer, each column contains customer’s attributes described on the column Metadata.

# In[67]:


df.head()


# **The data set includes information about:**
# * **Customers who left within the last month** – the column is called Churn
# 
# * **Services that each customer has signed up for** – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
# 
# * **Customer account information** - how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges
# 
# * **Demographic info about customers** – gender, age range, and if they have partners and dependents

# In[68]:


df.shape


# In[69]:


df.info()


# In[70]:


df.columns.values


# In[71]:


df.dtypes


# 
# * The target the we will use to guide the exploration is **Churn**

# # Visualizing the Data

# In[72]:


# Visualize missing values as a matrix
msno.matrix(df);


# # Data Manipulation

# In[73]:


#droping out the customerID attribute as it is not related to the analysis


# In[74]:


df = df.drop(['customerID'], axis = 1)
df.head()


# On deep analysis, we can find some indirect missingness in our data

# In[75]:


df['TotalCharges'] = pd.to_numeric(df.TotalCharges, errors='coerce')
df.isnull().sum()


# TotalCharges has 11 missing values. 

# In[76]:


df[np.isnan(df['TotalCharges'])]

 It can also be noted that the Tenure column is 0 for these entries even though the MonthlyCharges column is not empty.

checking for other 0 values in the tenure column.
# In[77]:


df[df['tenure'] == 0].index


# There are no additional missing values in the Tenure column. 
# deleting the rows with missing values in Tenure columns since there are only 11 rows and deleting them will not affect the data.

# In[78]:


df.drop(labels=df[df['tenure'] == 0].index, axis=0, inplace=True)
df[df['tenure'] == 0].index

 To solve the problem of missing values in TotalCharges column, fill it with the mean of TotalCharges values
# In[79]:


df.fillna(df["TotalCharges"].mean())


# In[80]:


df.isnull().sum()


# In[81]:


df["SeniorCitizen"]= df["SeniorCitizen"].map({0: "No", 1: "Yes"})
df.head()


# In[82]:


df["InternetService"].describe(include=['object', 'bool'])


# In[83]:


numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
df[numerical_cols].describe()


# # Splitting Data into testing and training

# In[84]:


def object_to_int(dataframe_series):
    if dataframe_series.dtype=='object':
        dataframe_series = LabelEncoder().fit_transform(dataframe_series)
    return dataframe_series


# In[85]:


df = df.apply(lambda x: object_to_int(x))
df.head()


# In[86]:


plt.figure(figsize=(14,7))
df.corr()['Churn'].sort_values(ascending = False)


# In[87]:


X = df.drop(columns = ['Churn'])
y = df['Churn'].values


# In[88]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.20, random_state = 40, stratify=y)


# In[89]:


def distplot(feature, frame, color='r'):
    plt.figure(figsize=(8,3))
    plt.title("Distribution for {}".format(feature))
    ax = sns.distplot(frame[feature])


# In[90]:


num_cols = ["tenure", 'MonthlyCharges', 'TotalCharges']
for feat in num_cols: distplot(feat, df)


# Since the numerical features are distributed over different value ranges, I will use standard scalar to scale them down to the same range.

# 
# Standardizing numeric attributes
# 

# In[91]:


df_std = pd.DataFrame(StandardScaler().fit_transform(df[num_cols].astype('float64')),
                       columns=num_cols)
for feat in numerical_cols: distplot(feat, df_std,)


# # Predictions 
# KNN

# In[92]:


# Divide the columns into 3 categories, one ofor standardisation, one for label encoding and one for one hot encoding

cat_cols_ohe =['PaymentMethod', 'Contract', 'InternetService'] # those that need one-hot encoding
cat_cols_le = list(set(X_train.columns)- set(num_cols) - set(cat_cols_ohe)) #those that need label encoding


# In[93]:


scaler= StandardScaler()

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])


# In[94]:


X = df.drop(columns = ['Churn'])
y = df['Churn'].values


# In[95]:


# Perform one-hot encoding for categorical variables
X_encoded = pd.get_dummies(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the KNN classifier
knn_model = KNeighborsClassifier(n_neighbors=5)

# Train the classifier
knn_model.fit(X_train_scaled, y_train)

# Make predictions on the test set
predicted_y = knn_model.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, predicted_y)
print("Accuracy:", accuracy)

# Generate classification report
print("Classification Report:")
print(classification_report(y_test, predicted_y))


# In[96]:


print(classification_report(y_test, predicted_y))


# <a id = "102" ></a>
# #### <b>SVC</b>
# <a id = "svc" ></a>

# In[97]:


svc_model = SVC(random_state = 1)
svc_model.fit(X_train,y_train)
predict_y = svc_model.predict(X_test)
accuracy_svc = svc_model.score(X_test,y_test)
print("SVM accuracy is :",accuracy_svc)


# In[98]:


print(classification_report(y_test, predict_y))


# <a id = "103" ></a>
# #### <b> Random Forest</b>
# <a id = "rf" ></a>

# In[36]:


model_rf = RandomForestClassifier(n_estimators=500, oob_score=True, n_jobs=-1,
                                  random_state=50, max_features='sqrt',
                                  max_leaf_nodes=30)

model_rf.fit(X_train, y_train)

# Make predictions
prediction_test = model_rf.predict(X_test)
print (metrics.accuracy_score(y_test, prediction_test))


# In[37]:


print(classification_report(y_test, prediction_test))


# In[38]:


plt.figure(figsize=(4,3))
sns.heatmap(confusion_matrix(y_test, prediction_test),
                annot=True,fmt = "d",linecolor="k",linewidths=3)
    
plt.title(" RANDOM FOREST CONFUSION MATRIX",fontsize=14)
plt.show()


# In[39]:


y_rfpred_prob = model_rf.predict_proba(X_test)[:,1]
fpr_rf, tpr_rf, thresholds = roc_curve(y_test, y_rfpred_prob)
plt.plot([0, 1], [0, 1], 'k--' )
plt.plot(fpr_rf, tpr_rf, label='Random Forest',color = "r")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest ROC Curve',fontsize=16)
plt.show();


# <a id = "104" ></a>
# #### <b>Logistic Regression</b>
# <a id = "lr" ></a>

# In[40]:


lr_model = LogisticRegression()
lr_model.fit(X_train,y_train)
accuracy_lr = lr_model.score(X_test,y_test)
print("Logistic Regression accuracy is :",accuracy_lr)


# In[41]:


lr_pred= lr_model.predict(X_test)
report = classification_report(y_test,lr_pred)
print(report)


# In[42]:


plt.figure(figsize=(4,3))
sns.heatmap(confusion_matrix(y_test, lr_pred),
                annot=True,fmt = "d",linecolor="k",linewidths=3)
    
plt.title("LOGISTIC REGRESSION CONFUSION MATRIX",fontsize=14)
plt.show()


# In[43]:


y_pred_prob = lr_model.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot([0, 1], [0, 1], 'k--' )
plt.plot(fpr, tpr, label='Logistic Regression',color = "r")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve',fontsize=16)
plt.show();


# <a id = "105" ></a>
# #### **Decision Tree Classifier**
# <a id = "dtc" ></a>

# In[44]:


dt_model = DecisionTreeClassifier()
dt_model.fit(X_train,y_train)
predictdt_y = dt_model.predict(X_test)
accuracy_dt = dt_model.score(X_test,y_test)
print("Decision Tree accuracy is :",accuracy_dt)


# Decision tree gives very low score.

# In[45]:


print(classification_report(y_test, predictdt_y))


# Relationship between multiple accuracies

# In[46]:


import matplotlib.pyplot as plt

# Model names
models = ['KNN', 'SVC', 'Random Forest', 'Logistic Regression', 'Decision Tree']

# Corresponding accuracies
accuracies = [0.704, 0.800, 0.793, 0.730, 0.704]

# Plotting the bar plot
plt.figure(figsize=(10, 6))
plt.bar(models, accuracies, color=['blue', 'orange', 'green', 'red', 'purple'])

# Adding labels and title
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Accuracy of Different Models')
plt.ylim(0, 1)  # Set the y-axis limit from 0 to 1
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

# Displaying the plot
plt.show()


# Model Comparison:
# KNN: Achieves 80% accuracy.
# SVC: Also achieves 80% accuracy.
# Random Forest: Performs equally well at 80% accuracy.
# Logistic Regression: Matches the others with 80% accuracy.
# Decision Tree: Again, 80% accuracy.

# In[47]:


df.info()


# Here we can see that we have two features which are not binary therefore converting it to binary

# In[48]:


# 'MonthlyCharges' and 'TotalCharges' from float to int

df['MonthlyCharges'] = df['MonthlyCharges'].astype(int)
df['TotalCharges'] = df['TotalCharges'].astype(int)


# In[49]:


df.info()


# In[50]:


df.isnull().sum()


# In[51]:


df['Churn'].value_counts()


# Here we can see that unique count of customers who will not churn is much higher compared to the customers who will churn.
# 
# This means there is imbalance in the class.

# oversampling the minority class using the Synthetic Minority Over-sampling Technique (SMOTE):

# In[52]:


from imblearn.over_sampling import SMOTE

# Define features and target variable
X = df.drop('Churn', axis=1)
y = df['Churn']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Apply SMOTE for oversampling
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Data preprocessing
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Define logistic regression model
logistic_regression = LogisticRegression(max_iter=1000)

# Train the logistic regression model
logistic_regression.fit(X_train_scaled, y_train_resampled)

# Make predictions on the test set
y_pred = logistic_regression.predict(X_test_scaled)

# Calculate precision
precision = precision_score(y_test, y_pred)

# Print precision score
print("Precision Score:", precision)

# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))


# Precision Score: The precision score for predicting churn (class 1) is 0.52, indicating that about 52% of the customers predicted to churn actually churned.
# 
# Confusion Matrix: We can see that there are 378 false positives (customers incorrectly predicted as churned) and 149 false negatives (customers incorrectly predicted as not churned).
# 
# Classification Report: The classification report provides a summary of various evaluation metrics such as precision, recall, and F1-score for each class (churned and not churned). It also includes support, which represents the number of occurrences of each class in the test set.

# In[53]:


from sklearn.utils.class_weight import compute_class_weight

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)

# Define logistic regression model with class weights
logistic_regression_weighted = LogisticRegression(class_weight={0: class_weights[0], 1: class_weights[1]}, max_iter=1000)

# Train the logistic regression model with adjusted class weights
logistic_regression_weighted.fit(X_train_scaled, y_train_resampled)


# It seems you've adjusted the class weights for the logistic regression model using the class_weight parameter. The values {0: 0.6809629219701162, 1: 1.8814984709480123} indicate the weights assigned to each class.
# 
# 
# Class 0 (customers who will not churn) has a weight of approximately 0.68.
# Class 1 (customers who will churn) has a weight of approximately 1.88.

# In[54]:


# Define logistic regression model with adjusted class weights
logistic_regression_weighted = LogisticRegression(class_weight={0: 0.6809629219701162, 1: 1.8814984709480123}, max_iter=1000)

# Train the logistic regression model with adjusted class weights
logistic_regression_weighted.fit(X_train_scaled, y_train_resampled)

# Make predictions on the test set
y_pred_weighted = logistic_regression_weighted.predict(X_test_scaled)

# Calculate precision
precision_weighted = precision_score(y_test, y_pred_weighted)

# Print precision score
print("Precision Score (with adjusted class weights):", precision_weighted)

# Print confusion matrix
print("Confusion Matrix (with adjusted class weights):")
print(confusion_matrix(y_test, y_pred_weighted))

# Print classification report
print("Classification Report (with adjusted class weights):")
print(classification_report(y_test, y_pred_weighted))


# defined a logistic regression model (logistic_regression_weighted) with the adjusted class weights.
# 
# trained the logistic regression model using the resampled training data (X_train_scaled, y_train_resampled).

# Precision Score (with adjusted class weights): The precision score is approximately 0.432, 
# indicating that around 43.2% of the customers predicted to churn actually churned. This is lower compared to the precision score without adjusted class weights.
# 
# Confusion Matrix (with adjusted class weights): We can see that there are 654 false positives (customers incorrectly predicted as churned) and 64 false negatives (customers incorrectly predicted as not churned).

# In[55]:


# SMOTE oversampling
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)


# In[56]:


# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_resampled, y_train_resampled, test_size=0.2, random_state=42)

# Create a logistic regression model
logistic_model = LogisticRegression()

# Fit the model to the training data
logistic_model.fit(X_train, y_train)

# Make predictions on the validation set
y_val_pred = logistic_model.predict(X_val)

# Evaluate the model
accuracy = accuracy_score(y_val, y_val_pred)
classification_report_str = classification_report(y_val, y_val_pred)

print(f"Validation Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report_str)


# Precision:
# Precision for class 0 (negative class) is 0.82, which means that when the model predicts class 0, it is correct 82% of the time.
# Precision for class 1 (positive class) is 0.79, indicating that when the model predicts class 1, it is correct 79% of the time.
# Recall (Sensitivity):
# Recall for class 0 is 0.78, meaning that the model correctly identifies 78% of the actual class 0 instances.
# Recall for class 1 is 0.83, indicating that the model captures 83% of the actual class 1 instances.
# F1-Score:
# The F1-score balances precision and recall. For class 0, it’s 0.80, and for class 1, it’s 0.81.
# Accuracy:
# The overall accuracy of the model on the validation set is 0.80 (80%).

# In[57]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

# Generate synthetic data (you can replace this with your own dataset)
X, y = make_blobs(n_samples=100, centers=2, random_state=42)

# Create an SVM classifier
svm_classifier = SVC(kernel='linear', C=1.0)

# Fit the model to the data
svm_classifier.fit(X, y)

# Plot the decision boundary
xfit = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
yfit = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
Xgrid, Ygrid = np.meshgrid(xfit, yfit)
Z = svm_classifier.decision_function(np.c_[Xgrid.ravel(), Ygrid.ravel()])
Z = Z.reshape(Xgrid.shape)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='autumn')
plt.contour(Xgrid, Ygrid, Z, colors='k', levels=[-1, 0, 1], alpha=0.5)
plt.xlabel("Featues")
plt.ylabel("Churn")
plt.title("SVM Classifier Decision Boundary")
plt.show()


# In[58]:


from sklearn.tree import DecisionTreeClassifier

# Define decision tree classifier
decision_tree = DecisionTreeClassifier(random_state=42)

# Train the decision tree classifier with resampled data
decision_tree.fit(X_train_resampled, y_train_resampled)

# Make predictions on the test set
y_pred = decision_tree.predict(X_test_scaled)

# Calculate precision
precision = precision_score(y_test, y_pred)

# Print precision score
print("Precision Score:", precision)

# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))


# Precision Score: The precision score is approximately 0.253, indicating that around 25.3% of the customers predicted to churn actually churned.
# 
# Confusion Matrix: We can see that there are 749 false positives (customers incorrectly predicted as churned) and 307 false negatives (customers incorrectly predicted as not churned).

# In[59]:


# Define Random Forest classifier
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the Random Forest classifier with resampled data
random_forest.fit(X_train_resampled, y_train_resampled)

# Make predictions on the test set
y_pred = random_forest.predict(X_test_scaled)  # Assuming X_test_scaled is already scaled

# Calculate precision
precision = precision_score(y_test, y_pred)

# Print precision score
print("Precision Score:", precision)

# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))


# Precision Score: The precision score is approximately 0.426, indicating that around 42.6% of the customers predicted to churn actually churned.
# 
# Confusion Matrix: We can see that there are 569 false positives (customers incorrectly predicted as churned) and 138 false negatives (customers incorrectly predicted as not churned).

# # Checking Feature Importance

# In[60]:


# Checking the feature importances of various features
# Sorting the importances by descending order (lowest importance at the bottom)
for score, name in sorted(zip(model_rf.feature_importances_, X_train.columns), reverse=True):
    print('Feature importance of', name, ':', score*100, '%')


# In[61]:


model_rf.feature_importances_*100


# In[62]:


# Plotting the feature importance of each feature
plt.figure(figsize=(12,7))
plt.bar(X_train.columns,model_rf.feature_importances_*100, color='orange')
plt.xlabel('Features', fontsize=14)
plt.ylabel('Importance', fontsize=14)
plt.xticks(rotation=90)
plt.title('Feature Importance of each feature', fontsize=16)


# # Hyperparameter Tuning
# 

# In[102]:


from sklearn.model_selection import GridSearchCV


# In[103]:


# Defining a parameter grid for hyperparameter tuning with different values to be tested for 'n_estimators', 'max_depth', and 'max_features' hyperparameters
param_grid = [{'n_estimators': [100, 200, 300], 'max_depth': [None,2,3,10,20], 'max_features': ['sqrt',2,4,8,16,'log2', None]}]


# In[104]:


# Creating a random forest classifier object 'temp_rf' with a random state of 0 and parallel processing enabled
temp_rf=RandomForestClassifier(random_state=0,n_jobs=-1)

# Creating a grid search object 'grid_search' using the 'GridSearchCV' function, with a random forest classifier as the estimator, a parameter grid, 'roc_auc' as the scoring metric, and 5-fold cross-validation with parallel processing
grid_search=GridSearchCV(estimator=temp_rf, param_grid=param_grid, scoring='roc_auc', cv=5, n_jobs=-1)


# In[105]:


# Performing grid search on the training data to find the best hyperparameters for the model
grid_search.fit(X_train,y_train)


# In[106]:


# Calculating the best RMSE score found by Grid Search 
grid_search.best_score_


# In[107]:


# Retrieving the best parameter values found by the grid search
grid_search.best_params_


# # XGBoost Hyperparameter Tuning

# In[108]:


from sklearn.model_selection import GridSearchCV

# Random Forest hyperparameter grid
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt']
}

# Initialize the Random Forest model
rf = RandomForestClassifier(random_state=42)

# Setup GridSearchCV
grid_search_rf = GridSearchCV(estimator=rf, param_grid=rf_param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)

# Fit GridSearchCV
grid_search_rf.fit(X_train, y_train)

# Best parameters and score
best_params_rf = grid_search_rf.best_params_
best_score_rf = grid_search_rf.best_score_

best_params_rf, best_score_rf


# # Best Parameters
# After running GridSearchCV, let's assume you found the best parameters for the Random Forest model to be:
# 
# n_estimators: 100
# max_depth: 10
# min_samples_split: 2
# min_samples_leaf: 2
# max_features: 'sqrt'

# # Evaluate Model Performance

# In[109]:


# Initialize the model with the best parameters
optimized_rf = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=2,
                                      min_samples_leaf=1, max_features='sqrt', random_state=42)

# Train the model
optimized_rf.fit(X_train, y_train)

# Make predictions
y_pred_optimized = optimized_rf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_optimized)
precision = precision_score(y_test, y_pred_optimized)
recall = recall_score(y_test, y_pred_optimized)
f1 = f1_score(y_test, y_pred_optimized)

print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")


# # Interpret the Model

# In[110]:


importances = optimized_rf.feature_importances_
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")
for f in range(X_train.shape[1]):
    print(f"{f + 1}. feature {X_train.columns[indices[f]]} ({importances[indices[f]]})")


# In[111]:


# Using the provided feature importance data to create a comparison graph

# Features and their importances
features = ["TotalCharges", "MonthlyCharges", "tenure", "Contract", "PaymentMethod", 
            "OnlineSecurity", "TechSupport", "InternetService", "gender", "OnlineBackup", 
            "PaperlessBilling", "Partner", "MultipleLines", "SeniorCitizen", "DeviceProtection", 
            "Dependents", "StreamingMovies", "StreamingTV", "PhoneService"]
importances = [0.18590472492273977, 0.1781053180579041, 0.1562856904959392, 0.08169260100242129, 
              0.05042443343117293, 0.049066527428709136, 0.041871754510007964, 0.02926031462316956, 
              0.0277477659533689, 0.026384681046870726, 0.02527880374588388, 0.023509568192995014, 
              0.02242241857672751, 0.02115112171461009, 0.020881870265634737, 0.019636890571001483, 
              0.01770687443240459, 0.017432630019875193, 0.0052360110085640284]

# Sort the features by importance
sorted_indices = np.argsort(importances)[::-1]
sorted_features = np.array(features)[sorted_indices]
sorted_importances = np.array(importances)[sorted_indices]

# Creating the plot
plt.figure(figsize=(12, 8))
plt.barh(sorted_features, sorted_importances, color='lightblue')
plt.xlabel('Importance')
plt.title('Feature Importances in Optimized Random Forest Model')
plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature on top
plt.show()


# Here we can see that there is a very huge gap in the feature importance between 'PhoneService' and 'TotalCharges'

# According to the current data , 'PhoneService' feature is likely unrelated in the prediction model

# # 10-Fold Cross Validation

# In[113]:


# Assuming X and y are your features and target variable
# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)

# Perform 10-fold cross-validation
scores = cross_val_score(rf_classifier, X, y, cv=10)

# Display the scores for each fold
print("Accuracy scores for each fold:")
print(scores)

# Calculate the average score across all folds
mean_score = scores.mean()
print(f"\nMean accuracy across 10 folds: {mean_score:.4f}")


# In[ ]:




