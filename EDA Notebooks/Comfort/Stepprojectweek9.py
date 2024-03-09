#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC as SVM
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, BaggingClassifier


# In[5]:


df = pd.read_csv("telecom.csv")


# In[6]:


df.drop(columns = ['tenure', 'customerID'],inplace = True)


# In[7]:


df = df.replace({
    'PhoneService': {'Yes': 1, 'No': 0},
    'Partner': {'Yes': 1, 'No': 0},
    'gender': {'Female': 1, 'Male': 0},
    'Dependents': {'Yes': 1, 'No': 0}, 
    'MultipleLines': {'Yes': 1, 'No': 0, 'No phone service': 2},
    'InternetService': {'DSL': 1, 'Fiber optic':2, 'No':0},
    'OnlineSecurity': {'Yes': 1, 'No': 0, 'No internet service': 2},
    'OnlineBackup': {'Yes': 1, 'No': 0, 'No internet service': 2},
    'DeviceProtection': {'Yes': 1, 'No': 0, 'No internet service': 2},
    'TechSupport': {'Yes': 1, 'No': 0, 'No internet service': 2},
    'StreamingTV': {'Yes': 1, 'No': 0, 'No internet service': 2},
    'StreamingMovies': {'Yes': 1, 'No': 0, 'No internet service': 2},
    'Contract':{'Month-to-month':1, 'One year': 0, 'Two year':2},
    'PaperlessBilling':{'Yes': 1, 'No': 0},
    'PaymentMethod':{'Electronic check':1, 'Mailed check':0, 'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3},
    'Churn':{'Yes':1, 'No':0}
    
})


# In[8]:


df['PhoneService'] = df['PhoneService'].astype(int)
df['Partner'] = df['Partner'].astype(int)
df['gender'] = df['gender'].astype(int)
df['Dependents'] = df['Dependents'].astype(int)
df['MultipleLines'] = df['MultipleLines'].astype(int)
df['OnlineSecurity'] = df['OnlineSecurity'].astype(int)
df['OnlineBackup'] = df['OnlineBackup'].astype(int)
df['DeviceProtection'] = df['DeviceProtection'].astype(int)
df['TechSupport'] = df['TechSupport'].astype(int)
df['StreamingTV'] = df['StreamingTV'].astype(int)
df['StreamingMovies'] = df['StreamingMovies'].astype(int)
df['Contract'] = df['Contract'].astype(int)
df['PaperlessBilling'] = df['PaperlessBilling'].astype(int)
df['PaymentMethod'] = df['PaymentMethod'].astype(int)
df['Churn'] = df['Churn'].astype(int)


# In[ ]:


df.info()


# In[ ]:


df.isnull().sum()


# In[9]:


df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
df['TotalCharges'] = df['TotalCharges'].astype(float)
df.fillna(df["TotalCharges"].mean(),inplace=True)


# In[10]:


X_col = df.columns[df.columns != 'Churn'].tolist()
y_col = 'Churn'
X = df[X_col]
y = df[y_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
knn = KNeighborsClassifier(n_neighbors=5)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[17]:


models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Support Vector Machine': SVM(),
    'K-Nearest Neighbors': KNeighborsClassifier()
}

# Train and evaluate models
results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy

# Display results
for name, accuracy in results.items():
    print(f'{name}: Accuracy = {accuracy:.2f}')


# In[ ]:


#Using Esembling techniques to improve model accuracy


# In[18]:


ada = AdaBoostClassifier(n_estimators=100, random_state=42)
ada.fit(X_train, y_train)


gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

bagging = BaggingClassifier(n_estimators=100, random_state=42)
bagging.fit(X_train, y_train)

adaboost_pred = ada.predict(X_test)


gb_pred = gb.predict(X_test)
rf_pred = rf.predict(X_test)

bagging_pred = bagging.predict(X_test)
# Accuracy scores
ada_accuracy = accuracy_score(y_test, adaboost_pred)
gb_accuracy = accuracy_score(y_test, gb_pred)
rf_accuracy = accuracy_score(y_test, rf_pred)
bagging_accuracy = accuracy_score(y_test, bagging_pred)

print("AdaBoostClassifier Accuracy:", ada_accuracy)
print("GradientBoostingClassifier Accuracy:", gb_accuracy)
print("RandomForestClassifier Accuracy:", rf_accuracy)
print("BaggingClassifier Accuracy:", bagging_accuracy)


# In[ ]:




