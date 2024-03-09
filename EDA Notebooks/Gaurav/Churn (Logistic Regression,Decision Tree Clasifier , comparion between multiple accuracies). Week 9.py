#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


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


# In[3]:


#loading data
df = pd.read_csv('telecom.csv')


# Each row represents a customer, each column contains customer’s attributes described on the column Metadata.

# In[4]:


df.head()


# **The data set includes information about:**
# * **Customers who left within the last month** – the column is called Churn
# 
# * **Services that each customer has signed up for** – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
# 
# * **Customer account information** - how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges
# 
# * **Demographic info about customers** – gender, age range, and if they have partners and dependents

# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.columns.values


# In[8]:


df.dtypes


# 
# * The target the we will use to guide the exploration is **Churn**

# # Visualizing the Data

# In[9]:


# Visualize missing values as a matrix
msno.matrix(df);


# # Data Manipulation

# In[10]:


df = df.drop(['customerID'], axis = 1)
df.head()


# On deep analysis, we can find some indirect missingness in our data

# In[11]:


df['TotalCharges'] = pd.to_numeric(df.TotalCharges, errors='coerce')
df.isnull().sum()


# TotalCharges has 11 missing values. 

# In[48]:


df[np.isnan(df['TotalCharges'])]

 It can also be noted that the Tenure column is 0 for these entries even though the MonthlyCharges column is not empty.

checking for other 0 values in the tenure column.
# In[46]:


df[df['tenure'] == 0].index


# There are no additional missing values in the Tenure column. 
# deleting the rows with missing values in Tenure columns since there are only 11 rows and deleting them will not affect the data.

# In[47]:


df.drop(labels=df[df['tenure'] == 0].index, axis=0, inplace=True)
df[df['tenure'] == 0].index

 To solve the problem of missing values in TotalCharges column, fill it with the mean of TotalCharges values
# In[15]:


df.fillna(df["TotalCharges"].mean())


# In[16]:


df.isnull().sum()


# In[17]:


df["SeniorCitizen"]= df["SeniorCitizen"].map({0: "No", 1: "Yes"})
df.head()


# In[18]:


df["InternetService"].describe(include=['object', 'bool'])


# In[19]:


numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
df[numerical_cols].describe()


# # Splitting Data into testing and training

# In[20]:


def object_to_int(dataframe_series):
    if dataframe_series.dtype=='object':
        dataframe_series = LabelEncoder().fit_transform(dataframe_series)
    return dataframe_series


# In[21]:


df = df.apply(lambda x: object_to_int(x))
df.head()


# In[22]:


plt.figure(figsize=(14,7))
df.corr()['Churn'].sort_values(ascending = False)


# In[23]:


X = df.drop(columns = ['Churn'])
y = df['Churn'].values


# In[24]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.30, random_state = 40, stratify=y)


# In[25]:


def distplot(feature, frame, color='r'):
    plt.figure(figsize=(8,3))
    plt.title("Distribution for {}".format(feature))
    ax = sns.distplot(frame[feature])


# In[26]:


num_cols = ["tenure", 'MonthlyCharges', 'TotalCharges']
for feat in num_cols: distplot(feat, df)


# Since the numerical features are distributed over different value ranges, I will use standard scalar to scale them down to the same range.

# <a id = "111" ></a>
# #### **Standardizing numeric attributes**
# <a id = "Standardizing" ></a>

# In[27]:


df_std = pd.DataFrame(StandardScaler().fit_transform(df[num_cols].astype('float64')),
                       columns=num_cols)
for feat in numerical_cols: distplot(feat, df_std,)


# # Predictions 
# KNN

# In[49]:


# Divide the columns into 3 categories, one ofor standardisation, one for label encoding and one for one hot encoding

cat_cols_ohe =['PaymentMethod', 'Contract', 'InternetService'] # those that need one-hot encoding
cat_cols_le = list(set(X_train.columns)- set(num_cols) - set(cat_cols_ohe)) #those that need label encoding


# In[50]:


scaler= StandardScaler()

X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])


# In[51]:


X = df.drop(columns = ['Churn'])
y = df['Churn'].values


# In[52]:


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


# In[53]:


print(classification_report(y_test, predicted_y))


# <a id = "102" ></a>
# #### <b>SVC</b>
# <a id = "svc" ></a>

# In[54]:


svc_model = SVC(random_state = 1)
svc_model.fit(X_train,y_train)
predict_y = svc_model.predict(X_test)
accuracy_svc = svc_model.score(X_test,y_test)
print("SVM accuracy is :",accuracy_svc)


# In[55]:


print(classification_report(y_test, predict_y))


# <a id = "103" ></a>
# #### <b> Random Forest</b>
# <a id = "rf" ></a>

# In[56]:


model_rf = RandomForestClassifier(n_estimators=500, oob_score=True, n_jobs=-1,
                                  random_state=50, max_features='sqrt',
                                  max_leaf_nodes=30)

model_rf.fit(X_train, y_train)

# Make predictions
prediction_test = model_rf.predict(X_test)
print (metrics.accuracy_score(y_test, prediction_test))


# In[57]:


print(classification_report(y_test, prediction_test))


# In[58]:


plt.figure(figsize=(4,3))
sns.heatmap(confusion_matrix(y_test, prediction_test),
                annot=True,fmt = "d",linecolor="k",linewidths=3)
    
plt.title(" RANDOM FOREST CONFUSION MATRIX",fontsize=14)
plt.show()


# In[38]:


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

# In[39]:


lr_model = LogisticRegression()
lr_model.fit(X_train,y_train)
accuracy_lr = lr_model.score(X_test,y_test)
print("Logistic Regression accuracy is :",accuracy_lr)


# In[40]:


lr_pred= lr_model.predict(X_test)
report = classification_report(y_test,lr_pred)
print(report)


# In[41]:


plt.figure(figsize=(4,3))
sns.heatmap(confusion_matrix(y_test, lr_pred),
                annot=True,fmt = "d",linecolor="k",linewidths=3)
    
plt.title("LOGISTIC REGRESSION CONFUSION MATRIX",fontsize=14)
plt.show()


# In[42]:


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

# In[43]:


dt_model = DecisionTreeClassifier()
dt_model.fit(X_train,y_train)
predictdt_y = dt_model.predict(X_test)
accuracy_dt = dt_model.score(X_test,y_test)
print("Decision Tree accuracy is :",accuracy_dt)


# Decision tree gives very low score.

# In[44]:


print(classification_report(y_test, predictdt_y))


# Relationship between multiple accuracies

# In[45]:


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


# In[ ]:




