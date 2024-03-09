#!/usr/bin/env python
# coding: utf-8

# # Telecom Churn Prediction - Week 9

# ## EDA, Data Visualization, Feature Scaling and Data Transformation

# The telecom industry faces a significant challenge in retaining customers, as increasing competition and evolving customer preferences contribute to a higher churn rate. **Churn** is defined as the percentage of subscribers who discontinue services within a given time period, poses a substantial threat to the revenue and sustainability of telecom service providers. To address this issue, there is a critical need for the development and implementation of an accurate and efficient churn prediction model.
# 
# The main aim of this project is to build Ensemble Machine learning Algorithms to predict the customer Churn.

# ## Understanding the dataset 
# 
# There are 7043 rows and 21 Features (Target Variable - Churn)
# 
#     1. customerID 
# 
#     2. gender - Whether the customer is a male or a female
# 
#     3. SeniorCitizen - Whether the customer is a senior citizen or not (1, 0)
# 
#     4. Partner - Whether the customer has a partner or not (Yes, No)
# 
#     5. Dependents - Whether the customer has dependents or not (Yes, No)
# 
#     6. tenure -Number of months the customer has stayed with the company
# 
#     7. PhoneService - Whether the customer has a phone service or not (Yes, No)
# 
#     8. MultipleLines - Whether the customer has multiple lines or not (Yes, No, No phone service)
# 
#     9. InternetService - Customer’s internet service provider (DSL, Fiber optic, No)
# 
#     10. OnlineSecurity - Whether the customer has online security or not (Yes, No, No internet service)
# 
#     11. OnlineBackup - Whether the customer has online backup or not (Yes, No, No internet service)
# 
#     12. DeviceProtection - Whether the customer has device protection or not (Yes, No, No internet service)
# 
#     13. TechSupport - Whether the customer has tech support or not (Yes, No, No internet service)
# 
#     14. StreamingTV - Whether the customer has streaming TV or not (Yes, No, No internet service)
# 
#     15. StreamingMovies - Whether the customer has streaming movies or not (Yes, No, No internet service)
# 
#     16. Contract - The contract term of the customer (Month-to-month, One year, Two year)
# 
#     17. PaperlessBilling - Whether the customer has paperless billing or not (Yes, No)
# 
#     18. PaymentMethod - The customer’s payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))
# 
#     19. MonthlyCharges - The amount charged to the customer monthly
# 
#     20. TotalCharges - The total amount charged to the customer
# 
#     21. Churn - Whether the customer churned or not (Yes or No)

# ### Importing necessary libraries

# In[1]:


## supressing warnings
import warnings 
warnings.filterwarnings('ignore')


# In[2]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns


# ### importing the dataset

# In[3]:


df = pd.read_csv('telecom.csv')


# In[4]:


# To display maximum rows and columns

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# In[5]:


# To display top 5 rows
df.head()


# In[6]:


# To display bottom 5 rows 
df.tail()


# In[7]:


# To check the dimension of the dataframe
df.shape


# In[8]:


# statistical info of the dataset 

df.describe()


# In[9]:


# lets check the datatype of each column 

df.info()


# In[10]:


# To check the duplicates in the data set

df.duplicated().sum()


# In[11]:


# To check the null values

df.isna().sum()


# ### Churn - Target Variable

# In[12]:


# Count of Churn - Yes & No

df['Churn'].value_counts()


# In[120]:


a = sns.countplot(x = 'Churn', data = df)
a.bar_label(a.containers[0])
plt.title("Bar Plot of Churn")
plt.xlabel("Churn")
plt.ylabel("Count")
#plt.savefig("Churn.png")
plt.show()


# ### From the above bar plot, it is clear that the target variable is imbalance
# 
# ### To assess the precision of the model, initially, we will construct machine learning algorithms using an imbalanced dataset and evaluate their accuracies. Subsequently, we will develop ML models with a balanced dataset for comparison 

# In[ ]:


# plot of Churn

a = df['Churn'].value_counts().plot(kind = 'bar')
a.bar_label(a.containers[0])
plt.title("Bar Plot of Churn")
plt.ylabel("Count")
plt.xlabel("Churn")
plt.show()


# ##### The datatype of TotalCharges is object, we shall convert this into int

# In[14]:


#The varaible was imported as a string we need to convert it to float
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'],errors='coerce')


# In[15]:


# The total charges are convereted into float
df['TotalCharges'].info()


# In[16]:


## Drop customerID 

df = df.drop(columns='customerID')


# In[17]:


df_categorical = df.select_dtypes(include='object')
df_categorical.head()


# In[18]:


df_continuous = df.select_dtypes(include='number')
df_continuous.head()


# #### from the data, SeniorCitizen has values 1 or 0

# In[19]:


## 'SeniorCitizen' feature has values 1 or 0
df['SeniorCitizen'].value_counts()


# In[20]:


x = sns.countplot(df['SeniorCitizen'])
x.bar_label(x.containers[0])
plt.show()


# In[21]:


df_continuous = df_continuous.drop(columns='SeniorCitizen')
df_continuous


# ### Categorical Data

# In[26]:


df_categorical.info()


# In[33]:


for i in df_categorical:
    b = sns.countplot(x = df[i], data = df)
    b.bar_label(b.containers[0])
    plt.xticks(rotation=45)
    plt.show()


# In[34]:


for i in df_categorical:
    b = sns.countplot(x = df[i], hue = df['Churn'], data = df)
    b.bar_label(b.containers[0])
    b.bar_label(b.containers[1])
    #plt.xticks(rotation=45)
    #plt.savefig(f'{i}')
    plt.show()


# In[35]:


#plt.figure(figsize=(8,6))
z = sns.countplot(x = 'PaymentMethod', hue = df['Churn'], data = df)
z.bar_label(z.containers[0])
z.bar_label(z.containers[1])
plt.xticks(rotation=20)
#plt.savefig('PaymentMethod.png')


# ### Continuous data

# In[36]:


for i in df_continuous:
    sns.boxplot(df[i])
    plt.savefig(f'{i}')
    plt.show()


# ### From the above boxplots No outliers are found 

# In[37]:


for i in df_continuous:
    sns.boxplot(x=df['Churn'], y=df[i])
    plt.show()


# ## histogram

# In[38]:


for i in df_continuous:
    sns.histplot(df_continuous[i])
    plt.show()


# ### pairsplot

# In[39]:


sns.pairplot(df)


# ### corelation plot

# In[45]:


sns.heatmap(df.corr(numeric_only=True), annot=True)


# In[46]:


df.info()


# #### From the above info we can see that there are many categorical columns and we shall to convert the categorical columns into numeric columns in order to build the algorithms.

# ### converting binary variable - Yes and No to 1 and 0

# In[47]:


var = ['Partner','Dependents','PhoneService','PaperlessBilling','Churn']

# mapping function
def mapping(n):
    return n.map({'Yes':1,'No':0})

df[var] = df[var].apply(mapping)
    


# ### Label encoding for gender column

# In[49]:


# import label encoder
from sklearn import preprocessing


# In[50]:


# object creation - label_encoder
label_encoder = preprocessing.LabelEncoder()


# In[51]:


# encode labels in gender column
df['gender'] = label_encoder.fit_transform(df['gender'])


# In[52]:


# to check the unique values in gender column
df['gender'].unique()


# ### For cagetorical columns with multiple levels - create dummies

# In[53]:


## create dummy variables

dummy = pd.get_dummies(df[['MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport',
                           'StreamingTV','StreamingMovies','Contract','PaymentMethod',]], drop_first=True)


# ### concat with the original dataframe

# In[54]:


df = pd.concat([df,dummy],axis =1)


# In[55]:


df.head(1)


# ### Dropping the repeated variables 

# In[58]:


df = df.drop(['MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport',
                           'StreamingTV','StreamingMovies','Contract','PaymentMethod'],axis=1)


# In[59]:


df.head()


# ### checking for missing values

# In[60]:


df.isnull().sum()


# ##### It means that 11/7043 = 0.001561834 i.e 0.1%, best is to remove these observations from the analysis

# In[61]:


# Removing NaN TotalCharges rows
df = df[~np.isnan(df['TotalCharges'])]


# ##### again check for missing values

# In[62]:


df.isna().sum()


# ##### Now we dont have missing values

# ## Splitting the data into training and testing set

# In[63]:


from sklearn.model_selection import train_test_split


# In[64]:


# Putting feature variable to X
X = df.drop(['Churn'], axis=1)

X.head()


# In[65]:


# response variable to y
y = df['Churn']

y.head()


# In[66]:


# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=300)


# ### Feature scaling and Data Transformation

# In[67]:


from sklearn.preprocessing import MinMaxScaler


# In[68]:


scaler = MinMaxScaler()

X_train[['tenure','MonthlyCharges','TotalCharges']] = scaler.fit_transform(X_train[['tenure','MonthlyCharges','TotalCharges']])

X_train.head()


# In[69]:


X_test.head()


# ### Transforming to test data

# In[70]:


X_test[['tenure','MonthlyCharges','TotalCharges']] = scaler.transform(X_test[['tenure','MonthlyCharges','TotalCharges']])

X_test.head()


# ## Now the Data is ready for Model Building

# ## Logistic Regression

# In[71]:


from sklearn.linear_model import LogisticRegression


# In[72]:


lr = LogisticRegression()


# In[73]:


# Training data is used for model building
lr.fit(X_train, y_train)


# In[74]:


# Testing data is used for prediction
y_pred_logreg = lr.predict(X_test)


# In[75]:


from sklearn.metrics import accuracy_score


# In[76]:


accuracy_score(y_test, y_pred_logreg)


# In[77]:


# Libraries for Validation of models
from sklearn.metrics import confusion_matrix


# In[78]:


logistic_confusion_matrix = confusion_matrix(y_test, y_pred_logreg)
logistic_confusion_matrix


# In[79]:


from sklearn.metrics import roc_curve, roc_auc_score


# In[80]:


# Function For Logistic Regression Create Summary For Logistic Regression

def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', lw=2,linestyle='--')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle=':')
    plt.xlabel('False Positive Rate(1-specificity)')
    plt.ylabel('True Positive Rate (sensitivity)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

def get_summary(y_test, y_pred_logreg):
    # Confusion Matrix
    conf_mat = confusion_matrix(y_test, y_pred_logreg)
    TP = conf_mat[0,0:1]
    FP = conf_mat[0,1:2]
    FN = conf_mat[1,0:1]
    TN = conf_mat[1,1:2]
    
    accuracy = (TP+TN)/((FN+FP)+(TP+TN))
    sensitivity = TP/(TP+FN)
    specificity = TN/(TN+FP)
    precision = TP/(TP+FP)
    recall =  TP / (TP + FN)
    fScore = (2 * recall * precision) / (recall + precision)
    auc = roc_auc_score(y_test, y_pred_logreg)

    print("Confusion Matrix:\n",conf_mat)
    print("Accuracy:",accuracy)
    print("Sensitivity :",sensitivity)
    print("Specificity :",specificity)
    print("Precision:",precision)
    print("Recall:",recall)
    print("F-score:",fScore)
    print("AUC:",auc)
    print("ROC curve:")
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_logreg)
    plot_roc_curve(fpr, tpr)


# In[81]:


get_summary(y_test, y_pred_logreg)


# ## Support Vector Machine (SVM)

# In[82]:


from sklearn.svm import SVC


# In[83]:


svc = SVC()


# In[84]:


svc.fit(X_train, y_train)


# In[85]:


y_pred_svc = svc.predict(X_test)


# In[86]:


accuracy_score(y_test, y_pred_svc)


# In[87]:


SVM_confusion_matrix = confusion_matrix(y_test, y_pred_svc)
SVM_confusion_matrix


# In[88]:


get_summary(y_test, y_pred_svc)


# ## Naive bayes Classification

# In[89]:


from sklearn.naive_bayes import GaussianNB


# In[90]:


gnb = GaussianNB()


# In[91]:


gnb.fit(X_train, y_train)


# In[92]:


y_pred_gnb = gnb.predict(X_test)


# In[93]:


accuracy_score(y_test,y_pred_gnb)


# In[94]:


gnb_confusion_matrix = confusion_matrix(y_test, y_pred_gnb)
gnb_confusion_matrix


# In[95]:


get_summary(y_test, y_pred_gnb)


# ## K - Nearest Neighbour

# In[96]:


from sklearn.neighbors import KNeighborsClassifier


# In[97]:


knn = KNeighborsClassifier()


# In[98]:


knn.fit(X_train, y_train)


# In[102]:


y_pred_knn = knn.predict(X_test.values)


# In[103]:


accuracy_score(y_test, y_pred_knn)


# In[104]:


knn_confusion_matrix = confusion_matrix(y_test, y_pred_knn)
knn_confusion_matrix


# In[105]:


get_summary(y_test, y_pred_knn)


# ## Decision Tree

# In[106]:


from sklearn.tree import DecisionTreeClassifier


# In[107]:


dtree = DecisionTreeClassifier()


# In[108]:


dtree.fit(X_train, y_train)


# In[109]:


y_pred_dtree = dtree.predict(X_test)


# In[110]:


accuracy_score(y_test, y_pred_dtree)


# In[111]:


dtree_confusion_matrix = confusion_matrix(y_test, y_pred_dtree)
dtree_confusion_matrix


# In[112]:


get_summary(y_test, y_pred_dtree)


# ## Random Forest

# In[113]:


from sklearn.ensemble import RandomForestClassifier


# In[114]:


rfc = RandomForestClassifier()


# In[115]:


rfc.fit(X_train, y_train)


# In[116]:


y_pred_rfc = rfc.predict(X_test)


# In[117]:


accuracy_score(y_test, y_pred_rfc)


# In[118]:


RandomForest_confusion_matrix = confusion_matrix(y_test, y_pred_rfc)
RandomForest_confusion_matrix


# In[119]:


get_summary(y_test, y_pred_rfc)


# ## AdaBoost Algorithm

# In[121]:


# importing Ada Boost classifier

from sklearn.ensemble import AdaBoostClassifier


# In[122]:


# creating instance
ada = AdaBoostClassifier()


# In[123]:


ada.fit(X_train, y_train)


# In[124]:


y_pred_ada = ada.predict(X_test)


# In[125]:


accuracy_score(y_test, y_pred_ada)


# In[126]:


AdaBoost_confusion_matrix = confusion_matrix(y_test, y_pred_ada)
AdaBoost_confusion_matrix


# In[127]:


get_summary(y_test, y_pred_ada)


# ## Hyperparameter tuning
# base_estimator: The model to the ensemble, the default is a decision tree.

# n_estimators: Number of models to be built.

# learning_rate: shrinks the contribution of each classifier by this value.

# random_state: The random number seed, so that the same random numbers generated every time.
# In[128]:


ada_clf = AdaBoostClassifier(random_state=100,base_estimator=RandomForestClassifier(random_state=101),
                            n_estimators=100, learning_rate=0.01)


# In[129]:


ada_clf.fit(X_train,y_train)


# In[130]:


y_pred_ada_clf = ada_clf.predict(X_test)


# In[131]:


accuracy_score(y_test, y_pred_ada_clf)


# In[132]:


ada_clf_confusion_matrix = confusion_matrix(y_test, y_pred_ada_clf)
ada_clf_confusion_matrix


# In[133]:


get_summary(y_test, y_pred_ada_clf)


# ## Gradient Boosting

# In[134]:


from sklearn.ensemble import GradientBoostingClassifier


# In[135]:


# creating instance
gb = GradientBoostingClassifier()


# In[137]:


gb.fit(X_train,y_train)


# In[138]:


y_pred_gb = gb.predict(X_test)


# In[139]:


accuracy_score(y_test, y_pred_gb)


# In[141]:


gb_confusion_matrix = confusion_matrix(y_test, y_pred_gb)
gb_confusion_matrix


# In[142]:


get_summary(y_test, y_pred_ada_clf)


# ## XGBoost 

# In[146]:


# pip install xgboost


# In[147]:


from xgboost import XGBClassifier


# In[148]:


# creating instance
xgb = XGBClassifier()


# In[149]:


# fit the model 

xgb.fit(X_train,y_train)


# In[150]:


# predict 
y_pred_xgb = xgb.predict(X_test)


# In[151]:


# accuracy score
accuracy_score(y_test, y_pred_xgb)


# In[152]:


xgb_confusion_matrix = confusion_matrix(y_test, y_pred_xgb)
xgb_confusion_matrix


# In[153]:


get_summary(y_test, y_pred_xgb)


# # Future work

# ## The above script is for imbalace data
# 
# ## Now we can balance using SMOTE the data and check for accuracy
# 
# 
# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




