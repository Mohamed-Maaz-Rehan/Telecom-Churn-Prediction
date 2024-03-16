#!/usr/bin/env python
# coding: utf-8

# # Telecom Churn Prediction - Week 10

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


# In[13]:


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

# In[14]:


# plot of Churn

a = df['Churn'].value_counts().plot(kind = 'bar')
a.bar_label(a.containers[0])
plt.title("Bar Plot of Churn")
plt.ylabel("Count")
plt.xlabel("Churn")
plt.show()


# ##### The datatype of TotalCharges is object, we shall convert this into int

# In[15]:


#The varaible was imported as a string we need to convert it to float
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'],errors='coerce')


# In[16]:


# The total charges are convereted into float
df['TotalCharges'].info()


# In[17]:


## Drop customerID 

df = df.drop(columns='customerID')


# In[18]:


df_categorical = df.select_dtypes(include='object')
df_categorical.head()


# In[19]:


df_continuous = df.select_dtypes(include='number')
df_continuous.head()


# #### from the data, SeniorCitizen has values 1 or 0

# In[20]:


## 'SeniorCitizen' feature has values 1 or 0
df['SeniorCitizen'].value_counts()


# In[177]:


x = sns.countplot(x=df['SeniorCitizen'], data=df)
x.bar_label(x.containers[0])
plt.show()


# In[22]:


df_continuous = df_continuous.drop(columns='SeniorCitizen')
df_continuous


# ### Categorical Data

# In[23]:


df_categorical.info()


# In[24]:


for i in df_categorical:
    b = sns.countplot(x = df[i], data = df)
    b.bar_label(b.containers[0])
    plt.xticks(rotation=45)
    plt.show()


# In[25]:


for i in df_categorical:
    b = sns.countplot(x = df[i], hue = df['Churn'], data = df)
    b.bar_label(b.containers[0])
    b.bar_label(b.containers[1])
    #plt.xticks(rotation=45)
    #plt.savefig(f'{i}')
    plt.show()


# In[26]:


#plt.figure(figsize=(8,6))
z = sns.countplot(x = 'PaymentMethod', hue = df['Churn'], data = df)
z.bar_label(z.containers[0])
z.bar_label(z.containers[1])
plt.xticks(rotation=20)
#plt.savefig('PaymentMethod.png')


# ### Continuous data

# In[27]:


for i in df_continuous:
    sns.boxplot(df[i])
    plt.savefig(f'{i}')
    plt.show()


# ### From the above boxplots No outliers are found 

# In[28]:


for i in df_continuous:
    sns.boxplot(x=df['Churn'], y=df[i])
    plt.show()


# ## histogram

# In[29]:


for i in df_continuous:
    sns.histplot(df_continuous[i])
    plt.show()


# ### pairsplot

# In[30]:


sns.pairplot(df)


# ### corelation plot

# In[31]:


sns.heatmap(df.corr(numeric_only=True), annot=True)


# In[32]:


df.info()


# #### From the above info we can see that there are many categorical columns and we shall to convert the categorical columns into numeric columns in order to build the algorithms.

# ### converting binary variable - Yes and No to 1 and 0

# In[33]:


var = ['Partner','Dependents','PhoneService','PaperlessBilling','Churn']

# mapping function
def mapping(n):
    return n.map({'Yes':1,'No':0})

df[var] = df[var].apply(mapping)
    


# ### Label encoding for gender column

# In[34]:


# import label encoder
from sklearn import preprocessing


# In[35]:


# object creation - label_encoder
label_encoder = preprocessing.LabelEncoder()


# In[36]:


# encode labels in gender column
df['gender'] = label_encoder.fit_transform(df['gender'])


# In[37]:


# to check the unique values in gender column
df['gender'].unique()


# ### For cagetorical columns with multiple levels - create dummies

# In[38]:


## create dummy variables

dummy = pd.get_dummies(df[['MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport',
                           'StreamingTV','StreamingMovies','Contract','PaymentMethod',]], drop_first=True)


# ### concat with the original dataframe

# In[39]:


df = pd.concat([df,dummy],axis =1)


# In[40]:


df.head(1)


# ### Dropping the repeated variables 

# In[41]:


df = df.drop(['MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport',
                           'StreamingTV','StreamingMovies','Contract','PaymentMethod'],axis=1)


# In[42]:


df.head()


# ### checking for missing values

# In[43]:


df.isnull().sum()


# ##### It means that 11/7043 = 0.001561834 i.e 0.1%, best is to remove these observations from the analysis

# In[44]:


# Removing NaN TotalCharges rows
df = df[~np.isnan(df['TotalCharges'])]


# ##### again check for missing values

# In[45]:


df.isna().sum()


# ##### Now we dont have missing values

# ## Splitting the data into training and testing set

# In[46]:


from sklearn.model_selection import train_test_split


# In[47]:


# Putting feature variable to X
X = df.drop(['Churn'], axis=1)

X.head()


# In[48]:


# response variable to y
y = df['Churn']

y.head()


# In[49]:


# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=300)


# ### Feature scaling and Data Transformation

# In[50]:


from sklearn.preprocessing import MinMaxScaler


# In[51]:


scaler = MinMaxScaler()

X_train[['tenure','MonthlyCharges','TotalCharges']] = scaler.fit_transform(X_train[['tenure','MonthlyCharges','TotalCharges']])

X_train.head()


# In[52]:


X_test.head()


# ### Transforming to test data

# In[53]:


X_test[['tenure','MonthlyCharges','TotalCharges']] = scaler.transform(X_test[['tenure','MonthlyCharges','TotalCharges']])

X_test.head()


# ## Now the Data is ready for Model Building

# ## Logistic Regression

# In[54]:


from sklearn.linear_model import LogisticRegression


# In[55]:


lr = LogisticRegression()


# In[56]:


# Training data is used for model building
lr.fit(X_train, y_train)


# In[57]:


# Testing data is used for prediction
y_pred_logreg = lr.predict(X_test)


# In[58]:


from sklearn.metrics import accuracy_score


# In[59]:


accuracy_score(y_test, y_pred_logreg)


# In[60]:


# Libraries for Validation of models
from sklearn.metrics import confusion_matrix


# In[61]:


logistic_confusion_matrix = confusion_matrix(y_test, y_pred_logreg)
logistic_confusion_matrix


# In[62]:


from sklearn.metrics import roc_curve, roc_auc_score


# In[63]:


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


# In[64]:


get_summary(y_test, y_pred_logreg)


# ## Support Vector Machine (SVM)

# In[65]:


from sklearn.svm import SVC


# In[66]:


svc = SVC()


# In[67]:


svc.fit(X_train, y_train)


# In[68]:


y_pred_svc = svc.predict(X_test)


# In[69]:


accuracy_score(y_test, y_pred_svc)


# In[70]:


SVM_confusion_matrix = confusion_matrix(y_test, y_pred_svc)
SVM_confusion_matrix


# In[71]:


get_summary(y_test, y_pred_svc)


# ## Naive bayes Classification

# In[72]:


from sklearn.naive_bayes import GaussianNB


# In[73]:


gnb = GaussianNB()


# In[74]:


gnb.fit(X_train, y_train)


# In[75]:


y_pred_gnb = gnb.predict(X_test)


# In[76]:


accuracy_score(y_test,y_pred_gnb)


# In[77]:


gnb_confusion_matrix = confusion_matrix(y_test, y_pred_gnb)
gnb_confusion_matrix


# In[78]:


get_summary(y_test, y_pred_gnb)


# ## K - Nearest Neighbour

# In[79]:


from sklearn.neighbors import KNeighborsClassifier


# In[80]:


knn = KNeighborsClassifier()


# In[81]:


knn.fit(X_train, y_train)


# In[82]:


y_pred_knn = knn.predict(X_test.values)


# In[83]:


accuracy_score(y_test, y_pred_knn)


# In[84]:


knn_confusion_matrix = confusion_matrix(y_test, y_pred_knn)
knn_confusion_matrix


# In[85]:


get_summary(y_test, y_pred_knn)


# ## Decision Tree

# In[86]:


from sklearn.tree import DecisionTreeClassifier


# In[87]:


dtree = DecisionTreeClassifier()


# In[88]:


dtree.fit(X_train, y_train)


# In[89]:


y_pred_dtree = dtree.predict(X_test)


# In[90]:


accuracy_score(y_test, y_pred_dtree)


# In[91]:


dtree_confusion_matrix = confusion_matrix(y_test, y_pred_dtree)
dtree_confusion_matrix


# In[92]:


get_summary(y_test, y_pred_dtree)


# ## Random Forest

# In[93]:


from sklearn.ensemble import RandomForestClassifier


# In[94]:


rfc = RandomForestClassifier()


# In[95]:


rfc.fit(X_train, y_train)


# In[96]:


y_pred_rfc = rfc.predict(X_test)


# In[97]:


accuracy_score(y_test, y_pred_rfc)


# In[98]:


RandomForest_confusion_matrix = confusion_matrix(y_test, y_pred_rfc)
RandomForest_confusion_matrix


# In[99]:


get_summary(y_test, y_pred_rfc)


# ## AdaBoost Algorithm

# In[100]:


# importing Ada Boost classifier

from sklearn.ensemble import AdaBoostClassifier


# In[101]:


# creating instance
ada = AdaBoostClassifier()


# In[102]:


ada.fit(X_train, y_train)


# In[103]:


y_pred_ada = ada.predict(X_test)


# In[104]:


accuracy_score(y_test, y_pred_ada)


# In[105]:


AdaBoost_confusion_matrix = confusion_matrix(y_test, y_pred_ada)
AdaBoost_confusion_matrix


# In[106]:


get_summary(y_test, y_pred_ada)


# ## Hyperparameter tuning
# base_estimator: The model to the ensemble, the default is a decision tree.

# n_estimators: Number of models to be built.

# learning_rate: shrinks the contribution of each classifier by this value.

# random_state: The random number seed, so that the same random numbers generated every time.
# In[107]:


ada_clf = AdaBoostClassifier(random_state=100,base_estimator=RandomForestClassifier(random_state=101),
                            n_estimators=100, learning_rate=0.01)


# In[108]:


ada_clf.fit(X_train,y_train)


# In[109]:


y_pred_ada_clf = ada_clf.predict(X_test)


# In[110]:


accuracy_score(y_test, y_pred_ada_clf)


# In[111]:


ada_clf_confusion_matrix = confusion_matrix(y_test, y_pred_ada_clf)
ada_clf_confusion_matrix


# In[112]:


get_summary(y_test, y_pred_ada_clf)


# ## Gradient Boosting

# In[113]:


from sklearn.ensemble import GradientBoostingClassifier


# In[114]:


# creating instance
gb = GradientBoostingClassifier()


# In[115]:


gb.fit(X_train,y_train)


# In[116]:


y_pred_gb = gb.predict(X_test)


# In[117]:


accuracy_score(y_test, y_pred_gb)


# In[118]:


gb_confusion_matrix = confusion_matrix(y_test, y_pred_gb)
gb_confusion_matrix


# In[119]:


get_summary(y_test, y_pred_ada_clf)


# ## XGBoost 

# In[120]:


# pip install xgboost


# In[121]:


from xgboost import XGBClassifier


# In[122]:


# creating instance
xgb = XGBClassifier()


# In[123]:


# fit the model 

xgb.fit(X_train,y_train)


# In[124]:


# predict 
y_pred_xgb = xgb.predict(X_test)


# In[125]:


# accuracy score
accuracy_score(y_test, y_pred_xgb)


# In[126]:


xgb_confusion_matrix = confusion_matrix(y_test, y_pred_xgb)
xgb_confusion_matrix


# In[127]:


get_summary(y_test, y_pred_xgb)


# ## Balancing the data set using SMOTE

# In[129]:


## conda install -c conda-forge imbalanced-learn


# In[130]:


## pip install -U imbalanced-learn


# In[131]:


from imblearn.over_sampling import SMOTE


# In[141]:


## before SMOTE
y_train.value_counts().plot(kind = 'bar')


# ### oversampling the train dataset using SMOTE

# In[138]:


# object creation
smt = SMOTE()


# In[139]:


X_train_sm, y_train_sm = smt.fit_resample(X_train,y_train)


# In[142]:


## After SMOTE
y_train_sm.value_counts().plot(kind = 'bar')


# ## After balancing the data set, Now lets build our model 

# ## Logistic Regression

# In[145]:


from sklearn.linear_model import LogisticRegression


# In[165]:


lr = LogisticRegression()

lr.fit(X_train_sm, y_train_sm)
y_pred_logreg = lr.predict(X_test)

train_accuracy = accuracy_score(y_train_sm, lr.predict(X_train_sm))
print("Train Accuracy Logistic Regression", train_accuracy)
test_accuracy = accuracy_score(y_test, y_pred_logreg)
print("Test Accuracy Logistic Regression", test_accuracy)


# ## Support Vector Machine

# In[171]:


svc = SVC()

svc.fit(X_train_sm, y_train_sm)
y_pred = svc.predict(X_test)

train_accuracy = accuracy_score(y_train_sm, svc.predict(X_train_sm))
print("Train Accuracy SVM", train_accuracy)
test_accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy SVM", test_accuracy)


# ## Naive Bayes Classification

# In[172]:


gnb = GaussianNB()

gnb.fit(X_train_sm, y_train_sm)
y_pred = gnb.predict(X_test)

train_accuracy = accuracy_score(y_train_sm, gnb.predict(X_train_sm))
print("Train Accuracy Naive Bayes", train_accuracy)
test_accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy Naive Bayes", test_accuracy)


# ## KNN Algorithm

# In[173]:


knn = KNeighborsClassifier()

knn.fit(X_train_sm, y_train_sm)
y_pred = knn.predict(X_test.values)

train_accuracy = accuracy_score(y_train_sm, knn.predict(X_train_sm.values))
print("Train Accuracy KNN Algorithm", train_accuracy)
test_accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy KNN Algorithm", test_accuracy)


# ## Decision Trees

# In[174]:


dtree = DecisionTreeClassifier()

dtree.fit(X_train_sm, y_train_sm)
y_pred = dtree.predict(X_test)

train_accuracy = accuracy_score(y_train_sm, dtree.predict(X_train_sm))
print("Train Accuracy Decision Trees", train_accuracy)
test_accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy Decision Trees", test_accuracy)


# ## Random Forest

# In[176]:


rfc = RandomForestClassifier()

rfc.fit(X_train_sm, y_train_sm)
y_pred = rfc.predict(X_test)

train_accuracy = accuracy_score(y_train_sm, rfc.predict(X_train_sm))
print("Train Accuracy Random Forest", train_accuracy)
test_accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy Random Forest", test_accuracy)


# # Future work

# ### After balancing the data, Model seems to be overfitted. Now, we shall reduce some features and build a model

# ## Recursive Feature Elimination Technique

# ### we shall try with other balancing techniques like Adasyn and hybridization techniques such as SMOTE+Tomek and SMOTE+ENN

# In[ ]:




