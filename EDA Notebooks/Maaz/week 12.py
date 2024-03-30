#!/usr/bin/env python
# coding: utf-8

# # Telecom Churn Prediction - Week 12

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


# In[21]:


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


# In[54]:


X_train.replace({False: 0, True: 1}, inplace=True)


# In[55]:


X_test.replace({False: 0, True: 1}, inplace=True)


# In[56]:


X_train.head()


# In[57]:


X_test.head()


# ## Now the Data is ready for Model Building

# ## Logistic Regression

# In[58]:


from sklearn.linear_model import LogisticRegression


# In[59]:


lr = LogisticRegression()


# In[60]:


# Training data is used for model building
lr.fit(X_train, y_train)


# In[61]:


# Testing data is used for prediction
y_pred_logreg = lr.predict(X_test)


# In[62]:


from sklearn.metrics import accuracy_score


# In[63]:


accuracy_score(y_test, y_pred_logreg)


# In[64]:


# Libraries for Validation of models
from sklearn.metrics import confusion_matrix


# In[65]:


logistic_confusion_matrix = confusion_matrix(y_test, y_pred_logreg)
logistic_confusion_matrix


# In[66]:


from sklearn.metrics import roc_curve, roc_auc_score


# In[67]:


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


# In[68]:


get_summary(y_test, y_pred_logreg)


# ## Support Vector Machine (SVM)

# In[69]:


from sklearn.svm import SVC


# In[70]:


svc = SVC()


# In[71]:


svc.fit(X_train, y_train)


# In[72]:


y_pred_svc = svc.predict(X_test)


# In[73]:


accuracy_score(y_test, y_pred_svc)


# In[74]:


SVM_confusion_matrix = confusion_matrix(y_test, y_pred_svc)
SVM_confusion_matrix


# In[75]:


get_summary(y_test, y_pred_svc)


# ## Naive bayes Classification

# In[76]:


from sklearn.naive_bayes import GaussianNB


# In[77]:


gnb = GaussianNB()


# In[78]:


gnb.fit(X_train, y_train)


# In[79]:


y_pred_gnb = gnb.predict(X_test)


# In[80]:


accuracy_score(y_test,y_pred_gnb)


# In[81]:


gnb_confusion_matrix = confusion_matrix(y_test, y_pred_gnb)
gnb_confusion_matrix


# In[82]:


get_summary(y_test, y_pred_gnb)


# ## K - Nearest Neighbour

# In[83]:


from sklearn.neighbors import KNeighborsClassifier


# In[84]:


knn = KNeighborsClassifier()


# In[85]:


knn.fit(X_train, y_train)


# In[86]:


y_pred_knn = knn.predict(X_test.values)


# In[87]:


accuracy_score(y_test, y_pred_knn)


# In[88]:


knn_confusion_matrix = confusion_matrix(y_test, y_pred_knn)
knn_confusion_matrix


# In[89]:


get_summary(y_test, y_pred_knn)


# ## Decision Tree

# In[90]:


from sklearn.tree import DecisionTreeClassifier


# In[91]:


dtree = DecisionTreeClassifier()


# In[92]:


dtree.fit(X_train, y_train)


# In[93]:


y_pred_dtree = dtree.predict(X_test)


# In[94]:


accuracy_score(y_test, y_pred_dtree)


# In[95]:


dtree_confusion_matrix = confusion_matrix(y_test, y_pred_dtree)
dtree_confusion_matrix


# In[96]:


get_summary(y_test, y_pred_dtree)


# ## Random Forest

# In[97]:


from sklearn.ensemble import RandomForestClassifier


# In[98]:


rfc = RandomForestClassifier()


# In[99]:


rfc.fit(X_train, y_train)


# In[100]:


y_pred_rfc = rfc.predict(X_test)


# In[101]:


accuracy_score(y_test, y_pred_rfc)


# In[102]:


RandomForest_confusion_matrix = confusion_matrix(y_test, y_pred_rfc)
RandomForest_confusion_matrix


# In[103]:


get_summary(y_test, y_pred_rfc)


# ## Accuracy Comparison

# In[104]:


LR = accuracy_score(y_test, y_pred_logreg)
SVM = accuracy_score(y_test, y_pred_svc)
NB = accuracy_score(y_test,y_pred_gnb)
KNN = accuracy_score(y_test, y_pred_knn)
DT = accuracy_score(y_test, y_pred_dtree) 
RF = accuracy_score(y_test, y_pred_rfc)


# ## Bar chart to compare the accuracy of all classification models - Unbalanced data

# In[105]:


algorithms = ['LR','SVM','NB','KNN','DT', 'RF']
accuracies = [LR,SVM,NB,KNN,DT,RF]


# In[106]:


c = ['red', 'yellow', 'pink', 'blue', 'orange','green']
plt.bar(algorithms, accuracies,color=c)
plt.xlabel('Algorithm')
plt.ylabel('Accuracy (%)')
plt.title('Comparison of Classifier Accuracy - Unbalanced Data')
plt.ylim([0, 1])  # Set the y-axis limits between 0 and 1 or 0 and 100. 
plt.xticks(rotation=45)

for i in range(len(algorithms)):
    plt.text(i, accuracies[i],f"{accuracies[i]*100:.2f}%", ha='center',va= 'bottom')
plt.show()


# ## AdaBoost Algorithm

# In[107]:


# importing Ada Boost classifier

from sklearn.ensemble import AdaBoostClassifier


# In[108]:


# creating instance
ada = AdaBoostClassifier()


# In[109]:


ada.fit(X_train, y_train)


# In[110]:


y_pred_ada = ada.predict(X_test)


# In[111]:


accuracy_score(y_test, y_pred_ada)


# In[112]:


AdaBoost_confusion_matrix = confusion_matrix(y_test, y_pred_ada)
AdaBoost_confusion_matrix


# In[113]:


get_summary(y_test, y_pred_ada)


# ## Hyperparameter tuning
# base_estimator: The model to the ensemble, the default is a decision tree.

# n_estimators: Number of models to be built.

# learning_rate: shrinks the contribution of each classifier by this value.

# random_state: The random number seed, so that the same random numbers generated every time.
# In[114]:


ada_clf = AdaBoostClassifier(random_state=100,base_estimator=RandomForestClassifier(random_state=101),
                            n_estimators=100, learning_rate=0.01)


# In[115]:


ada_clf.fit(X_train,y_train)


# In[116]:


y_pred_ada_clf = ada_clf.predict(X_test)


# In[117]:


accuracy_score(y_test, y_pred_ada_clf)


# In[118]:


ada_clf_confusion_matrix = confusion_matrix(y_test, y_pred_ada_clf)
ada_clf_confusion_matrix


# In[119]:


get_summary(y_test, y_pred_ada_clf)


# ## Gradient Boosting

# In[120]:


from sklearn.ensemble import GradientBoostingClassifier


# In[121]:


# creating instance
gb = GradientBoostingClassifier()


# In[122]:


gb.fit(X_train,y_train)


# In[123]:


y_pred_gb = gb.predict(X_test)


# In[124]:


accuracy_score(y_test, y_pred_gb)


# In[125]:


gb_confusion_matrix = confusion_matrix(y_test, y_pred_gb)
gb_confusion_matrix


# In[126]:


get_summary(y_test, y_pred_ada_clf)


# ## XGBoost 

# In[127]:


# pip install xgboost


# In[128]:


from xgboost import XGBClassifier


# In[129]:


# creating instance
xgb = XGBClassifier()


# In[130]:


# fit the model 

xgb.fit(X_train,y_train)


# In[131]:


# predict 
y_pred_xgb = xgb.predict(X_test)


# In[132]:


# accuracy score
accuracy_score(y_test, y_pred_xgb)


# In[133]:


xgb_confusion_matrix = confusion_matrix(y_test, y_pred_xgb)
xgb_confusion_matrix


# In[134]:


get_summary(y_test, y_pred_xgb)


# ## Accuracy comparison for boosing techniques

# In[135]:


AdaBoost = accuracy_score(y_test, y_pred_ada)
GradientBoost = accuracy_score(y_test, y_pred_gb)
XGBoost = accuracy_score(y_test, y_pred_xgb)


# In[136]:


algorithms = ['AdaBoost','GradientBoost','XGBoost']
accuracies = [AdaBoost,GradientBoost,XGBoost]


# In[137]:


c = ['blue','orange','green']
plt.bar(algorithms, accuracies,color=c)
plt.xlabel('Algorithm')
plt.ylabel('Accuracy (%)')
plt.title('Comparison of Boosting Algorithm')
plt.ylim([0, 1])  # Set the y-axis limits between 0 and 1 or 0 and 100. 
plt.xticks(rotation=45)

for i in range(len(algorithms)):
    plt.text(i, accuracies[i],f"{accuracies[i]*100:.2f}%", ha='center',va= 'bottom')
plt.show()


# ## Balancing the data set using SMOTE

# In[138]:


## conda install -c conda-forge imbalanced-learn


# In[139]:


## pip install -U imbalanced-learn


# In[140]:


from imblearn.over_sampling import SMOTE


# In[141]:


## before SMOTE
y_train.value_counts().plot(kind = 'bar')


# ### oversampling the train dataset using SMOTE

# In[142]:


# object creation
smt = SMOTE()


# In[143]:


X_train_sm, y_train_sm = smt.fit_resample(X_train,y_train)


# In[144]:


## After SMOTE
y_train_sm.value_counts().plot(kind = 'bar')


# ## After balancing the data set, Now lets build our model 

# ## Logistic Regression

# In[145]:


from sklearn.linear_model import LogisticRegression


# In[146]:


lr = LogisticRegression()

lr.fit(X_train_sm, y_train_sm)
y_pred_logreg = lr.predict(X_test)

train_accuracy = accuracy_score(y_train_sm, lr.predict(X_train_sm))
print("Train Accuracy Logistic Regression", train_accuracy)
test_accuracy = accuracy_score(y_test, y_pred_logreg)
print("Test Accuracy Logistic Regression", test_accuracy)


# ## Support Vector Machine

# In[147]:


svc = SVC()

svc.fit(X_train_sm, y_train_sm)
y_pred = svc.predict(X_test)

train_accuracy = accuracy_score(y_train_sm, svc.predict(X_train_sm))
print("Train Accuracy SVM", train_accuracy)
test_accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy SVM", test_accuracy)


# ## Naive Bayes Classification

# In[148]:


gnb = GaussianNB()

gnb.fit(X_train_sm, y_train_sm)
y_pred = gnb.predict(X_test)

train_accuracy = accuracy_score(y_train_sm, gnb.predict(X_train_sm))
print("Train Accuracy Naive Bayes", train_accuracy)
test_accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy Naive Bayes", test_accuracy)


# ## KNN Algorithm

# In[149]:


knn = KNeighborsClassifier()

knn.fit(X_train_sm, y_train_sm)
y_pred = knn.predict(X_test.values)

train_accuracy = accuracy_score(y_train_sm, knn.predict(X_train_sm.values))
print("Train Accuracy KNN Algorithm", train_accuracy)
test_accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy KNN Algorithm", test_accuracy)


# ## Decision Trees

# In[150]:


dtree = DecisionTreeClassifier()

dtree.fit(X_train_sm, y_train_sm)
y_pred = dtree.predict(X_test)

train_accuracy = accuracy_score(y_train_sm, dtree.predict(X_train_sm))
print("Train Accuracy Decision Trees", train_accuracy)
test_accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy Decision Trees", test_accuracy)


# ## Random Forest

# In[151]:


rfc = RandomForestClassifier()

rfc.fit(X_train_sm, y_train_sm)
y_pred = rfc.predict(X_test)

train_accuracy = accuracy_score(y_train_sm, rfc.predict(X_train_sm))
print("Train Accuracy Random Forest", train_accuracy)
test_accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy Random Forest", test_accuracy)


# ## Bar chart to compare the accuracy of all classification models - balanced data

# In[152]:


LR = accuracy_score(y_train_sm, lr.predict(X_train_sm))
SVM = accuracy_score(y_train_sm, svc.predict(X_train_sm))
NB = accuracy_score(y_train_sm, gnb.predict(X_train_sm))
KNN = accuracy_score(y_train_sm, knn.predict(X_train_sm.values))
DT = accuracy_score(y_train_sm, dtree.predict(X_train_sm))
RF = accuracy_score(y_train_sm, rfc.predict(X_train_sm))


# In[153]:


algorithms = ['LR','SVM','NB','KNN','DT', 'RF']
accuracies = [LR,SVM,NB,KNN,DT,RF]


# In[154]:


c = ['red', 'yellow', 'pink', 'blue', 'orange','green']
plt.bar(algorithms, accuracies,color=c)
plt.xlabel('Algorithm')
plt.ylabel('Accuracy (%)')
plt.title('Comparison of Classifier Accuracy - Balanced Data - Train Data')
#plt.ylim([0, 1])  # Set the y-axis limits between 0 and 1 or 0 and 100. 
plt.xticks(rotation=45)

for i in range(len(algorithms)):
    plt.text(i, accuracies[i],f"{accuracies[i]*100:.2f}%", ha='center',va= 'bottom')
plt.show()


# In[155]:


LR_test = accuracy_score(y_test, lr.predict(X_test))
SVM_test = accuracy_score(y_test, svc.predict(X_test))
NB_test = accuracy_score(y_test, gnb.predict(X_test))
KNN_test = accuracy_score(y_test, knn.predict(X_test.values))
DT_test = accuracy_score(y_test, dtree.predict(X_test))
RF_test = accuracy_score(y_test, rfc.predict(X_test))


# In[156]:


algorithms = ['LR','SVM','NB','KNN','DT', 'RF']
accuracies = [LR_test,SVM_test,NB_test,KNN_test,DT_test,RF_test]


# In[157]:


c = ['red', 'yellow', 'pink', 'blue', 'orange','green']
plt.bar(algorithms, accuracies,color=c)
plt.xlabel('Algorithm')
plt.ylabel('Accuracy (%)')
plt.title('Comparison of Classifier Accuracy - Balanced Data - Test Data')
#plt.ylim([0, 1])  # Set the y-axis limits between 0 and 1 or 0 and 100. 
plt.xticks(rotation=45)

for i in range(len(algorithms)):
    plt.text(i, accuracies[i],f"{accuracies[i]*100:.2f}%", ha='center',va= 'bottom')
plt.show()


# ## from the above training and testing accuracy comparison we can infer that our model is overfitted, To address this issue we shall introduce Regularization.

# ## Lasso

# In[158]:


from sklearn.linear_model import Lasso


# In[159]:


lasso = Lasso(alpha=0.001)

lasso.fit(X_train_sm, y_train_sm)
y_pred_lasso = lasso.predict(X_test).round()

train_accuracy = accuracy_score(y_train_sm, lasso.predict(X_train_sm).round())
print("Train Accuracy Lasso", train_accuracy)
test_accuracy = accuracy_score(y_test, y_pred_lasso)
print("Test Accuracy Lasso", test_accuracy)


# In[160]:


# Number of features Lasso has used
print(f"Number of features: {sum(lasso.coef_ != 0)}")


# ## Ridge Regularization

# In[161]:


from sklearn.linear_model import Ridge


# In[162]:


ridge = Ridge()

ridge.fit(X_train_sm, y_train_sm)
y_pred_ridge = ridge.predict(X_test).round()

train_accuracy = accuracy_score(y_train_sm, ridge.predict(X_train_sm).round())
print("Train Accuracy Ridge", train_accuracy)
test_accuracy = accuracy_score(y_test, y_pred_ridge)
print("Test Accuracy Ridge", test_accuracy)


# ## Elastic Net

# In[163]:


from sklearn.linear_model import ElasticNet
elastic_net = ElasticNet(alpha=0.01, l1_ratio=0.01)

elastic_net.fit(X_train_sm, y_train_sm)
y_pred_elastic = ridge.predict(X_test).round()

train_accuracy = accuracy_score(y_train_sm, elastic_net.predict(X_train_sm).round())
print("Train Accuracy Ridge", train_accuracy)
test_accuracy = accuracy_score(y_test, y_pred_elastic)
print("Test Accuracy Ridge", test_accuracy)


# # Future work

# ### Model building using stats model - reduce the number of features using p value and VIF values

# ## Recursive Feature Elimination Technique

# ### we shall try with other balancing techniques like Adasyn and hybridization techniques such as SMOTE+Tomek and SMOTE+ENN

# In[ ]:




