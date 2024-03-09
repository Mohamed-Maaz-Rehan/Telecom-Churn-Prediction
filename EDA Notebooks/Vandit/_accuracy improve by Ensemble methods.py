#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Loading the libraries

import pandas as pd 
import numpy as np 
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt 
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')


# In[2]:


Data=pd.read_csv('telecom_churn.csv')# Importing the dataframe
pd.set_option('display.max_columns',25)# this helps to view every column 
Data.head()# to view rows


# In[ ]:






# In[ ]:





# In[3]:


# Removing CustomerID as it is of no use 
Data.drop('customerID',axis=1,inplace=True)


# In[4]:


# rows and columns the data has 

Data.shape


# In[5]:


# To see the basic data types of each column

Data.info()


#  # From the above we can see that total charges is dispalyed as objects whereas it is a integer.
# 
# # So we are converting it to a numeric column.

# In[6]:


Data.TotalCharges = pd.to_numeric(Data.TotalCharges, errors='coerce')


# In[7]:


# Checking whethere there is a null value in the data 

Data.isnull().sum()


# We can see that total charges has eleven NaN values.
# Since we have around 7000 records we can remove the null values.

# In[8]:


# Removing the null values 

Data.dropna(inplace=True)


# In[9]:


# Converting the String churn column in to binary by replacing Yes with 1 and No with 0

Data['Churn'].replace("Yes",1,inplace=True)
Data['Churn'].replace("No",0,inplace=True)


# In[10]:


# Converting the categorical columns in to binary using get dummies

Data1=pd.get_dummies(Data)
Data1.head()


# In[11]:


Data1.replace({True: 1, False: 0})


# In[12]:


plt.figure(figsize=(15,6))
Data1.corr()['Churn'].sort_values(ascending=False).plot.bar()


# We can see that month to month contract and absence of Online Security and Tech support is strongly positively correlated with the traget variable Churn and we also got the same info while doing visual anlaysis of the data in Tableau. 
# 
# Also we can see that Tenure and Two Year contract is negatively correlated with target variable Churn. 

# With basic Visual Analytics done in Tableau and Basic Exploration done here we are heading to model building and evaluation.

# In[13]:


Data1['Churn'].value_counts().plot(kind='barh',figsize=(8,6))
plt.xlabel("Count")
plt.ylabel("Target Variable")
plt.title("Count of TARGET Variable")


# In[14]:


Data1['Churn'].value_counts()


# # here 0= Not churn and 1= Churned
# # but these numbers are not making sense lets make a sense out of it  

# In[15]:


len(Data1['Churn'])


# In[16]:


100*Data1['Churn'].value_counts()/len(Data1['Churn'])


# Here we can infer that only approx 26% of customers churn where as 73% of customers continue with hteir isp 

# In[17]:


Data1.isnull().sum()


# no null values great!!

# # lets plot every column with respect to churn 
# ## this is part of data exploration

# In[18]:


for i, predictor in enumerate(Data1.drop(columns=['Churn','TotalCharges','MonthlyCharges'])):
    plt.figure(i)
    sns.countplot(x=predictor,data=Data1,hue='Churn')


# # Checking relationship between totalcharges and monthly charges

# In[19]:


sns.lmplot(x='MonthlyCharges',y='TotalCharges',data=Data1,fit_reg=False)


# # Model building
# # 

# # Validation Dataset

# We will use 70% of the dataset for modeling and hold back 30% for Test/Validation.

# In[29]:


# Split-out validation dataset
x = Data1.drop('Churn',axis=1)
y = Data1['Churn']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=42)


# # Evaluating Baseline Models 

# We are going to make simple kfold cross validation accross models and see what works the best for the data.

# In[30]:


# Test options and evaluation metric
num_folds = 10
scoring = 'accuracy'


# Let's create a baseline of performance on this problem and spot-check a number of different algorithms. We will select a suite of different algorithms capable of working on this classification problem. The algorithms selected include:
# 1. Linear Algorithms: Logistic Regression (LR)
# 2. Nonlinear Algorithms: Classiffication and Regression Trees (CART),k-Nearest Neighbors (KNN),Naive Bayes and Support Vector Classifier (SVC)

# We suspect that the differing scales of the raw data may be negatively impacting the skill of some of the algorithms. 
# 
# Let's evaluate the  algorithms with a standardized copy of the dataset. 
# 
# This is where the data is transformed such that each attribute has a mean value of zero and a standard deviation of 1. 
# 
# We also need to avoid data leakage when we transform the data. 
# 
# A good way to avoid leakage is to use pipelines that standardize the data and build the model for each fold in the cross-validation test harness. 
# 
# That way we can get a fair estimation of how each model with standardized data might perform on unseen data.

# In[31]:


# Standardize the dataset
pipelines = []
pipelines.append(('LR', Pipeline([('Scaler', StandardScaler()), ('LR', LogisticRegression())])))
pipelines.append(('KNN', Pipeline([('Scaler', StandardScaler()), ('KNN', KNeighborsClassifier())])))
pipelines.append(('CART', Pipeline([('Scaler', StandardScaler()), ('CART', DecisionTreeClassifier())])))
pipelines.append(('NB', Pipeline([('Scaler', StandardScaler()), ('NB', GaussianNB())])))
pipelines.append(('SVC', Pipeline([('Scaler', StandardScaler()), ('SVC', SVC())])))


# In[32]:


results = []
names = []
for name, model in pipelines:
    kfold = KFold(n_splits=num_folds, random_state=42, shuffle=True)
    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# In[33]:


# Compare Algorithms
fig = plt.figure(figsize=(6,6))
fig.suptitle('Scaled Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# Logistic Regression and SVC has the best accuracy with minimum standard deviation. 
# 
# Lets see whethere we can improve the same with tuning.

# # Improve Results With Tuning

# In[34]:


# SVC tuning 
scaler = StandardScaler().fit(x_train)
scaledx = scaler.transform(x_train)
scaledtest=scaler.transform(x_test)

c_values = np.arange(1,5)
kernel=['linear','rbf']
param_grid = dict(C=c_values,kernel=kernel)
model_svc=SVC()
kfold = KFold(n_splits=num_folds, random_state=42, shuffle=True)
grid_svc = GridSearchCV(estimator=model_svc, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result_svc = grid_svc.fit(scaledx, y_train)


# In[35]:


grid_result_svc.score(scaledx,y_train)


# With Tuning the accuracy of SVC is increased by 2%.
# 
# Now we will check the test score.

# In[36]:


grid_result_svc.score(scaledtest,y_test)


# The model is almost a perfect fit.

# In[37]:


# DecesionTreee Tuning 

scaler = StandardScaler().fit(x_train)
scaledx = scaler.transform(x_train)
scaledtest=scaler.transform(x_test)

max_depth= np.arange(1,8)
max_features=np.arange(1,45)
min_samples_split=np.arange(2,6)
param_grid = dict(max_depth=max_depth,max_features=max_features,min_samples_split=min_samples_split)
model_dt=DecisionTreeClassifier()
kfold = KFold(n_splits=num_folds, random_state=42, shuffle=True)
grid_dt = GridSearchCV(estimator=model_dt, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result_dt = grid_dt.fit(scaledx, y_train)


# In[38]:


grid_result_dt.score(scaledx,y_train)


# The accuracy is improved by 8 % but still it is lower than SVC.
# 
# We will check the test score.

# In[39]:


grid_result_dt.score(scaledtest,y_test)


# There is a slight over fit. 

# In[40]:


# Logistic regression Tuning

scaler = StandardScaler().fit(x_train)
scaledx = scaler.transform(x_train)
scaledtest=scaler.transform(x_test)

c_values = np.arange(2**-5,2**5)
param_grid = dict(C=c_values)
model_lr=LogisticRegression()
kfold = KFold(n_splits=num_folds, random_state=42, shuffle=True)
grid_lr = GridSearchCV(estimator=model_lr, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result_lr = grid_lr.fit(scaledx, y_train)


# In[41]:


grid_result_lr.score(scaledx,y_train)


# There isnt a much improvement in the score. 
# 
# We wil check the test score.

# In[42]:


grid_result_lr.score(scaledtest,y_test)


# We can see that the model is great with perfect fit.

# # Ensemble Methods

# Another way that we can improve the performance of algorithms on this problem is by using ensemble methods. 
# 
# We will evaluate two different ensemble machine learning algorithms, one boosting and one bagging method:
# 
# 1. Boosting Methods: AdaBoost (AB),GradientBoost (GB)
# 2. Bagging Methods: Random Forests (RF), Bagging (BG)

# In[52]:


# ensembles
ensembles = []
ensembles.append(('AB', Pipeline([('Scaler', StandardScaler()),('AB', AdaBoostClassifier())])))
ensembles.append(('GB', Pipeline([('Scaler', StandardScaler()),('GB', GradientBoostingClassifier())])))
ensembles.append(('RF', Pipeline([('Scaler', StandardScaler()),('RF', RandomForestClassifier())])))
ensembles.append(('Bag', Pipeline([('Scaler', StandardScaler()),('Bag', BaggingClassifier())])))


# In[53]:


results = []
names = []
for name, model in ensembles: 
    kfold = KFold(n_splits=num_folds, random_state=42,shuffle=True) 
    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# In[54]:


# Compare Algorithms
fig = plt.figure(figsize=(6,6))
fig.suptitle('Scaled Ensemble Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()


# Adaboost and GradientBoost ensemble gives the best results.

# We will try imporvingg the score further by tuning. 

# In[55]:


# Adaboost Tuning 

scaler = StandardScaler().fit(x_train)
scaledx = scaler.transform(x_train)
scaledtest=scaler.transform(x_test)
base_estimator=[LogisticRegression(C=6.03125, class_weight=None, dual=False,
                fit_intercept=True, intercept_scaling=1, max_iter=100,
          multi_class='ovr', n_jobs=1, penalty='l2', random_state=None,
          solver='liblinear', tol=0.0001, verbose=0, warm_start=False)]
n_estimators=np.arange(1,51)
param_grid = dict(base_estimator=base_estimator,n_estimators=n_estimators)
model_ab=AdaBoostClassifier()
kfold = KFold(n_splits=num_folds, random_state=42,shuffle=True)
grid_ab = GridSearchCV(estimator=model_ab, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result_ab = grid_ab.fit(scaledx, y_train)


# In[56]:


grid_ab.score(scaledx,y_train)


# There is a slight increase of 0.4 %. 

# In[57]:


grid_ab.score(scaledtest,y_test)


# Model is almost perfect fit. 

# In[59]:


# GradientBoost Tuning 

scaler = StandardScaler().fit(x_train)
scaledx = scaler.transform(x_train)
scaledtest=scaler.transform(x_test)


n_estimators=np.arange(1,101)
param_grid = dict(n_estimators=n_estimators)
model_gb=GradientBoostingClassifier()
kfold = KFold(n_splits=num_folds, random_state=42,shuffle=True)
grid_gb = GridSearchCV(estimator=model_ab, param_grid=param_grid, scoring=scoring, cv=kfold)
grid_result_gb = grid_gb.fit(scaledx, y_train)


# In[60]:


grid_gb.score(scaledx, y_train)


# There is 0.1% increase in the accuracy. 

# In[61]:


grid_gb.score(scaledtest, y_test)


# Model is at the right fit. 

# In[62]:


results=pd.DataFrame(index=['Logistic Regression',
                            'Decision Tree','Support Vector Classifier',
                            'Adaboost Classifier','Gradientboost Classifier'])


# In[63]:


results['Train']=[grid_result_lr.score(scaledx,y_train),
                  grid_result_dt.score(scaledx,y_train),
                  grid_result_svc.score(scaledx,y_train),
                  grid_result_ab.score(scaledx,y_train),
                  grid_result_gb.score(scaledx,y_train)]


# In[64]:


results['Test']=[grid_result_lr.score(scaledtest,y_test),
                  grid_result_dt.score(scaledtest,y_test),
                  grid_result_svc.score(scaledtest,y_test),
                  grid_result_ab.score(scaledtest,y_test),
                  grid_result_gb.score(scaledtest,y_test)]


# In[65]:


results


# In[ ]:




