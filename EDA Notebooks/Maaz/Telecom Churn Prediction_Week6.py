#!/usr/bin/env python
# coding: utf-8

# # Telecom Churn Prediction - Week 6

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


a = sns.countplot(df['Churn'])
a.bar_label(a.containers[0])
plt.title("Bar Plot of Churn")
plt.xlabel("Churn")
plt.ylabel("Count")
plt.savefig("Churn.png")
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


x = sns.countplot(df['SeniorCitizen'])
x.bar_label(x.containers[0])
plt.show()


# In[22]:


df_continuous = df_continuous.drop(columns='SeniorCitizen')
df_continuous


# ### Categorical Data

# In[23]:


for i in df_categorical:
    b = sns.countplot(df[i])
    b.bar_label(b.containers[0])
    plt.xticks(rotation=45)
    plt.show()


# In[24]:


for i in df_categorical:
    b = sns.countplot(df[i], hue = df['Churn'])
    b.bar_label(b.containers[0])
    b.bar_label(b.containers[1])
    plt.xticks(rotation=45)
    plt.show()


# ### Continuous data

# In[25]:


for i in df_continuous:
    sns.boxplot(df[i])
    plt.show()


# ### From the above boxplots No outliers are found 

# In[26]:


for i in df_continuous:
    sns.boxplot(x=df['Churn'], y=df[i])
    plt.show()


# ## histogram

# In[27]:


for i in df_continuous:
    sns.histplot(df_continuous[i])
    plt.show()


# ### pairsplot

# In[28]:


sns.pairplot(df)


# ### corelation plot

# In[29]:


sns.heatmap(df.corr(), annot=True)


# In[30]:


df.info()


# #### From the above info we can see that there are many categorical columns and we shall to convert the categorical columns into numeric columns in order to build the algorithms.

# ### converting binary variable - Yes and No to 1 and 0

# In[31]:


var = ['Partner','Dependents','PhoneService','PaperlessBilling','Churn']

# mapping function
def mapping(n):
    return n.map({'Yes':1,'No':0})

df[var] = df[var].apply(mapping)
    


# ### Label encoding for gender column

# In[32]:


# import label encoder
from sklearn import preprocessing


# In[33]:


# object creation - label_encoder
label_encoder = preprocessing.LabelEncoder()


# In[34]:


# encode labels in gender column
df['gender'] = label_encoder.fit_transform(df['gender'])


# In[35]:


# to check the unique values in gender column
df['gender'].unique()


# ### For cagetorical columns with multiple levels - create dummies

# In[36]:


## create dummy variables

dummy = pd.get_dummies(df[['MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport',
                           'StreamingTV','StreamingMovies','Contract','PaymentMethod',]], drop_first=True)


# ### concat with the original dataframe

# In[37]:


df = pd.concat([df,dummy],axis =1)


# In[38]:


df.head(1)


# ### Dropping the repeated variables 

# In[39]:


df = df.drop(['MultipleLines','InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport',
                           'StreamingTV','StreamingMovies','Contract','PaymentMethod'],1)


# In[40]:


df.head()


# ### checking for missing values

# In[41]:


df.isnull().sum()


# ##### It means that 11/7043 = 0.001561834 i.e 0.1%, best is to remove these observations from the analysis

# In[42]:


# Removing NaN TotalCharges rows
df = df[~np.isnan(df['TotalCharges'])]


# ##### again check for missing values

# In[43]:


df.isna().sum()


# ##### Now we dont have missing values

# ## Splitting the data into training and testing set

# In[44]:


from sklearn.model_selection import train_test_split


# In[45]:


# Putting feature variable to X
X = df.drop(['Churn'], axis=1)

X.head()


# In[46]:


# response variable to y
y = df['Churn']

y.head()


# In[47]:


# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=300)


# ### Feature scaling and Data Transformation

# In[48]:


from sklearn.preprocessing import MinMaxScaler


# In[49]:


scaler = MinMaxScaler()

X_train[['tenure','MonthlyCharges','TotalCharges']] = scaler.fit_transform(X_train[['tenure','MonthlyCharges','TotalCharges']])

X_train.head()


# In[50]:


X_test.head()


# ### Transforming to test data

# In[51]:


X_test[['tenure','MonthlyCharges','TotalCharges']] = scaler.transform(X_test[['tenure','MonthlyCharges','TotalCharges']])

X_test.head()


# ## Now the Data is ready for Model Building

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### PaymentMethod 

# In[ ]:


a = sns.countplot(df['PaymentMethod'])
a.bar_label(a.containers[0])
plt.xticks(rotation=45)
plt.title("Bar Plot of PaymentMethod")
plt.xlabel("PaymentMethod")
plt.ylabel("Count")
plt.show()


# From the above bar plot, the count of electronic check is higher than other Payment methods

# In[ ]:


a = sns.countplot(df['PaymentMethod'],hue = df['Churn'])
a.bar_label(a.containers[0])
a.bar_label(a.containers[1])
plt.xticks(rotation=45)
plt.title("Bar Plot of PaymentMethod")
plt.xlabel("PaymentMethod")
plt.ylabel("Count")
plt.show()


# ## MonthlyCharges 

# In[ ]:


sns.boxplot(df['MonthlyCharges'])
plt.title("BoxPlot of Monthly Charges")


# #### Histogram of Monthly Charges

# In[ ]:


sns.histplot(df['MonthlyCharges'])
plt.title("Histogram of Monthly Charges")


# In[ ]:


sns.boxplot(x = df['Churn'], y = df['MonthlyCharges'])


# ## TotalCharges

# ##### The datatype of TotalCharges is object, we shall convert this into int

# In[ ]:


#The varaible was imported as a string we need to convert it to float
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'],errors='coerce')


# In[ ]:


df.info()


# In[ ]:


sns.boxplot(df['TotalCharges'])
plt.title("BoxPlot of Total Charges")


# In[ ]:


sns.histplot(df['TotalCharges'])
plt.title("Histogram of Total Charges")


# ##### The total charges in the dataset is positively skewed

# In[ ]:


sns.boxplot(x = df['Churn'], y = df['TotalCharges'])


# In[ ]:




