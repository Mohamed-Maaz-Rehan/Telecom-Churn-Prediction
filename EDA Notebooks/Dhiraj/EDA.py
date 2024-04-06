import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


df= pd.read_csv('telecom.csv')
df

df.sample(5)

df.shape

df.info()

df.columns = df.columns.str.lower()

df.columns

df.drop(columns=['customerid'],inplace=True)

df.columns

#replacing ' ' empty values with nan
df['totalcharges'] = df['totalcharges'].replace(' ', np.nan).astype(float)

df.describe()

df['churn'].value_counts()

100*df['churn'].value_counts()/len(df['churn'])

df.select_dtypes(include=['object']).shape

df.select_dtypes(include=['float64','int64']).shape

df.isnull().sum()

df.loc[df ['totalcharges'].isnull() == True]

#missing value ratio
(11/7044)*100

#Removing missing values 
df.dropna(how = 'any', inplace = True)

df_updated = df.copy()

# Get the max tenure
print(df_updated['tenure'].max())

# # Group the tenure in bins of 12 months
# labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]

# df_updated['tenure_group'] = pd.cut(df_updated.tenure, range(1, 80, 12), right=False, labels=labels)

# df_updated['tenure_group'].value_counts()

#null value is dropped
df.isnull().sum()

# Check for any duplicated rows
print("\nNumber of duplicated rows before cleaning:", df.duplicated().sum())

# Remove duplicated rows
df = df.drop_duplicates()

# Check for any duplicated rows after cleaning
print("Number of duplicated rows after cleaning:", df.duplicated().sum())

sns.histplot(df['totalcharges'],kde=True)


sns.boxplot(df['totalcharges'],orient='h')

sns.countplot(x = df['gender'])
plt.title('Telecom Churn Gender')
plt.show()

for i, predictor in enumerate(df.drop(columns=['churn', 'totalcharges', 'monthlycharges'])):
    plt.figure(i)
    sns.countplot(data=df, x=predictor, hue='churn')

df_updated['churn'] = np.where(df_updated.churn == 'Yes',1,0)

df_updated.head()

# Convert all the categorical variables into dummy variables
df_updated_dummies = pd.get_dummies(df_updated)
df_updated_dummies.head()

# Churn by Monthly charges
Mth = sns.kdeplot(df_updated_dummies.monthlycharges[(df_updated_dummies["churn"] == 0) ],
                color="Red", shade = True)
Mth = sns.kdeplot(df_updated_dummies.monthlycharges[(df_updated_dummies["churn"] == 1) ],
                ax =Mth, color="Blue", shade= True)
Mth.legend(["No Churn","Churn"],loc='upper right')
Mth.set_ylabel('Density')
Mth.set_xlabel('Monthly Charges')
Mth.set_title('Monthly charges by churn')

# Churn by Total charges
Mth = sns.kdeplot(df_updated_dummies.totalcharges[(df_updated_dummies["churn"] == 0) ],
                color="Red", shade = True)
Mth = sns.kdeplot(df_updated_dummies.totalcharges[(df_updated_dummies["churn"] == 1) ],
                ax =Mth, color="Blue", shade= True)
Mth.legend(["No Churn","Churn"],loc='upper right')
Mth.set_ylabel('Density')
Mth.set_xlabel('Total Charges')
Mth.set_title('Total charges by churn')

plt.figure(figsize=(19,7))
df_updated_dummies.corr()['churn'].sort_values(ascending = False).plot(kind='bar')

plt.figure(figsize=(10,10))
sns.heatmap(df_updated_dummies.corr(), cmap="Paired")

new_df1_target0=df_updated.loc[df_updated["churn"]==0]
new_df1_target1=df_updated.loc[df_updated["churn"]==1]

def uniplot(df,col,title,hue =None):
    
    sns.set_style('whitegrid')
    sns.set_context('talk')
    plt.rcParams["axes.labelsize"] = 20
    plt.rcParams['axes.titlesize'] = 22
    plt.rcParams['axes.titlepad'] = 30
    
    
    temp = pd.Series(data = hue)
    fig, ax = plt.subplots()
    width = len(df[col].unique()) + 7 + 4*len(temp.unique())
    fig.set_size_inches(width , 8)
    plt.xticks(rotation=45)
    plt.yscale('log')
    plt.title(title)
    ax = sns.countplot(data = df, x= col, order=df[col].value_counts().index,hue = hue,palette='bright') 
        
    plt.show()

uniplot(new_df1_target1,col='partner',title='Distribution of Gender for Churned Customers',hue='gender')

uniplot(new_df1_target0,col='partner',title='Distribution of Gender for Non Churned Customers',hue='gender')

uniplot(new_df1_target1,col='paymentmethod',title='Distribution of PaymentMethod for Churned Customers',hue='gender')

uniplot(new_df1_target1,col='contract',title='Distribution of Contract for Churned Customers',hue='gender')

uniplot(new_df1_target1,col='techsupport',title='Distribution of TechSupport for Churned Customers',hue='gender')

uniplot(new_df1_target1,col='seniorcitizen',title='Distribution of SeniorCitizen for Churned Customers',hue='gender')


sns.scatterplot(x='monthlycharges', y='totalcharges',data=df)

#selecting only categorical column
cat_cols = df.select_dtypes(include=['object']).columns
cat_cols

#dropping out target column from categorical
cat_cols = cat_cols.drop('churn')

cat_cols

sns.histplot(df['monthlycharges'],kde=True)

sns.histplot(df['tenure'],kde=True)


# Plot count plots for categorical columns
plt.figure(figsize=(25, 12))
sns.countplot(x='churn', data=df)
plt.title('Distribution of Categorical columns')
plt.show()

# Assuming 'Yes' is encoded as 1 and 'No' as 0 (instead of Hot-coding)
df_updated['gender'] = df_updated['gender'].replace({'Female': 0,'Male': 1})
df_updated['partner'] = df_updated['partner'].replace({'Yes': 1, 'No': 0})
df_updated['phoneservice'] = df_updated['phoneservice'].replace({'Yes': 1, 'No': 0})
df_updated['dependents'] = df_updated['dependents'].replace({'Yes': 1, 'No': 0})
df_updated['multiplelines'] = df_updated['multiplelines'].replace({'Yes': 1, 'No': 0, 'No phone service':2})
df_updated['internetservice'] = df_updated['internetservice'].replace({'DSL': 1, 'Fiber optic': 2,'No':0})
df_updated['onlinesecurity'] = df_updated['onlinesecurity'].replace({'Yes': 1, 'No': 0, 'No internet service': 0})
df_updated['onlinebackup'] = df_updated['onlinebackup'].replace({'Yes': 1, 'No': 0, 'No internet service': 0})
df_updated['deviceprotection'] = df_updated['deviceprotection'].replace({'Yes': 1, 'No': 0, 'No internet service': 0})
df_updated['techsupport'] = df_updated['techsupport'].replace({'Yes': 1, 'No': 0, 'No internet service': 0})
df_updated['streamingtv'] = df_updated['streamingtv'].replace({'Yes': 1, 'No': 0, 'No internet service': 0})
df_updated['streamingmovies'] = df_updated['streamingmovies'].replace({'Yes': 1, 'No': 0, 'No internet service': 0})
df_updated['contract'] = df_updated['contract'].replace({'Month-to-month': 1, 'One year': 0, 'Two year': 0})
df_updated['paperlessbilling'] = df_updated['paperlessbilling'].replace({'Yes': 1, 'No': 0})
df_updated['paymentmethod'] = df_updated['paymentmethod'].replace({'Electronic check': 1, 'Mailed check': 0, 'Bank transfer (automatic)':2,'Credit card (automatic)':3})
df_updated['churn'] = df_updated['churn'].replace({'Yes': 1, 'No': 0})


df_updated.sample(10)

df.tenure.describe()

df.select_dtypes(include=['object']).columns

df.churn.unique()

# Calculate correlation matrix
correlation_matrix = df_updated[['phoneservice', 'multiplelines', 'internetservice', 'churn']].corr()

# Create heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Heatmap of PhoneService, MultipleLines, InternetService, and Churn')
plt.show()


# Calculate correlation matrix
correlation_matrix_cats = df_updated[['gender', 'partner', 'dependents', 'phoneservice', 'multiplelines',
       'internetservice', 'onlinesecurity', 'onlinebackup', 'deviceprotection',
       'techsupport', 'streamingtv', 'streamingmovies', 'contract',
       'paperlessbilling', 'paymentmethod', 'churn']].corr()

# Create heatmap for categorical data
plt.figure(figsize=(20, 9))
sns.heatmap(correlation_matrix_cats, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Heatmap of PhoneService, MultipleLines, InternetService, and Churn')
plt.show()

num_cols = df.select_dtypes(include=['float64', 'int64','int32']).columns
for column in num_cols:
    unique_values = df[column].unique()
    print(f"Column '{column}': {unique_values}")

# # Identifying numeric columns
# numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Assigning feature variable to X
x = df_updated.drop(['churn'], axis=1)

x.head()

y = df_updated['churn']

y.head()

#selecting only totalcharges column to apply imputation by mean which contains null values
impute_cols = df[['totalcharges']].columns
impute_cols

df

#selecting independent column
X = df.iloc[:,:-1]
X

#selecting dependent column
y = df.iloc[:,-1:]
y

#creating pipeline to impute totalcharges
impute_pipeline = Pipeline(steps=[('si',SimpleImputer())])

# Defining the parameter grid for Logistic Regression
param_grid_lr = {
    'logisticregression__C': [0.01, 0.1, 1, 10, 100]
}

# Create a GridSearchCV object
grid_search_lr = GridSearchCV(lr_pipeline, param_grid=param_grid_lr, cv=5, scoring='roc_auc', verbose=1)

# Best parameters and best score
print("Best parameters for Logistic Regression:", grid_search_lr.best_params_)
print("Best ROC-AUC score from Grid Search for Logistic Regression:", grid_search_lr.best_score_)

# Defining the parameter grid for Random Forest
param_grid_rf = {
    'randomforestclassifier__n_estimators': [10, 50, 100, 200],
    'randomforestclassifier__max_depth': [None, 10, 20, 30]
}

# Create a GridSearchCV object
grid_search_rf = GridSearchCV(rf_pipeline, param_grid=param_grid_rf, cv=5, scoring='roc_auc', verbose=1)


# Best parameters and best score
print("Best parameters for Random Forest:", grid_search_rf.best_params_)
print("Best ROC-AUC score from Grid Search for Random Forest:", grid_search_rf.best_score_)

