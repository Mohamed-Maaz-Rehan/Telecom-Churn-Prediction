
import numpy as np
df['totalcharges'] = df['totalcharges'].replace(' ', np.nan).astype(float)

df.describe()

df['churn'].value_counts()

100 * df['churn'].value_counts() / len(df['churn'])

df.select_dtypes(include=['object']).shape

df.select_dtypes(include=['float64', 'int64']).shape

# df_updated['tenure_group'] = pd.cut(df_updated.tenure, range(1, 80, 12), right=False, labels=labels)

# df_updated['tenure_group'].value_counts()

# null value is dropped
df_updated['gender'] = df_updated['gender'].replace({'Female': 0, 'Male': 1})
df_updated['partner'] = df_updated['partner'].replace({'Yes': 1, 'No': 0})
df_updated['phoneservice'] = df_updated['phoneservice'].replace({'Yes': 1, 'No': 0})
df_updated['dependents'] = df_updated['dependents'].replace({'Yes': 1, 'No': 0})
df_updated['multiplelines'] = df_updated['multiplelines'].replace({'Yes': 1, 'No': 0, 'No phone service': 2})
df_updated['internetservice'] = df_updated['internetservice'].replace({'DSL': 1, 'Fiber optic': 2, 'No': 0})
df_updated['onlinesecurity'] = df_updated['onlinesecurity'].replace({'Yes': 1, 'No': 0, 'No internet service': 0})
df_updated['onlinebackup'] = df_updated['onlinebackup'].replace({'Yes': 1, 'No': 0, 'No internet service': 0})
df_updated['deviceprotection'] = df_updated['deviceprotection'].replace({'Yes': 1, 'No': 0, 'No internet service': 0})
df_updated['techsupport'] = df_updated['techsupport'].replace({'Yes': 1, 'No': 0, 'No internet service': 0})
df_updated['streamingtv'] = df_updated['streamingtv'].replace({'Yes': 1, 'No': 0, 'No internet service': 0})
df_updated['streamingmovies'] = df_updated['streamingmovies'].replace({'Yes': 1, 'No': 0, 'No internet service': 0})
df_updated['contract'] = df_updated['contract'].replace({'Month-to-month': 1, 'One year': 0, 'Two year': 0})
df_updated['paperlessbilling'] = df_updated['paperlessbilling'].replace({'Yes': 1, 'No': 0})
df_updated['paymentmethod'] = df_updated['paymentmethod'].replace(
    {'Electronic check': 1, 'Mailed check': 0, 'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3})
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

num_cols = df.select_dtypes(include=['float64', 'int64', 'int32']).columns
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

# selecting only totalcharges column to apply imputation by mean which contains null values
impute_cols = df[['totalcharges']].columns
impute_cols

df

# selecting independent column
X = df.iloc[:, :-1]
X

# selecting dependent column
y = df.iloc[:, -1:]
y

# splitting data into train and test, testing data contains 20%
