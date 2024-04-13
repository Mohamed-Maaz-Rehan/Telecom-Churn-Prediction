
import pandas as pd
import numpy as np
df = pd.read_csv('telecom.csv')

df.sample(5)

df.shape

df.info()

df.columns = df.columns.str.lower()

df.drop(columns=['customerid'], inplace=True)

# replacing ' ' empty values with nan
df.isnull().sum()

df.loc[df['totalcharges'].isnull() == True]

# missing value ratio
(11 / 7044) * 100

# Removing missing values
df.dropna(how='any', inplace=True)

df_updated = df.copy()

# Get the max tenure
print(df_updated['tenure'].max())

# # Group the tenure in bins of 12 months
# labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]

df.isnull().sum()

# Check for any duplicated rows
print("\nNumber of duplicated rows before cleaning:", df.duplicated().sum())

# Remove duplicated rows
df = df.drop_duplicates()

# Check for any duplicated rows after cleaning
print("Number of duplicated rows after cleaning:", df.duplicated().sum())

sns.histplot(df['totalcharges'], kde=True)

sns.boxplot(df['totalcharges'], orient='h')

sns.countplot(x=df['gender'])
plt.title('Telecom Churn Gender')
plt.show()

