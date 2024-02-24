#!/usr/bin/env python
# coding: utf-8

# # Telecom Churn Prediction 

# In[1]:


## supress warnings

import warnings
warnings.filterwarnings('ignore')


# In[50]:


# Importing necessary libraries 

import numpy as np
import pandas as pd 

import seaborn as sns 
import matplotlib.pyplot as plt 


# In[51]:


## Importing a dataset 

# There are two ways to import dataset 
# 1. Path 
# 2. filename

df = pd.read_csv('telecom.csv')


# In[52]:


# To check the top 5 rows in the dataset 

df.head()


# In[53]:


# To check the bottom 5 rows in the dataset 

df.tail()


# In[54]:


# To check the shape of the dataset 

df.shape


# In[55]:


# To get the statistical info of the dataset 

df.describe()


# In[56]:


# DATA VISUALIZATION


# In[57]:


# two libraries for data visualization

## seaborn 
## matplotlib 


# In[58]:


# data visualization for InternetService,PhoneService and MultipleLines 


# # The four types of EDA are univariate non-graphical, multivariate non- graphical, univariate graphical, and multivariate graphical.

# # 1 Univariate non-graphical Analysis

# # Frequency Count

# In[59]:


# Count occurrences of categories in 'PhoneService' column
phone_service_counts = df['PhoneService'].value_counts()
print("Phone Service Counts:")
print(phone_service_counts)

# Count occurrences of categories in 'MultipleLines' column
multiple_lines_counts = df['MultipleLines'].value_counts()
print("\nMultiple Lines Counts:")
print(multiple_lines_counts)

# Count occurrences of categories in 'InternetService' column
internet_service_counts = df['InternetService'].value_counts()
print("\nInternet Service Counts:")
print(internet_service_counts)


# # Missing Values

# In[60]:


# Check for missing values in 'PhoneService' column
missing_values = df['PhoneService'].isnull().sum()
print(f"Number of missing values in 'PhoneService': {missing_values}")

# Check for missing values in 'MultipleLines' column
missing_values = df['MultipleLines'].isnull().sum()
print(f"Number of missing values in 'MultipleLines': {missing_values}")

# Check for missing values in 'InternetService' column
missing_values = df['InternetService'].isnull().sum()
print(f"Number of missing values in 'InternetService': {missing_values}")


# # To check the duplicates

# In[61]:


phone_service_duplicates = df['PhoneService'].duplicated()
print("Duplicate values in 'PhoneService':")
print(phone_service_duplicates)

multiple_lines_duplicates = df['MultipleLines'].duplicated()
print("\nDuplicate values in 'MultipleLines':")
print(multiple_lines_duplicates)

internet_service_duplicates = df['InternetService'].duplicated()
print("\nDuplicate values in 'InternetService':")
print(internet_service_duplicates)


# # Summary Statistics: Although summary statistics like mean, median, and mode are more relevant for numerical data, for binary categorical data (like Yes/No), you could calculate the proportion of “Yes” responses (which could serve as a “mean”).

# In[62]:


# Calculate the proportion of 'Yes' values in the 'PhoneService' column
phone_service_mean = (df['PhoneService'] == 'Yes').mean()
print("Proportion of 'Yes' values in 'PhoneService':", phone_service_mean)

# Calculate the proportion of 'Yes' values in the 'MultipleLines' column
multiple_lines_mean = (df['MultipleLines'] == 'Yes').mean()
print("Proportion of 'Yes' values in 'MultipleLines':", multiple_lines_mean)

# Calculate the proportion of 'Yes' values in the 'InternetService' column
internet_service_mean = (df['InternetService'] == 'Yes').mean()
print("Proportion of 'Yes' values in 'InternetService':", internet_service_mean)


# # Percentage Analysis

# In[63]:


phone_service_percentages = df['PhoneService'].value_counts(normalize=True) * 100
print("Percentage of each category in 'PhoneService':")
print(phone_service_percentages)

multiple_lines_percentages = df['MultipleLines'].value_counts(normalize=True) * 100
print("\nPercentage of each category in 'MultipleLines':")
print(multiple_lines_percentages)

internet_service_percentages = df['InternetService'].value_counts(normalize=True) * 100
print("\nPercentage of each category in 'InternetService':")
print(internet_service_percentages)


# # 2 Multivariate non- graphical Analysis

# # Cross-tabulation

# In[64]:


# Crosstab between 'PhoneService' and 'Churn'
phone_service_churn_crosstab = pd.crosstab(df['PhoneService'], df['Churn'])
print("Crosstab between 'PhoneService' and 'Churn':")
print(phone_service_churn_crosstab)

# Crosstab between 'MultipleLines' and 'Churn'
multiple_lines_churn_crosstab = pd.crosstab(df['MultipleLines'], df['Churn'])
print("\nCrosstab between 'MultipleLines' and 'Churn':")
print(multiple_lines_churn_crosstab)

# Crosstab between 'InternetService' and 'Churn'
internet_service_churn_crosstab = pd.crosstab(df['InternetService'], df['Churn'])
print("\nCrosstab between 'InternetService' and 'Churn':")
print(internet_service_churn_crosstab)


# # CORRELATION between (PhoneService , MultipleLines and InternetService) and Churn
 We can calculate the correlation coefficient between ‘PhoneService, MultipleLines, InternetService’ and ‘Churn’ after encoding them to numerical values.
# In[17]:


df_encoded = df.copy()

# Map 'Yes' and 'No' to 1 and 0 for 'PhoneService', 'MultipleLines', 'InternetService', and 'Churn' columns
df_encoded['PhoneService'] = df_encoded['PhoneService'].map({'Yes': 1, 'No': 0})
df_encoded['MultipleLines'] = df_encoded['MultipleLines'].map({'Yes': 1, 'No': 0})
df_encoded['InternetService'] = df_encoded['InternetService'].map({'Yes': 1, 'No': 0})
df_encoded['Churn'] = df_encoded['Churn'].map({'Yes': 1, 'No': 0})

# Calculate correlation
correlation_phone_churn = df_encoded[['PhoneService', 'Churn']].corr()
print("Correlation between 'PhoneService' and 'Churn':")
print(correlation_phone_churn)

correlation_multiple_churn = df_encoded[['MultipleLines', 'Churn']].corr()
print("\nCorrelation between 'MultipleLines' and 'Churn':")
print(correlation_multiple_churn)

correlation_internet_churn = df_encoded[['InternetService', 'Churn']].corr()
print("\nCorrelation between 'InternetService' and 'Churn':")
print(correlation_internet_churn)


# # 3. Univariate graphical analysis

# # Bar Plot

# In[18]:


# Count the frequency of each category in 'PhoneService'
phone_service_counts = df['PhoneService'].value_counts()

# Create bar plot for 'PhoneService'
sns.barplot(x=phone_service_counts.index, y=phone_service_counts.values)
plt.title('PhoneService Distribution')
plt.xlabel('PhoneService')
plt.ylabel('Count')
plt.show()

# Count the frequency of each category in 'MultipleLines'
multiple_lines_counts = df['MultipleLines'].value_counts()

# Create bar plot for 'MultipleLines'
sns.barplot(x=multiple_lines_counts.index, y=multiple_lines_counts.values)
plt.title('MultipleLines Distribution')
plt.xlabel('MultipleLines')
plt.ylabel('Count')
plt.show()

# Count the frequency of each category in 'InternetService'
internet_service_counts = df['InternetService'].value_counts()

# Create bar plot for 'InternetService'
sns.barplot(x=internet_service_counts.index, y=internet_service_counts.values)
plt.title('InternetService Distribution')
plt.xlabel('InternetService')
plt.ylabel('Count')
plt.show()


# # Count Plot

# In[19]:


# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot countplots for each variable
sns.countplot(data=df, x='PhoneService', ax=axes[0])
axes[0].set_title('PSD')

sns.countplot(data=df, x='MultipleLines', ax=axes[1])
axes[1].set_title('MLD')

sns.countplot(data=df, x='InternetService', ax=axes[2])
axes[2].set_title('ISD')

# Adjust labels and title
plt.xlabel('Category')
plt.ylabel('Count')

# Display plot
plt.tight_layout()
plt.show()


# # Pie Chart

# In[20]:


import matplotlib.pyplot as plt

# Calculate the frequencies of each category in 'PhoneService'
phone_service_counts = df['PhoneService'].value_counts()

# Create a pie chart for 'PhoneService'
plt.figure(figsize=(8, 6))
plt.pie(phone_service_counts, labels=phone_service_counts.index, autopct='%1.1f%%', startangle=140)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('PhoneService Distribution')
plt.show()

# Calculate the frequencies of each category in 'MultipleLines'
multiple_lines_counts = df['MultipleLines'].value_counts()

# Create a pie chart for 'MultipleLines'
plt.figure(figsize=(8, 6))
plt.pie(multiple_lines_counts, labels=multiple_lines_counts.index, autopct='%1.1f%%', startangle=140)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('MultipleLines Distribution')
plt.show()

# Calculate the frequencies of each category in 'InternetService'
internet_service_counts = df['InternetService'].value_counts()

# Create a pie chart for 'InternetService'
plt.figure(figsize=(8, 6))
plt.pie(internet_service_counts, labels=internet_service_counts.index, autopct='%1.1f%%', startangle=140)
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.title('InternetService Distribution')
plt.show()



# # 4 Multivariate Graphical Analysis

# # Jitter Plot

# In[21]:


import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'Yes' is encoded as 1 and 'No' as 0
df_encoded = df.copy()
df_encoded['PhoneService'] = df_encoded['PhoneService'].map({'Yes': 1, 'No': 0})
df_encoded['Churn'] = df_encoded['Churn'].map({'Yes': 1, 'No': 0})

# Create jitter plot
plt.figure(figsize=(8, 6))
sns.stripplot(x='PhoneService', y='Churn', data=df_encoded, jitter=True, palette='Set1')
plt.title('Jitter Plot of PhoneService vs Churn')
plt.xlabel('PhoneService')
plt.ylabel('Churn')
plt.show()


# In[22]:


import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'Yes' is encoded as 1 and 'No' as 0
df_encoded = df.copy()
df_encoded['MultipleLines'] = df_encoded['MultipleLines'].map({'Yes': 1, 'No': 0})
df_encoded['Churn'] = df_encoded['Churn'].map({'Yes': 1, 'No': 0})

# Create jitter plot
plt.figure(figsize=(8, 6))
sns.stripplot(x='MultipleLines', y='Churn', data=df_encoded, jitter=True, palette='Set1')
plt.title('Jitter Plot of MultipleLines vs Churn')
plt.xlabel('MultipleLines')
plt.ylabel('Churn')
plt.show()


# In[23]:


import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'Yes' is encoded as 1 and 'No' as 0
df_encoded = df.copy()
df_encoded['InternetService'] = df_encoded['InternetService'].map({'Yes': 1, 'No': 0})
df_encoded['Churn'] = df_encoded['Churn'].map({'Yes': 1, 'No': 0})

# Create swarm plot
plt.figure(figsize=(8, 6))
sns.swarmplot(x='InternetService', y='Churn', data=df_encoded, palette='Set1')
plt.title('Swarm Plot of InternetService vs Churn')
plt.xlabel('InternetService')
plt.ylabel('Churn')
plt.show()


# # Countplot of relationship with churn 

# In[24]:


import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'Yes' is encoded as 1 and 'No' as 0
df_encoded = df.copy()
df_encoded['PhoneService'] = df_encoded['PhoneService'].map({'Yes': 1, 'No': 0})
df_encoded['Churn'] = df_encoded['Churn'].map({'Yes': 1, 'No': 0})

# Create a grouped bar plot
plt.figure(figsize=(8, 6))
sns.countplot(data=df_encoded, x='PhoneService', hue='Churn', palette='Set1')
plt.title('Relationship between PhoneService and Churn')
plt.xlabel('PhoneService')
plt.ylabel('Count')
plt.legend(title='Churn')
plt.show()


# In[25]:


import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'Yes' is encoded as 1 and 'No' as 0
df_encoded = df.copy()
df_encoded['MultipleLines'] = df_encoded['MultipleLines'].map({'Yes': 1, 'No': 0})
df_encoded['Churn'] = df_encoded['Churn'].map({'Yes': 1, 'No': 0})

# Create a grouped bar plot
plt.figure(figsize=(8, 6))
sns.countplot(data=df_encoded, x='MultipleLines', hue='Churn', palette='Set1')
plt.title('Relationship between MultipleLines and Churn')
plt.xlabel('MultipleLines')
plt.ylabel('Count')
plt.legend(title='Churn')
plt.show()


# In[26]:


import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'Yes' is encoded as 1 and 'No' as 0
df_encoded = df.copy()
df_encoded['InternetService'] = df_encoded['InternetService'].map({'Yes': 1, 'No': 0})
df_encoded['Churn'] = df_encoded['Churn'].map({'Yes': 1, 'No': 0})

# Create a grouped bar plot
plt.figure(figsize=(8, 6))
sns.countplot(data=df_encoded, x='InternetService', hue='Churn', palette='Set1', alpha=0.8)
plt.title('Relationship between InternetService and Churn')
plt.xlabel('InternetService')
plt.ylabel('Count')
plt.legend(title='Churn')
plt.show()


# # Scatter Plot

# In[27]:


# Assuming 'Yes' is encoded as 1 and 'No' as 0
df_encoded = df.copy()
df_encoded['PhoneService'] = df_encoded['PhoneService'].map({'Yes': 1, 'No': 0})
df_encoded['Churn'] = df_encoded['Churn'].map({'Yes': 1, 'No': 0})

# Create scatter plot
plt.scatter(df_encoded['PhoneService'], df_encoded['Churn'])
plt.title('Scatter Plot of PhoneService vs Churn')
plt.xlabel('PhoneService')
plt.ylabel('Churn')
plt.show()


# In[28]:


# Assuming 'Yes' is encoded as 1 and 'No' as 0
df_encoded = df.copy()
df_encoded['InternetService'] = df_encoded['InternetService'].map({'Yes': 1, 'No': 0})
df_encoded['Churn'] = df_encoded['Churn'].map({'Yes': 1, 'No': 0})

# Create scatter plot
plt.scatter(df_encoded['InternetService'], df_encoded['Churn'])
plt.title('Scatter Plot of InternetService vs Churn')
plt.xlabel('InternetService')
plt.ylabel('Churn')
plt.show()


# In[29]:


# Assuming 'Yes' is encoded as 1 and 'No' as 0
df_encoded = df.copy()
df_encoded['MultipleLines'] = df_encoded['MultipleLines'].map({'Yes': 1, 'No': 0})
df_encoded['Churn'] = df_encoded['Churn'].map({'Yes': 1, 'No': 0})

# Create scatter plot
plt.scatter(df_encoded['MultipleLines'], df_encoded['Churn'])
plt.title('Scatter Plot of MultipleLines vs Churn')
plt.xlabel('MultipleLines')
plt.ylabel('Churn')
plt.show()


# # Heatmap

# In[30]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Assuming 'Yes' is encoded as 1 and 'No' as 0
df_encoded = df.copy()
df_encoded['PhoneService'] = df_encoded['PhoneService'].map({'Yes': 1, 'No': 0})
df_encoded['MultipleLines'] = df_encoded['MultipleLines'].map({'Yes': 1, 'No': 0})
df_encoded['InternetService'] = df_encoded['InternetService'].map({'Yes': 1, 'No': 0})
df_encoded['Churn'] = df_encoded['Churn'].map({'Yes': 1, 'No': 0})

# Calculate correlation matrix
correlation_matrix = df_encoded[['PhoneService', 'MultipleLines', 'InternetService', 'Churn']].corr()

# Create heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Heatmap of PhoneService, MultipleLines, InternetService, and Churn')
plt.show()


# # Pairplot

# In[31]:


sns.pairplot(df_encoded)
plt.show()


# # multivariate graphical analysis

# In[32]:


import pandas as pd
import matplotlib.pyplot as plt

# Create a cross-tabulation of 'PhoneService', 'MultipleLines', and 'InternetService'
contingency_table = pd.crosstab([df['PhoneService'], df['MultipleLines']], df['InternetService'])

# Plot the contingency table as a stacked bar plot
contingency_table.plot(kind='bar', stacked=True, figsize=(10,7))

plt.title('PhoneService and MultipleLines vs InternetService')
plt.xlabel('PhoneService, MultipleLines')
plt.ylabel('Count')
plt.show()


# # One-Hot Encoding
This technique converts each category value into a new column and assigns a 1 or 0 (True/False) value to the column. This has the benefit of not weighting a value improperly but does add more columns to the data set.
# In[33]:


# Assuming 'Yes' is encoded as 1 and 'No' as 0
df_encoded = df.copy()
df_encoded = pd.get_dummies(df_encoded, columns=['PhoneService'], prefix = ['PhoneService'])

print(df_encoded.head())


# In[34]:


# Assuming 'Yes' is encoded as 1 and 'No' as 0
df_encoded = df.copy()
df_encoded = pd.get_dummies(df_encoded, columns=['MultipleLines'], prefix = ['MultipleLines'])

print(df_encoded.head())


# In[35]:


# Assuming 'Yes' is encoded as 1 and 'No' as 0
df_encoded = df.copy()
df_encoded = pd.get_dummies(df_encoded, columns=['InternetService'], prefix = ['InternetService'])

print(df_encoded.head())


# # Chi-Square Test
A statistical test used to determine if there is a significant association between two categorical variables.It helps to find out whether a difference between two categorical variables is due to chance or a relationship between them.
# In[36]:


from scipy.stats import chi2_contingency

# Assuming 'Yes' is encoded as 1 and 'No' as 0
df_encoded = df.copy()
df_encoded['PhoneService'] = df_encoded['PhoneService'].map({'Yes': 1, 'No': 0})
df_encoded['Churn'] = df_encoded['Churn'].map({'Yes': 1, 'No': 0})

# Create contingency table
contingency_table = pd.crosstab(df_encoded['PhoneService'], df_encoded['Churn'])

# Perform Chi-Square test
chi2, p, dof, expected = chi2_contingency(contingency_table)

print(f"Chi-Square statistic: {chi2}")
print(f"P-value: {p}")
print(f"Degrees of freedom: {dof}")


# In[37]:


from scipy.stats import chi2_contingency

# Assuming 'Yes' is encoded as 1 and 'No' as 0
df_encoded = df.copy()
df_encoded['MultipleLines'] = df_encoded['MultipleLines'].map({'Yes': 1, 'No': 0})
df_encoded['Churn'] = df_encoded['Churn'].map({'Yes': 1, 'No': 0})

# Create contingency table
contingency_table = pd.crosstab(df_encoded['MultipleLines'], df_encoded['Churn'])

# Perform Chi-Square test
chi2, p, dof, expected = chi2_contingency(contingency_table)

print(f"Chi-Square statistic: {chi2}")
print(f"P-value: {p}")
print(f"Degrees of freedom: {dof}")


# In[38]:


from scipy.stats import chi2_contingency

# Assuming 'Yes' is encoded as 1 and 'No' as 0
df_encoded = df.copy()
df_encoded['InternetService'] = df_encoded['InternetService'].map({'Yes': 1, 'No': 0})
df_encoded['Churn'] = df_encoded['Churn'].map({'Yes': 1, 'No': 0})

# Create contingency table
contingency_table = pd.crosstab(df_encoded['InternetService'], df_encoded['Churn'])

# Perform Chi-Square test
chi2, p, dof, expected = chi2_contingency(contingency_table)

print(f"Chi-Square statistic: {chi2}")
print(f"P-value: {p}")
print(f"Degrees of freedom: {dof}")


# # Ordinal Encoding
This technique converts each value to a whole number. It can be used when there is ordered relationship between the categories (like ‘bad’, ‘average’, ‘good’). In case of ‘PhoneService’ attribute, since it’s a binary variable (Yes/No), you can map ‘Yes’ to 1 and ‘No’ to 0:
# In[39]:


df_encoded = df.copy()
df_encoded['PhoneService'] = df_encoded['PhoneService'].map({'Yes': 1, 'No': 0})

print(df_encoded.head())


# In[40]:


df_encoded = df.copy()
df_encoded['MultipleLines'] = df_encoded['MultipleLines'].map({'Yes': 1, 'No': 0})

print(df_encoded.head())


# In[41]:


df_encoded = df.copy()
df_encoded['InternetService'] = df_encoded['InternetService'].map({'Yes': 1, 'No': 0})

print(df_encoded.head())


# In[42]:


from scipy.stats import chi2_contingency
chi2, p, _, _ = chi2_contingency(pd.crosstab( df['PhoneService'],df['Churn'],))
print("\nChi-square test between PhoneService and Churn:")
print("Chi-square:", chi2)
print("p-value:", p)


# In[43]:


from scipy.stats import chi2_contingency
chi2, p, _, _ = chi2_contingency(pd.crosstab( df['InternetService'],df['Churn'],))
print("\nChi-square test betweenInternetService and Churn:")
print("Chi-square:", chi2)
print("p-value:", p)


# In[44]:


from scipy.stats import chi2_contingency
chi2, p, _, _ = chi2_contingency(pd.crosstab(df['MultipleLines'], df['Churn']))
print("\nChi-square test between MultipleLines and Churn:")
print("Chi-square:", chi2)
print("p-value:", p)


# In[45]:


# Count of each class
churn_counts = df['Churn'].value_counts()

# Visualize the distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Churn', palette='Set1')
plt.title('Distribution of Churn')
plt.xlabel('Churn')
plt.ylabel('Count')
plt.show()

# Print the count of each class
print("Count of each class:")
print(churn_counts)

# Calculate class balance ratio
class_balance_ratio = churn_counts[1] / churn_counts[0]
print("Class balance ratio:", class_balance_ratio)


# In[ ]:




