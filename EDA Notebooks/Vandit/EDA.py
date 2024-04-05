import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv("telecom.csv")

print(data.head())


print("\nData Types:\n", data.dtypes)
data.TotalCharges = pd.to_numeric(data.TotalCharges, errors='coerce')
print("\nData Types:\n", data.dtypes)

print("\nBasic Statistics:\n", data.describe())


print("\nMissing Values:\n", data.isnull().sum())
data.dropna(inplace=True)
data['Churn'].replace("Yes",1,inplace=True)
data['Churn'].replace("No",0,inplace=True)
data.replace({True:1, False:0})

df = pd.get_dummies(data)
print(df.isnull().sum())




