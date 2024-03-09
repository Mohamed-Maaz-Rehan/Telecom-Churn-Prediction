import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("Telecom Churn Prediction.csv")

df.info()

df.drop("customerID", axis=1, inplace=True)


# Define a mapping dictionary

df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})
df['MultipleLines'] = df['MultipleLines'].map({'No': 0, 'Yes': 1, 'No phone service': 2})
df['Contract'] = df['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
df['InternetService'] = df['InternetService'].map({'No': 0, 'DSL': 1, 'Fiber optic': 2})
df['PaymentMethod'] = df['PaymentMethod'].map({'Electronic check': 0, 'Mailed check': 1, 'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3})
df['TotalCharges']=df['TotalCharges'].replace({' ': 0})
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])

for col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']:
    df[col] = df[col].map({'No': 0, 'Yes': 1})
for col in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']:
    df[col] = df[col].map({'No': 0, 'Yes': 1, 'No internet service': 2})


# Using min max scaler technique on tenure, monthlycharges and total charges column

scaler = MinMaxScaler()
df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(df[['tenure', 'MonthlyCharges', 'TotalCharges']])

# Splitting the data

X = df.drop("Churn", axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=100)

y_train.value_counts()

# Performing undersampling as the target column has not equal values

under = RandomUnderSampler(sampling_strategy=1) 
                     
X_train, y_train = under.fit_resample(X_train, y_train)

y_train.value_counts()

model = LogisticRegression()

# Testing data
model.fit(X_train, y_train)

predict = model.predict(X_test)

accuracy = accuracy_score(y_test, predict)
print(f"Testing Accuracy: {accuracy * 100:.2f}%")

logistic_confusion_matrix = confusion_matrix(y_test, predict)
logistic_confusion_matrix

# Training data
model.fit(X_train, y_train)

predict = model.predict(X_train)

accuracy = accuracy_score(y_train, predict)
print(f"Training Accuracy: {accuracy * 100:.2f}%")

knn = KNeighborsClassifier()

knn.fit(X_train, y_train)
predict = knn.predict(X_test)
accuracy = accuracy_score(y_test, predict)
print(f"Testing Accuracy: {accuracy * 100:.2f}%")


# Testing data
model.fit(X_train, y_train)

predict = model.predict(X_train)

accuracy = accuracy_score(y_train, predict)
print(f"Training Accuracy: {accuracy * 100:.2f}%")


