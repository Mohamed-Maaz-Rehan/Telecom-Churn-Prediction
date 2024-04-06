import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from Model import models
def dtreformat(df):
    df = df.drop(['customerID', 'gender', 'PhoneService', 'MultipleLines', 'StreamingMovies'], axis=1)
    df['TotalCharges'] = df['TotalCharges'].replace({' ': 0})
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
    df['Churn'] = df['Churn'].replace({'Yes': 0, 'No': 1})
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_cols)
    return df

def datasplit(df):
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    return X_train, X_test, y_train, y_test

def dtscale(X_train, X_test):
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def smote(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled

# Load dataset
df = pd.read_csv('/Users/HP/ML_Projects/telecom.csv')

# Data preprocessing
ftData = dtreformat(df)
X_train, X_test, Y_train, Y_test = datasplit(ftData)
X_train_scaled, X_test_scaled = dtscale(X_train, X_test)
X_oversampled, Y_oversampled = smote(X_train_scaled, Y_train)


