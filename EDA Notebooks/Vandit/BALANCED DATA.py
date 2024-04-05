import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from EDA import df

# def encode_categorical_variables(df):
#     le = LabelEncoder()
#     for col in df.select_dtypes(include=['object']).columns:
#         df[col] = le.fit_transform(df[col])
#     return df

# Applying encoding
# df_encoded = encode_categorical_variables(df.copy())

# Split the data into features and target
X = df.drop('Churn', axis=1)
y = df['Churn'].astype(int)  # Ensure the target is integer

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


def apply_smote(X, y):
    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(X, y)
    return X_smote, y_smote


def apply_rus(X, y):
    undersampler = RandomUnderSampler(random_state=42)
    X_rus, y_rus = undersampler.fit_resample(X, y)
    return X_rus, y_rus


X_train_smote, y_train_smote = apply_smote(X_train_scaled, y_train)


X_train_rus, y_train_rus = apply_rus(X_train_scaled, y_train)


