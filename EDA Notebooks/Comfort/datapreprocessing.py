import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    df.drop(columns=['tenure', 'customerID'], inplace=True)
    

    mappings = {
        'PhoneService': {'Yes': 1, 'No': 0},
        'Partner': {'Yes': 1, 'No': 0},
        'gender': {'Female': 1, 'Male': 0},
        'Dependents': {'Yes': 1, 'No': 0}, 
        'MultipleLines': {'Yes': 1, 'No': 0, 'No phone service': 2},
        'InternetService': {'DSL': 1, 'Fiber optic': 2, 'No': 0},
        'OnlineSecurity': {'Yes': 1, 'No': 0, 'No internet service': 2},
        'OnlineBackup': {'Yes': 1, 'No': 0, 'No internet service': 2},
        'DeviceProtection': {'Yes': 1, 'No': 0, 'No internet service': 2},
        'TechSupport': {'Yes': 1, 'No': 0, 'No internet service': 2},
        'StreamingTV': {'Yes': 1, 'No': 0, 'No internet service': 2},
        'StreamingMovies': {'Yes': 1, 'No': 0, 'No internet service': 2},
        'Contract': {'Month-to-month': 1, 'One year': 0, 'Two year': 2},
        'PaperlessBilling': {'Yes': 1, 'No': 0},
        'PaymentMethod': {'Electronic check': 1, 'Mailed check': 0, 'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3},
        'Churn': {'Yes': 1, 'No': 0}
    }
    
    for column, mapping in mappings.items():
        df[column] = df[column].map(mapping).astype(int)


    df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
    df['TotalCharges'] = df['TotalCharges'].astype(float)
    df.fillna(df["TotalCharges"].mean(),inplace=True)

    return df


def preprocess_and_split_data(df, target_col='Churn', test_size=0.3, random_state=42):
    # Extract features and target variable
    X_col = df.columns[df.columns != target_col].tolist()
    y_col = target_col
    X = df[X_col]
    y = df[y_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train = X_train_scaled
    X_test = X_test_scaled

    return X_train, X_test, y_train, y_test


