import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import warnings

warnings.filterwarnings("ignore")

# Assuming the models function is appropriately defined in 'models_module.py'
from MODELS import models  # Make sure to import your models function correctly

def load_data(filepath):
    return pd.read_csv(filepath)

def encode_categorical_variables(df):
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])
    return df

def apply_smote(X, y):
    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(X, y)
    return X_smote, y_smote

def apply_rus(X, y):
    undersampler = RandomUnderSampler(random_state=42)
    X_rus, y_rus = undersampler.fit_resample(X, y)
    return X_rus, y_rus

def scale_features(X_train, X_test):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def main():
    # Load and preprocess data
    filepath = 'telecom.csv'
    df = load_data(filepath)
    df_encoded = encode_categorical_variables(df)

    # Split the data into features and target
    X = df_encoded.drop('Churn', axis=1)
    y = df_encoded['Churn'].astype(int)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Scale features
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    # Train and evaluate on original data
    print("Evaluating on Original Data:")
    models(X_train_scaled, y_train, X_test_scaled, y_test)

    # Apply and train on SMOTE balanced data
    X_train_smote, y_train_smote = apply_smote(X_train_scaled, y_train)
    print("Evaluating on SMOTE Balanced Data:")
    models(X_train_smote, y_train_smote, X_test_scaled, y_test)

    # Apply and train on RUS balanced data
    X_train_rus, y_train_rus = apply_rus(X_train_scaled, y_train)
    print("Evaluating on RUS Balanced Data:")
    models(X_train_rus, y_train_rus, X_test_scaled, y_test)

if __name__ == "__main__":
    main()
