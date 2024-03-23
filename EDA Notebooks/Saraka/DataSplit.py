import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import MinMaxScaler
from imblearn.combine import SMOTETomek


def dtreformat(df):
    df = df.drop((['customerID', 'gender', 'PhoneService', 'MultipleLines','StreamingMovies']), axis=1)
    # df = df.drop((['customerID', 'gender', 'PhoneService', 'MultipleLines', 'InternetService', 'StreamingTV',
    #              'StreamingMovies']), axis=1)
    df['TotalCharges'] = df['TotalCharges'].replace({' ': 0})
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
    numeric_cols = df._get_numeric_data().columns
    categ_cols = list(set(df.columns) - set(numeric_cols))
    lb = LabelEncoder()
    for i in categ_cols:
        df[i] = lb.fit_transform(df[i])

    return df


def datasplit(df):
    X = df.drop(columns=['Churn'])
    Y = df['Churn']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=10)
    return X_train, X_test, Y_train, Y_test


def dtscale(X_train, X_test):
    numeric_cols = X_train._get_numeric_data().columns
    scaler = MinMaxScaler()
    for i in numeric_cols:
        X_train[[i]] = scaler.fit_transform(X_train[[i]])
    numeric_cols = X_test._get_numeric_data().columns
    scaler = MinMaxScaler()
    for i in numeric_cols:
        X_test[[i]] = scaler.fit_transform(X_test[[i]])
    return X_train, X_test


def smote(x, y):
    smote = SMOTE(random_state=42)
    X_oversampled, y_oversampled = smote.fit_resample(x, y)
    return X_oversampled, y_oversampled


def RUS(x, y):
    undersampler = RandomUnderSampler(random_state=42)
    X_undersampled, y_undersampled = undersampler.fit_resample(x, y)
    return X_undersampled, y_undersampled


def combine(x, y):
    smote_enn = SMOTETomek(random_state=0)
    X_combined, y_combined = smote_enn.fit_resample(x, y)
    return X_combined, y_combined
