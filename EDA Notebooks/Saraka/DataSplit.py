import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import MinMaxScaler
from imblearn.combine import SMOTETomek
from sklearn.preprocessing import OneHotEncoder


def dtreformat(df):
    df = df.drop((['customerID', 'gender', 'PhoneService', 'MultipleLines', 'StreamingMovies']), axis=1)
    # df = df.drop((['customerID', 'gender', 'PhoneService', 'MultipleLines', 'InternetService', 'StreamingTV',
    #              'StreamingMovies']), axis=1)
    df['TotalCharges'] = df['TotalCharges'].replace({' ': 0})
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
    df['Churn'] = df['Churn'].replace({'Yes': 0, 'No': 1})
    numeric_cols = df._get_numeric_data().columns
    categ_cols = list(set(df.columns) - set(numeric_cols))
    # df = pd.get_dummies(df, columns=categ_cols)

    return df


def datasplit(df):
    X = df.drop(columns=['Churn'])
    Y = df['Churn']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=10)
    return X_train, X_test, Y_train, Y_test


def dtscale(X_train, X_test):
    trcols = 'MonthlyCharges', 'TotalCharges', 'tenure'
    scaler = MinMaxScaler()
    for i in trcols:
        X_train[[i]] = scaler.fit_transform(X_train[[i]])
        X_test[[i]] = scaler.fit_transform(X_test[[i]])
    return X_train, X_test


def encoder(x_train, x_test):
    encod = OneHotEncoder(handle_unknown="ignore")
    encod.fit(x_train)

    x_train = encod.transform(x_train)
    x_test = encod.transform(x_test)

    with open('encoder.pickle', 'wb') as f:
        pickle.dump(encod, f)
    return x_train, x_test


def smote(x, y):
    smote = SMOTE(random_state=42)
    X_oversampled, y_oversampled = smote.fit_resample(x, y)
    return X_oversampled, y_oversampled


def rus(x, y):
    undersampler = RandomUnderSampler(random_state=42)
    X_undersampled, y_undersampled = undersampler.fit_resample(x, y)
    return X_undersampled, y_undersampled


def combine(x, y):
    smote_enn = SMOTETomek(random_state=0)
    X_combined, y_combined = smote_enn.fit_resample(x, y)
    return X_combined, y_combined
