import pandas as pd
from sklearn.model_selection import train_test_split

from EDA import df


def split_data(df, test_size=0.3, random_state=42):

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":

    data_clean = df.dropna()


    X_train, X_test, y_train, y_test = split_data(data_clean, test_size=0.3, random_state=42)


    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
