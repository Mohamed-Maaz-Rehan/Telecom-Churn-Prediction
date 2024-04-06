import pandas as pd
from sklearn.model_selection import train_test_split
from MODELS import models
from DATASPLIT import X_train, X_test, y_train, y_test
filepath="telecom.csv"
def load_data(filepath):
    df = pd.read_csv(filepath)
    return df


def preprocess_data(df):
    return df


def main():
    # Load data
    filepath = 'telecom.csv'
    df = load_data(filepath)
    df_preprocessed = preprocess_data(df)

    models(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    main()
