import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression, Lasso, Ridge, ElasticNet
from sklearn.metrics import accuracy_score


class TelecomChurnPredictor:
    """
    This class contains all the method for model building and predicts the telecom churn
    """
    def __init__(self, file_path):
        """
        Reads the csv file
        :param file_path: csv file
        """
        self.df = pd.read_csv(file_path)

    def print_data(self, rows=2):
        for i in range(rows):
            print(self.df[i])

    def data_cleaning(self):
        """
        Method to get the descriptive statistic summary,
        check missing and duplicate values in the dataset df.
        Dropped Customer ID column from the df.
        Updated TotalCharges and Churn
        :return: statistics, missing_values and duplicate_values
        """
        self.df.drop("customerID", axis=1, inplace=True)
        self.df['TotalCharges'] = self.df['TotalCharges'].replace({' ': 0})
        self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'])
        self.df['Churn'] = self.df['Churn'].replace({'Yes': 0, 'No': 1})
        statistics = self.df.describe()
        missing_values = self.df.isna().any()
        duplicate_values = self.df.duplicated()
        return statistics, missing_values, duplicate_values

    def data_scaling(self):
        """
        Method to normalise the dataset
        """
        print("The count of Churn is  ", self.df['Churn'].value_counts())
        scaler = MinMaxScaler()
        self.df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(
            self.df[['tenure', 'MonthlyCharges', 'TotalCharges']])

    def data_encoding(self, columns):
        """
        Method to perform one-hot encoding and label encoding
        :param columns: columns on which one-hot encoding should be applied
        """
        # One-hot encoding
        self.df = pd.get_dummies(self.df, columns=columns)

        # label encoding
        numeric_cols = self.df._get_numeric_data().columns
        categ_cols = list(set(self.df.columns) - set(numeric_cols))
        lb = LabelEncoder()
        for i in categ_cols:
            self.df[i] = lb.fit_transform(self.df[i])

    def split_data(self):
        """
        Method to split the dataset into train and test
        :return: X_train, X_test, y_train, y_test
        """
        X = self.df.drop("Churn", axis=1)
        y = self.df['Churn']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=100)
        return X_train, X_test, y_train, y_test

    def oversample_data(self, X_train, y_train):
        """
        Method to perform oversampling using SMOTE
        :param X_train: training data
        :param y_train: training data with target column
        :return:
        """
        smote = SMOTE(random_state=42)
        X_oversampled, y_oversampled = smote.fit_resample(X_train, y_train)
        return X_oversampled, y_oversampled

    def logistic_model(self, X_train, y_train, X_test, y_test):
        """
        Method to perform logistic regression
        :param X_train: train data with all features except target
        :param y_train: train data with target feature
        :param X_test: test data with all features except target
        :param y_test: test data with target feature
        """
        logistic = LogisticRegression()
        logistic.fit(X_train, y_train)
        predict = logistic.predict(X_test)
        accuracy = accuracy_score(y_test, predict)
        print(f"Testing Accuracy (Logistic Regression): {accuracy * 100:.2f}%")

    def logistic_model_lasso(self, X_train, y_train, X_test, y_test):
        """
        Method to perform logistic regression with Lasso
        :param X_train: train data with all features except target
        :param y_train: train data with target feature
        :param X_test: test data with all features except target
        :param y_test: test data with target feature
        """
        lasso = Lasso(alpha=0.1)
        lasso.fit(X_train, y_train)
        predict = lasso.predict(X_test)
        # Assuming y_test and predict are binary classes for logistic regression
        predict_binary = [1 if p > 0.5 else 0 for p in predict]  # Convert to binary
        accuracy = accuracy_score(y_test, predict_binary)
        print(f"Testing Accuracy (Lasso Regression): {accuracy * 100:.2f}%")

    def logistic_model_rigid(self, X_train, y_train, X_test, y_test):
        """
        Method to perform logistic regression with Rigid
        :param X_train: train data with all features except target
        :param y_train: train data with target feature
        :param X_test: test data with all features except target
        :param y_test: test data with target feature
        """
        ridge = Ridge(alpha=0.1)
        ridge.fit(X_train, y_train)
        predict = ridge.predict(X_test)
        # Assuming y_test and predict are binary classes for logistic regression
        predict_binary = [1 if p > 0.5 else 0 for p in predict]  # Convert to binary
        accuracy = accuracy_score(y_test, predict_binary)
        print(f"Testing Accuracy (Ridge Regression): {accuracy * 100:.2f}%")

    def logistic_model_elastic_net(self, X_train, y_train, X_test, y_test):
        """
        Method to perform logistic regression with elastic net
        :param X_train: train data with all features except target
        :param y_train: train data with target feature
        :param X_test: test data with all features except target
        :param y_test: test data with target feature
        """
        elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
        elastic_net.fit(X_train, y_train)
        predict = elastic_net.predict(X_test)
        # Assuming y_test and predict are binary classes for logistic regression
        predict_binary = [1 if p > 0.5 else 0 for p in predict]  # Convert to binary
        accuracy = accuracy_score(y_test, predict_binary)
        print(f"Testing Accuracy (Elastic Net Regression): {accuracy * 100:.2f}%")

telecom_predictor = TelecomChurnPredictor("Telecom Churn Prediction.csv")
statistics, missing_values, duplicate_values = telecom_predictor.data_cleaning()
print("\n The summary statistic is ", statistics)
print("\n The missing values are ", missing_values)
print("\n The duplicate values are ", duplicate_values)

telecom_predictor.data_scaling()
telecom_predictor.print_data()

cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
telecom_predictor.data_encoding(columns=cols)
telecom_predictor.print_data(rows=3)

X_train, X_test, y_train, y_test = telecom_predictor.split_data()
X_oversampled, y_oversampled = telecom_predictor.oversample_data(X_train, y_train)

telecom_predictor.logistic_model(X_oversampled, y_oversampled, X_test, y_test)
telecom_predictor.logistic_model_rigid(X_oversampled, y_oversampled, X_test, y_test)
telecom_predictor.logistic_model_elastic_net(X_oversampled, y_oversampled, X_test, y_test)
