import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve


class TelecomChurnPredictor:
    """
    This class contains all the method for model building and predicts the telecom churn using
    Logistic regression, AdaBoost and SVM
    """

    def __init__(self, file_path):
        """
        Reads the csv file
        :param file_path: csv file
        """
        self.df = pd.read_csv(file_path)

    def print_data(self):
        print(self.df.head())

    def data_cleaning(self):
        """
        Method to dropped Customer ID column from the dataset
        Updated TotalCharges, Churn, SeniorCitizen columns to categorical
        """
        self.df.drop("customerID", axis=1, inplace=True)
        self.df['TotalCharges'] = self.df['TotalCharges'].replace({' ': 0})
        self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'])
        self.df['Churn'] = self.df['Churn'].replace({'Yes': 0, 'No': 1})
        self.df['SeniorCitizen'] = self.df['SeniorCitizen'].replace({0: 'No', 1: 'Yes'})

    def data_scaling(self, data_list):
        """
        Method to normalise the dataset
        """
        scaler = MinMaxScaler()
        for data in data_list:
            data[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(
                data[['tenure', 'MonthlyCharges', 'TotalCharges']])

    def data_encoding(self):
        """
        Method to perform get dummies on dataset
        """
        numeric_cols = self.df._get_numeric_data().columns
        categ_cols = list(set(self.df.columns) - set(numeric_cols))
        self.df = pd.get_dummies(self.df, columns=categ_cols)

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
        :return: oversampled data as X_oversampled, y_oversampled
        """
        smote = SMOTE(random_state=42)
        X_oversampled, y_oversampled = smote.fit_resample(X_train, y_train)
        return X_oversampled, y_oversampled

    def plot_auroc(self, fprs, tprs, aucs, model):
        """
        Method to plot the AUC curve
        :param fprs: false positive rates
        :param tprs: true positive rates
        :param aucs: auc scores
        :param model: name of the model
        """
        plt.figure(figsize=(8, 6))
        for fpr, tpr, auc_score, clf_name in zip(fprs, tprs, aucs, model):
            sns.lineplot(x=fpr, y=tpr, label='%s (AUC = %0.2f)' % (clf_name, auc_score))

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (AUROC)')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()

    def ml_model(self, model, X_train, y_train, X_test, y_test):
        """
        Method to perform model building
        :param model: name of the model
        :param X_train: train data with all features except target
        :param y_train: train data with target feature
        :param X_test: test data with all features except target
        :param y_test: test data with target feature
        """
        if model == 'LogisticRegression':
            model_obj = LogisticRegression(max_iter=10000, C=1, solver='saga', penalty='l1')
        elif model == 'AdaBoost':
            model_obj = AdaBoostClassifier(n_estimators=50)
        else:
            model = 'SVM'
            model_obj = SVC(C=1, probability=True, kernel='rbf')

        model_obj.fit(X_train, y_train)
        train_predict = model_obj.predict(X_train)
        test_predict = model_obj.predict(X_test)

        train_accuracy = accuracy_score(y_train, train_predict)
        print(f"\nTraining Accuracy {model}: {train_accuracy * 100:.2f}%")
        test_accuracy = accuracy_score(y_test, test_predict)
        print(f"Testing Accuracy {model}: {test_accuracy * 100:.2f}%")

        train_precision = precision_score(y_train, train_predict)
        print(f"\nTraining Precision {model}: {train_precision * 100:.2f}%")
        test_precision = precision_score(y_test, test_predict)
        print(f"Testing Precision {model}: {test_precision * 100:.2f}%")

        train_recall = recall_score(y_train, train_predict)
        print(f"\nTraining Recall {model}: {train_recall * 100:.2f}%")
        test_recall = recall_score(y_test, test_predict)
        print(f"Testing Recall {model}: {test_recall * 100:.2f}%")

        train_f1 = f1_score(y_train, train_predict)
        print(f"\nTraining F1 score {model}: {train_f1 * 100:.2f}%")
        test_f1 = f1_score(y_test, test_predict)
        print(f"Testing F1 score {model}: {test_f1 * 100:.2f}%")

        fpr, tpr, _ = roc_curve(y_test, model_obj.predict_proba(X_test)[:, 1])
        test_auc_score = roc_auc_score(y_test, model_obj.predict_proba(X_test)[:, 1])
        print(f"\nTesting AUC score {model}: {test_auc_score * 100:.2f}%")

        # Appending false positive, true positive and auc score values
        fprs.append(fpr)
        tprs.append(tpr)
        aucs.append(test_auc_score)

        return fprs, tprs, aucs

    def execute_process(self):
        """
        Method to call all the methods of the file
        """
        self.data_cleaning()
        self.data_encoding()
        self.print_data()

        X_train, X_test, y_train, y_test = self.split_data()
        self.data_scaling([X_train, X_test])

        X_oversampled, y_oversampled = self.oversample_data(X_train, y_train)

        model_names = ['LogisticRegression', 'AdaBoost', 'SVM']
        for model in model_names:
            fprs, tprs, aucs = self.ml_model(model, X_oversampled, y_oversampled, X_test, y_test)

        self.plot_auroc(fprs, tprs, aucs, model_names)


fprs, tprs, aucs = [], [], []
telecom_predictor = TelecomChurnPredictor("Telecom Churn Prediction.csv")
telecom_predictor.execute_process()
