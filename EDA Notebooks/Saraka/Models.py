import pandas as pd
import DataSplit
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('Telecom Churn Prediction.csv')

ftData = DataSplit.dtreformat(df)
X_train, X_test, Y_train, Y_test = DataSplit.datasplit(ftData, 'Churn')
X_oversampled, y_oversampled = DataSplit.smote(X_train, Y_train)
X_undersampled, y_undersampled = DataSplit.RUS(X_train, Y_train)


def models(x_train, y_train, x_test, y_test):
    classifiers = [
        LogisticRegression(max_iter=1000),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        SVC(),
        KNeighborsClassifier()
    ]

    results_dict = {'Classifier': [], 'Training Accuracy': [], 'Testing Accuracy': []}

    for clf in classifiers:
        clf.fit(x_train, y_train)

        train_predictions = clf.predict(x_train)
        train_accuracy = accuracy_score(y_train, train_predictions)

        test_predictions = clf.predict(x_test)
        test_accuracy = accuracy_score(y_test, test_predictions)

        results_dict['Classifier'].append(clf.__class__.__name__)
        results_dict['Training Accuracy'].append(train_accuracy)
        results_dict['Testing Accuracy'].append(test_accuracy)

    results_df = pd.DataFrame(results_dict)
    print(results_df)


print("\n-------------Oversampled------------------\n")
models(X_oversampled, y_oversampled, X_test, Y_test)
print("\n-------------Undersampled-----------------\n")
models(X_undersampled, y_undersampled, X_test, Y_test)
print("\n-------------Unbalanced------------------\n")
models(X_train, Y_train, X_test, Y_test)
