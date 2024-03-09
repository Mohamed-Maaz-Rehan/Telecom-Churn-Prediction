import pandas as pd
import DataSplit
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score

df = pd.read_csv('Telecom Churn Prediction.csv')

ftData = DataSplit.dtreformat(df)
X_train, X_test, Y_train, Y_test = DataSplit.datasplit(ftData, 'Churn')
X_oversampled, y_oversampled = DataSplit.smote(X_train, Y_train)
X_undersampled, y_undersampled = DataSplit.RUS(X_train, Y_train)
X_combined, y_combined = DataSplit.combine(X_train, Y_train)
print(Y_train.value_counts())
print(y_combined.value_counts())


def models(x_train, y_train, x_test, y_test):
    classifiers = [
        LogisticRegression(max_iter=1000),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        AdaBoostClassifier(n_estimators=50),  # AdaBoost with 50 base estimators
        SVC(probability=True),
        KNeighborsClassifier()
    ]

    results_dict = {'Classifier': [], 'Training Accuracy': [], 'Testing Accuracy': [],
                    'Training Precision': [], 'Testing Precision': [],
                    'Training Recall': [], 'Testing Recall': [],
                    'Training F1': [], 'Testing F1': []}

    for i, clf in enumerate(classifiers):
        clf.fit(x_train, y_train)

        train_predictions = clf.predict(x_train)
        train_accuracy = accuracy_score(y_train, train_predictions)
        train_precision = precision_score(y_train, train_predictions)
        train_recall = recall_score(y_train, train_predictions)
        train_f1 = f1_score(y_train, train_predictions)

        test_predictions = clf.predict(x_test)
        test_accuracy = accuracy_score(y_test, test_predictions)
        test_precision = precision_score(y_test, test_predictions)
        test_recall = recall_score(y_test, test_predictions)
        test_f1 = f1_score(y_test, test_predictions)

        results_dict['Classifier'].append(clf.__class__.__name__)
        results_dict['Training Accuracy'].append(train_accuracy)
        results_dict['Testing Accuracy'].append(test_accuracy)
        results_dict['Training Precision'].append(train_precision)
        results_dict['Testing Precision'].append(test_precision)
        results_dict['Training Recall'].append(train_recall)
        results_dict['Testing Recall'].append(test_recall)
        results_dict['Training F1'].append(train_f1)
        results_dict['Testing F1'].append(test_f1)

    results_df = pd.DataFrame(results_dict)
    print(results_df.to_string())


print("\n-------------Oversampled------------------\n")
models(X_oversampled, y_oversampled, X_test, Y_test)
print("\n-------------Undersampled-----------------\n")
models(X_undersampled, y_undersampled, X_test, Y_test)
print("\n-------------Unbalanced------------------\n")
models(X_train, Y_train, X_test, Y_test)
print("\n-------------Combined------------------\n")
models(X_combined, y_combined, X_test, Y_test)
