import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from sklearn.feature_selection import SelectKBest, chi2

import warnings

warnings.filterwarnings("ignore")


def models(x_train, y_train, x_test, y_test):
    classifiers = [
        LogisticRegression(max_iter=10000),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        AdaBoostClassifier(n_estimators=50),  # AdaBoost with 50 base estimators
        SVC(probability=True),
        #KNeighborsClassifier()
    ]

    results_dict = {'Classifier': [], 'Training Accuracy': [], 'Testing Accuracy': [],
                    'Training Precision': [], 'Testing Precision': [],
                    'Training Recall': [], 'Testing Recall': [],
                    'Training F1': [], 'Testing F1': []}

    for i, clf in enumerate(classifiers):
        clf.fit(x_train, y_train)

        k_best = SelectKBest(score_func=chi2, k=15)  # Selecting 15 best features using chi-square test
        x_train_selected = k_best.fit_transform(x_train, y_train)
        x_test_selected = k_best.transform(x_test)

        clf.fit(x_train_selected, y_train)

        if isinstance(clf, KNeighborsClassifier):
            train_predictions = clf.predict(x_train.values)
        else:
            train_predictions = clf.predict(x_train_selected)
        train_accuracy = accuracy_score(y_train, train_predictions)
        train_precision = precision_score(y_train, train_predictions)
        train_recall = recall_score(y_train, train_predictions)
        train_f1 = f1_score(y_train, train_predictions)

        if isinstance(clf, KNeighborsClassifier):
            test_predictions = clf.predict(x_test.values)
        else:
            test_predictions = clf.predict(x_test_selected)
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
    return results_df
