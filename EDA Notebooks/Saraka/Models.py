import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score


def models(x_train, y_train, x_test, y_test, classifiers):
    results_dict = {'Classifier': [], 'Training Accuracy': [], 'Testing Accuracy': [],
                    'Training Precision': [], 'Testing Precision': [],
                    'Training Recall': [], 'Testing Recall': [],
                    'Training F1': [], 'Testing F1': [], 'Model': []}

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

        if isinstance(clf, LogisticRegression):
            coefficients = clf.coef_
            intercept = clf.intercept_
            print(f"Logistic Regression Coefficients (Weights): {coefficients}")
            print(f"Logistic Regression Intercept: {intercept}")

        results_dict['Model'].append(clf)
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
