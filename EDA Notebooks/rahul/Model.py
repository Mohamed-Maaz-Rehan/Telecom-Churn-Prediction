import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, roc_auc_score, roc_curve
import shap
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


def models(x_train, y_train, x_test, y_test):
    classifiers = [
        LogisticRegression(solver='liblinear', penalty='l1', max_iter=10000),
        RandomForestClassifier(),
        SVC(C=1, probability=True, kernel='rbf'),
        DecisionTreeClassifier()
    ]

    results_dict = {'Classifier': [], 'Training Accuracy': [], 'Testing Accuracy': [],
                    'Training Precision': [], 'Testing Precision': [],
                    'Training Recall': [], 'Testing Recall': [],
                    'Training F1': [], 'Testing F1': [], 'AUC': []}

    fprs, tprs, aucs = [], [], []
    classifier_names = []

    for clf in classifiers:
        classifier_name = clf.__class__.__name__

        clf.fit(x_train, y_train)

        if isinstance(clf, LogisticRegression):
            feature_names = x_train.columns.tolist()
            print(feature_names)
            coefficients = dict(zip(feature_names, clf.coef_))
            print("Weights for features:")
            for feature, weight in coefficients.items():
                print(f"{feature}: {weight}")

            explainer = shap.Explainer(clf, x_train)

            shap_values = explainer.shap_values(x_test)
            print("Variable Importance Plot - Global Interpretation")
            figure = plt.figure()
            shap.summary_plot(shap_values, x_test)
            plt.title(f'SHAP Values for {classifier_name}')
            plt.savefig(f'SHAP_{classifier_name}.png', bbox_inches='tight')
            plt.close()

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

        fpr, tpr, _ = roc_curve(y_test, clf.predict_proba(x_test)[:, 1])
        auc_score = roc_auc_score(y_test, clf.predict_proba(x_test)[:, 1])
        fprs.append(fpr)
        tprs.append(tpr)
        aucs.append(auc_score)
        classifier_names.append(classifier_name)

        df = pd.DataFrame(x_train)
        df['actual'] = y_train
        df['predicted'] = train_predictions

        incorrect = df[df['actual'] != df['predicted']]
        incorrect.to_csv(f'FP_{classifier_name}.csv', index=False)

        results_dict['Classifier'].append(classifier_name)
        results_dict['Training Accuracy'].append(train_accuracy)
        results_dict['Testing Accuracy'].append(test_accuracy)
        results_dict['Training Precision'].append(train_precision)
        results_dict['Testing Precision'].append(test_precision)
        results_dict['Training Recall'].append(train_recall)
        results_dict['Testing Recall'].append(test_recall)
        results_dict['Training F1'].append(train_f1)
        results_dict['Testing F1'].append(test_f1)
        results_dict['AUC'].append(auc_score)

    results_df = pd.DataFrame(results_dict)
    return results_df
