import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, roc_auc_score, roc_curve
import warnings
import pickle

import DataSplit

warnings.filterwarnings("ignore")


def models(x_train, y_train, x_test, y_test):
    classifiers = [
        (LogisticRegression(C=1,solver='saga', penalty='l1', max_iter=10000)),
        #RandomForestClassifier(),
        #AdaBoostClassifier(n_estimators=50),  # AdaBoost with 50 base estimators
        #SVC(C=1, probability=True, kernel='rbf'),
    ]

    results_dict = {'Classifier': [], 'Training Accuracy': [], 'Testing Accuracy': [],
                    'Training Precision': [], 'Testing Precision': [],
                    'Training Recall': [], 'Testing Recall': [],
                    'Training F1': [], 'Testing F1': [], 'AUC': []}

    fprs, tprs, aucs = [], [], []
    classifier_names = []

    #x_train, x_test = DataSplit.skBest(x_train,x_test,y_train)
    #print(x_train)
    for i, clf in enumerate(classifiers):
        if isinstance(clf, LogisticRegression):
            penalty = clf.get_params()['penalty'].upper()
            classifier_name = f"LogisticRegression({penalty})"
        else:
            classifier_name = clf.__class__.__name__

        clf.fit(x_train, y_train)

        save_model(clf, classifier_name)
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
        #incorrect.to_csv(f'FP{classifier_name}.csv', index=False)

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

    # Outputs.plot_auroc(fprs, tprs, aucs, classifier_names)
    results_df = pd.DataFrame(results_dict)
    return results_df


def save_model(clf, cls):
    with open(f'{cls}.pickle', 'wb') as f:
        pickle.dump(clf, f)