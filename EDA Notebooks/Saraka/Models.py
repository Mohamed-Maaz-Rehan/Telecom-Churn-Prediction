import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, roc_auc_score, roc_curve
import warnings
import pickle
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

fixed_column_order = ['tenure', 'MonthlyCharges', 'TotalCharges',
                      'PaperlessBilling_No', 'PaperlessBilling_Yes',
                      'Partner_No', 'Partner_Yes',
                      'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No',
                      'OnlineBackup_No', 'OnlineBackup_No internet service', 'OnlineBackup_Yes',
                      'Dependents_No', 'Dependents_Yes',
                      'SeniorCitizen_No', 'SeniorCitizen_Yes',
                      'DeviceProtection_No', 'DeviceProtection_No internet service', 'DeviceProtection_Yes',
                      'OnlineSecurity_No', 'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
                      'TechSupport_No', 'TechSupport_No internet service', 'TechSupport_Yes',
                      'Contract_Month-to-month', 'Contract_One year', 'Contract_Two year',
                      'StreamingTV_No', 'StreamingTV_No internet service', 'StreamingTV_Yes',
                      'PaymentMethod_Bank transfer (automatic)', 'PaymentMethod_Credit card (automatic)',
                      'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check']


def models(x_train, y_train, x_test, y_test):
    x_train = x_train[fixed_column_order]
    x_test = x_test[fixed_column_order]

    classifiers = [
        LogisticRegression(C=1, solver='saga', penalty='l1', max_iter=10000),
        AdaBoostClassifier(n_estimators=50),  # AdaBoost with 50 base estimators
        SVC(C=1, probability=True, kernel='poly')
    ]

    results_dict = {'Classifier': [], 'Training Accuracy': [], 'Testing Accuracy': [],
                    'Training Precision': [], 'Testing Precision': [],
                    'Training Recall': [], 'Testing Recall': [],
                    'Training F1': [], 'Testing F1': [], 'AUC': []}

    fprs, tprs, aucs = [], [], []
    classifier_names = []
    shap_values_list = []

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

        if isinstance(clf, LogisticRegression):
            # Coefficients
            coefficients = clf.coef_[0]  # Extracting coefficients for binary classification
            feature_names = x_train.columns


            # Create a DataFrame containing the coefficients
            coef_df = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': coefficients
            })

            # Sort the DataFrame by the absolute values of the coefficients for better visualization
            coef_df = coef_df.sort_values(by='Coefficient', ascending=False)

            # Plotting using Seaborn
            plt.figure(figsize=(16, 10))
            sns.barplot(data=coef_df, x='Coefficient', y='Feature')
            plt.title('Feature Importance in Logistic Regression')
            plt.xlabel('Coefficient Value')
            plt.ylabel('Features')
            plt.yticks(fontsize=6)
            plt.show()

        # if isinstance(clf, (AdaBoostClassifier, SVC)):
        #     background = shap.kmeans(x_train, 20)  # Summarize using 10 clusters
        #     explainer = shap.KernelExplainer(clf.predict_proba, background)
        #     # Generate SHAP values
        #     shap_values = explainer(x_train.iloc[:50])
        #     shap.summary_plot(shap_values[:,:,1])
        #     shap.plots.bar(shap_values[:,:,1],max_display=70)

    #Outputs.plot_auroc(fprs, tprs, aucs, classifier_names)

    results_df = pd.DataFrame(results_dict)
    return results_df


def save_model(clf, cls):
    with open(f'{cls}.pickle', 'wb') as f:
        pickle.dump(clf, f)
