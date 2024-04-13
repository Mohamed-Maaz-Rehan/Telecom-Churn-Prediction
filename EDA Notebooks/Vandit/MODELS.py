import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, roc_auc_score
import warnings

warnings.filterwarnings("ignore")


def models(X_train, y_train, X_test, y_test):
    classifiers = [
        ('Logistic Regression', LogisticRegression(solver='saga', penalty='l1', max_iter=10000)),
        ('K-Nearest Neighbors', KNeighborsClassifier()),
        ('CART (Decision Tree Classifier)', DecisionTreeClassifier()),
        ('SVC', SVC(C=1, probability=True, kernel='rbf')),
    ]

    for name, clf in classifiers:
        try:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_proba)

            print(f"Results for {name}:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
            print(f"ROC AUC Score: {roc_auc:.4f}\n")
        except Exception as e:
            print(f"Error processing {name}: {e}")


