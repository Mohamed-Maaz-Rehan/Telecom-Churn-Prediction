import pandas as pd
import DataSplit
import Models
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('Telecom Churn Prediction.csv')

ftData = DataSplit.dtreformat(df)
X_train, X_test, Y_train, Y_test = DataSplit.datasplit(ftData, 'Churn')
X_oversampled, y_oversampled = DataSplit.smote(X_train, Y_train)
X_undersampled, y_undersampled = DataSplit.RUS(X_train, Y_train)
X_combined, y_combined = DataSplit.combine(X_train, Y_train)

classifiers = [
    LogisticRegression(max_iter=1000),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(n_estimators=50),  # AdaBoost with 50 base estimators
    SVC(probability=True),
    #KNeighborsClassifier()
]

print("\n-------------Oversampled------------------\n")
results_df = Models.models(X_oversampled, y_oversampled, X_test, Y_test, classifiers)
print(results_df.to_string())

print("\n-------------Undersampled-----------------\n")
results_df = Models.models(X_undersampled, y_undersampled, X_test, Y_test, classifiers)
print(results_df.to_string())

print("\n-------------Unbalanced------------------\n")
results_df = Models.models(X_train, Y_train, X_test, Y_test, classifiers)
print(results_df.to_string())

print("\n-------------Combined------------------\n")
results_df = Models.models(X_combined, y_combined, X_test, Y_test, classifiers)
print(results_df.to_string())



