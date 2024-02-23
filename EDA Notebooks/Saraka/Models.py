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
X_train, X_test, y_train, y_test = DataSplit.datasplit(ftData, 'Churn')
X_oversampled, y_oversampled = DataSplit.smote(X_train, y_train)
X_undersampled, y_undersampled = DataSplit.RUS(X_train, y_train)


def models(x_resampled, y_resampled):
    logistic_model = LogisticRegression(max_iter=1000)
    logistic_model.fit(x_resampled, y_resampled)
    logistic_predictions = logistic_model.predict(X_test)
    logistic_accuracy = accuracy_score(y_test, logistic_predictions)
    print(f'Logistic Regression Accuracy: {logistic_accuracy}')

    # Decision Tree
    tree_model = DecisionTreeClassifier()
    tree_model.fit(x_resampled, y_resampled)
    tree_predictions = tree_model.predict(X_test)
    tree_accuracy = accuracy_score(y_test, tree_predictions)
    print(f'Decision Tree Accuracy: {tree_accuracy}')

    # Random Forest
    forest_model = RandomForestClassifier()
    forest_model.fit(x_resampled, y_resampled)
    forest_predictions = forest_model.predict(X_test)
    forest_accuracy = accuracy_score(y_test, forest_predictions)
    print(f'Random Forest Accuracy: {forest_accuracy}')

    # Support Vector Machine (SVM)
    svm_model = SVC()
    svm_model.fit(x_resampled, y_resampled)
    svm_predictions = svm_model.predict(X_test)
    svm_accuracy = accuracy_score(y_test, svm_predictions)
    print(f'SVM Accuracy: {svm_accuracy}')

    # K-Nearest Neighbors (KNN)
    knn_model = KNeighborsClassifier()
    knn_model.fit(x_resampled, y_resampled)
    knn_predictions = knn_model.predict(X_test)
    knn_accuracy = accuracy_score(y_test, knn_predictions)
    print(f'KNN Accuracy: {knn_accuracy}')


print("-------------Oversampled------------------")
models(X_oversampled, y_oversampled)
print("-------------Undersampled-----------------")
models(X_undersampled, y_undersampled)
print("-------------Unbalanced------------------")
models(X_train, y_train)
