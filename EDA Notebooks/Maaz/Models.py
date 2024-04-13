import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

def logistic_regression(X_train,y_train,X_test,y_test):
    # creating object
    model = LogisticRegression()
    #train the model
    model.fit(X_train,y_train)
    
    #make prediction
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    #calculate accuracy
    train_accuracy = accuracy_score(y_train,y_pred_train)
    test_accuracy = accuracy_score(y_test,y_pred_test)
    print(train_accuracy)
    print(test_accuracy)

    #confusion matrix
    train_conf_matrix = confusion_matrix(y_train,y_pred_train)
    print(train_conf_matrix)
    test_conf_matrix = confusion_matrix(y_test,y_pred_test)
    print(test_conf_matrix)

    # AUC ROC

    def plot_roc_curve(fpr, tpr):
        plt.plot(fpr, tpr, color='orange', lw=2, linestyle='--')
        plt.plot([0, 1], [0, 1], color='darkblue', linestyle=':')
        plt.xlabel('False Positive Rate(1-specificity)')
        plt.ylabel('True Positive Rate (sensitivity)')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        plt.show()

    def get_summary(y_test, y_pred_logreg):
        # Confusion Matrix
        conf_mat = confusion_matrix(y_test, y_pred_logreg)
        TP = conf_mat[0, 0:1]
        FP = conf_mat[0, 1:2]
        FN = conf_mat[1, 0:1]
        TN = conf_mat[1, 1:2]

        accuracy = (TP + TN) / ((FN + FP) + (TP + TN))
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        fScore = (2 * recall * precision) / (recall + precision)
        auc = roc_auc_score(y_test, y_pred_logreg)

        print("Confusion Matrix:\n", conf_mat)
        print("Accuracy:", accuracy)
        print("Sensitivity :", sensitivity)
        print("Specificity :", specificity)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F-score:", fScore)
        print("AUC:", auc)
        print("ROC curve:")
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_logreg)
        plot_roc_curve(fpr, tpr)
    get_summary(y_test,y_pred_test)

    return train_accuracy,test_accuracy,train_conf_matrix,test_conf_matrix



# Support Vector Machine

def SVM(X_train, y_train, X_test, y_test):
    # creating object
    model = SVC()
    # train the model
    model.fit(X_train, y_train)

    # make prediction
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # calculate accuracy
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    print(train_accuracy)
    print(test_accuracy)

    # confusion matrix
    train_conf_matrix = confusion_matrix(y_train, y_pred_train)
    print(train_conf_matrix)
    test_conf_matrix = confusion_matrix(y_test, y_pred_test)
    print(test_conf_matrix)

    return train_accuracy, test_accuracy, train_conf_matrix, test_conf_matrix

# Naive Baye's

def gnb(X_train, y_train, X_test, y_test):
    # creating object
    model = GaussianNB()
    # train the model
    model.fit(X_train, y_train)

    # make prediction
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # calculate accuracy
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    print(train_accuracy)
    print(test_accuracy)

    # confusion matrix
    train_conf_matrix = confusion_matrix(y_train, y_pred_train)
    print(train_conf_matrix)
    test_conf_matrix = confusion_matrix(y_test, y_pred_test)
    print(test_conf_matrix)

    return train_accuracy, test_accuracy, train_conf_matrix, test_conf_matrix

# KNN Algorithm

def knn(X_train, y_train, X_test, y_test):
    # creating object
    model = KNeighborsClassifier()
    # train the model
    model.fit(X_train, y_train)

    # make prediction
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # calculate accuracy
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    print(train_accuracy)
    print(test_accuracy)

    # confusion matrix
    train_conf_matrix = confusion_matrix(y_train, y_pred_train)
    print(train_conf_matrix)
    test_conf_matrix = confusion_matrix(y_test, y_pred_test)
    print(test_conf_matrix)

    return train_accuracy, test_accuracy, train_conf_matrix, test_conf_matrix

# Decision Tree Algorithm

def DT(X_train, y_train, X_test, y_test):
    # creating object
    model = DecisionTreeClassifier()
    # train the model
    model.fit(X_train, y_train)

    # make prediction
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # calculate accuracy
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    print(train_accuracy)
    print(test_accuracy)

    # confusion matrix
    train_conf_matrix = confusion_matrix(y_train, y_pred_train)
    print(train_conf_matrix)
    test_conf_matrix = confusion_matrix(y_test, y_pred_test)
    print(test_conf_matrix)

    return train_accuracy, test_accuracy, train_conf_matrix, test_conf_matrix

# Random Forest Classifier

def RF(X_train, y_train, X_test, y_test):
    # creating object
    model = RandomForestClassifier()
    # train the model
    model.fit(X_train, y_train)

    # make prediction
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # calculate accuracy
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    print(train_accuracy)
    print(test_accuracy)

    # confusion matrix
    train_conf_matrix = confusion_matrix(y_train, y_pred_train)
    print(train_conf_matrix)
    test_conf_matrix = confusion_matrix(y_test, y_pred_test)
    print(test_conf_matrix)

    return train_accuracy, test_accuracy, train_conf_matrix, test_conf_matrix

# AdaBoost Algorithm

def AdaBoost(X_train, y_train, X_test, y_test):
    # creating object
    model = AdaBoostClassifier()
    # train the model
    model.fit(X_train, y_train)

    # make prediction
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # calculate accuracy
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    print(train_accuracy)
    print(test_accuracy)

    # confusion matrix
    train_conf_matrix = confusion_matrix(y_train, y_pred_train)
    print(train_conf_matrix)
    test_conf_matrix = confusion_matrix(y_test, y_pred_test)
    print(test_conf_matrix)

    return train_accuracy, test_accuracy, train_conf_matrix, test_conf_matrix


## Gradient Boosting
def gb(X_train, y_train, X_test, y_test):
    # creating object
    model = GradientBoostingClassifier()
    # train the model
    model.fit(X_train, y_train)

    # make prediction
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # calculate accuracy
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    print(train_accuracy)
    print(test_accuracy)

    # confusion matrix
    train_conf_matrix = confusion_matrix(y_train, y_pred_train)
    print(train_conf_matrix)
    test_conf_matrix = confusion_matrix(y_test, y_pred_test)
    print(test_conf_matrix)

    return train_accuracy, test_accuracy, train_conf_matrix, test_conf_matrix


## XGBoost Algorithm
def xgb(X_train, y_train, X_test, y_test):
    # creating object
    model = XGBClassifier()
    # train the model
    model.fit(X_train, y_train)

    # make prediction
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # calculate accuracy
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    print(train_accuracy)
    print(test_accuracy)

    # confusion matrix
    train_conf_matrix = confusion_matrix(y_train, y_pred_train)
    print(train_conf_matrix)
    test_conf_matrix = confusion_matrix(y_test, y_pred_test)
    print(test_conf_matrix)

    return train_accuracy, test_accuracy, train_conf_matrix, test_conf_matrix

# Regularization Techniques

# Lasso

def lasso(X_train, y_train, X_test, y_test):
    # creating object
    model = Lasso(alpha=0.001)
    # train the model
    model.fit(X_train, y_train)

    # make prediction
    y_pred_train = model.predict(X_train).round()
    y_pred_test = model.predict(X_test).round()

    # calculate accuracy
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    print(train_accuracy)
    print(test_accuracy)

    # confusion matrix
    train_conf_matrix = confusion_matrix(y_train, y_pred_train)
    print(train_conf_matrix)
    test_conf_matrix = confusion_matrix(y_test, y_pred_test)
    print(test_conf_matrix)

    return train_accuracy, test_accuracy, train_conf_matrix, test_conf_matrix

# Ridge
def ridge(X_train, y_train, X_test, y_test):
    # creating object
    model = Ridge()
    # train the model
    model.fit(X_train, y_train)

    # make prediction
    y_pred_train = model.predict(X_train).round()
    y_pred_test = model.predict(X_test).round()

    # calculate accuracy
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    print(train_accuracy)
    print(test_accuracy)

    # confusion matrix
    train_conf_matrix = confusion_matrix(y_train, y_pred_train)
    print(train_conf_matrix)
    test_conf_matrix = confusion_matrix(y_test, y_pred_test)
    print(test_conf_matrix)

    return train_accuracy, test_accuracy, train_conf_matrix, test_conf_matrix

# Elastic Net
def elasticnet(X_train, y_train, X_test, y_test):
    # creating object
    model = ElasticNet()
    # train the model
    model.fit(X_train, y_train)

    # make prediction
    y_pred_train = model.predict(X_train).round()
    y_pred_test = model.predict(X_test).round()

    # calculate accuracy
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    print(train_accuracy)
    print(test_accuracy)

    # confusion matrix
    train_conf_matrix = confusion_matrix(y_train, y_pred_train)
    print(train_conf_matrix)
    test_conf_matrix = confusion_matrix(y_test, y_pred_test)
    print(test_conf_matrix)

    return train_accuracy, test_accuracy, train_conf_matrix, test_conf_matrix
