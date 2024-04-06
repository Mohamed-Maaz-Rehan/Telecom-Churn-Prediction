import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import Models
from imblearn.over_sampling import SMOTE

df = pd.read_csv('TelecomChurn.csv')

# split the data into training set and testing set

from sklearn.model_selection import train_test_split

def traintestsplit(df):
    X = df.drop(['Churn'],axis = 1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=300)
    return X_train,X_test,y_train,y_test


# Scaling and transformation

from sklearn.preprocessing import MinMaxScaler

def scale(X_train,X_test):
    col = 'tenure','MonthlyCharges','TotalCharges'
    scaler = MinMaxScaler()
    for i in col:
        X_train[[i]] = scaler.fit_transform(X_train[[i]])
        X_test[[i]] = scaler.transform(X_test[[i]])
        return X_train,X_test

# Oversampling
def smote(X,y):
    smt = SMOTE()
    X_oversampled, y_oversampled = smt.fit_resample(X,y)
    return X_oversampled, y_oversampled

# Logistic Regression

X_train, X_test, y_train, y_test = traintestsplit(df)
X_train_scaled, X_test_scaled = scale(X_train, X_test)
X_oversampled,y_oversampled = smote(X_train_scaled,y_train)

## unbalanced - Logistic Regression
Models.logistic_regression(X_train_scaled,y_train,X_test_scaled,y_test)

## Balanced - Logistic Regression
Models.logistic_regression(X_oversampled,y_oversampled,X_test_scaled,y_test)

# Support Vector Machine

## Unbalanced - SVM
Models.SVM(X_train_scaled,y_train,X_test_scaled,y_test)

## Balanced - SVM
Models.SVM(X_oversampled,y_oversampled,X_test_scaled,y_test)

# Naive Baye's

## Unbalanced - SVM
Models.gnb(X_train_scaled,y_train,X_test_scaled,y_test)

## Balanced - SVM
Models.gnb(X_oversampled,y_oversampled,X_test_scaled,y_test)

# KNN Algorithm

## Unbalanced - SVM
Models.knn(X_train_scaled,y_train,X_test_scaled,y_test)

## Balanced - SVM
Models.knn(X_oversampled,y_oversampled,X_test_scaled,y_test)

# Decision Tree Algorithm

## Unbalanced - SVM
Models.DT(X_train_scaled,y_train,X_test_scaled,y_test)

## Balanced - SVM
Models.DT(X_oversampled,y_oversampled,X_test_scaled,y_test)

# Random Forest Classifier
## Unbalanced - SVM
Models.RF(X_train_scaled,y_train,X_test_scaled,y_test)

## Balanced - SVM
Models.RF(X_oversampled,y_oversampled,X_test_scaled,y_test)




