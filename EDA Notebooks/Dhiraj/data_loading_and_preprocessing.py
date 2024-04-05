import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings('ignore')

#splitting data into train and test, testing data contains 20%
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Preprocessing for numeric data: imputation + scaling
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

# Preprocessing for categorical data: imputation + one-hot encoding
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Bundle preprocessing for numeric and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)])

# Creating a pipeline with preprocessing and the logistic regression model
lr_pipeline = make_pipeline(preprocessor, LogisticRegression(solver='liblinear'))

# Creating a pipeline with preprocessing and the random forest model
rf_pipeline = make_pipeline(preprocessor, RandomForestClassifier(n_estimators=100))

from sklearn.model_selection import GridSearchCV

# Creating and training the Logistic Regression model with the best parameters
lr_pipeline_best = make_pipeline(preprocessor, LogisticRegression(C=10, solver='liblinear'))

# Creating and training the Random Forest model with the best parameters
rf_pipeline_best = make_pipeline(preprocessor, RandomForestClassifier(max_depth=10, n_estimators=100))

# Creating a pipeline with preprocessing and the Gradient Boosting classifier
gb_pipeline = make_pipeline(preprocessor, GradientBoostingClassifier(random_state=42))