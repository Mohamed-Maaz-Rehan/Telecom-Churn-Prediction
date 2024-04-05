X_train

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
import pandas as pd 

y_train

# Fit it to the training data
grid_search_lr.fit(X_train, y_train)

# Fit it to the training data
grid_search_rf.fit(X_train, y_train)

# Defining the parameter grid for Gradient Boosting
param_grid_gb = {
    'gradientboostingclassifier__n_estimators': [100, 200],
    'gradientboostingclassifier__learning_rate': [0.01, 0.1],
    'gradientboostingclassifier__max_depth': [3, 5]
}

# Create a GridSearchCV object for the Gradient Boosting pipeline
grid_search_gb = GridSearchCV(gb_pipeline, param_grid=param_grid_gb, cv=5, scoring='roc_auc', verbose=1)

# Fit it to the training data
grid_search_gb.fit(X_train, y_train)

# Best parameters and best score
print("Best parameters for Gradient Boosting:", grid_search_gb.best_params_)
print("Best ROC-AUC score from Grid Search for Gradient Boosting:", grid_search_gb.best_score_)