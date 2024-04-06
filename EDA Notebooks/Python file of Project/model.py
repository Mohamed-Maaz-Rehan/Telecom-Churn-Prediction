from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train,y_train)
predictdt_y = dt_model.predict(X_test)
accuracy_dt = dt_model.score(X_test,y_test)
print("Decision Tree accuracy is :",accuracy_dt)
# Creating a random forest classifier object 'temp_rf' with a random state of 0 and parallel processing enabled
temp_rf=RandomForestClassifier(random_state=0,n_jobs=-1)

# Creating a grid search object 'grid_search' using the 'GridSearchCV' function, with a random forest classifier as the estimator, a parameter grid, 'roc_auc' as the scoring metric, and 5-fold cross-validation with parallel processing
grid_search=GridSearchCV(estimator=temp_rf, param_grid=param_grid, scoring='roc_auc', cv=5, n_jobs=-1)
from sklearn.model_selection import GridSearchCV

# Random Forest hyperparameter grid
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['sqrt']
}

# Initialize the Random Forest model
rf = RandomForestClassifier(random_state=42)

# Setup GridSearchCV
grid_search_rf = GridSearchCV(estimator=rf, param_grid=rf_param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)

# Fit GridSearchCV
grid_search_rf.fit(X_train, y_train)

# Best parameters and score
best_params_rf = grid_search_rf.best_params_
best_score_rf = grid_search_rf.best_score_

best_params_rf, best_score_rf


# Assuming X and y are your features and target variable
# Initialize the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)

# Perform 10-fold cross-validation
scores = cross_val_score(rf_classifier, X, y, cv=10)

# Display the scores for each fold
print("Accuracy scores for each fold:")
print(scores)

# Calculate the average score across all folds
mean_score = scores.mean()
print(f"\nMean accuracy across 10 folds: {mean_score:.4f}")
