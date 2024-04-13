import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer


def train_and_evaluate_models():
    # Load the dataset
    df = pd.read_csv('D:\\Telecom-Churn-Prediction\\Telecom Churn Prediction.csv')
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=40, stratify=y)

    # Train a Decision Tree model
    dt_model = DecisionTreeClassifier()
    dt_model.fit(X_train, y_train)
    predictdt_y = dt_model.predict(X_test)
    accuracy_dt = accuracy_score(y_test, predictdt_y)
    print(f"Decision Tree accuracy is: {accuracy_dt:.4f}")

    # Setup hyperparameter grid for RandomForest
    rf_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt']
    }

    # Initialize and fit GridSearchCV for RandomForest
    rf = RandomForestClassifier(random_state=42)
    grid_search_rf = GridSearchCV(estimator=rf, param_grid=rf_param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
    grid_search_rf.fit(X_train, y_train)

    # Display the best parameters and score from GridSearch
    best_params_rf = grid_search_rf.best_params_
    best_score_rf = grid_search_rf.best_score_
    print(f"Best GridSearch parameters: {best_params_rf}")
    print(f"Best GridSearch score: {best_score_rf:.4f}")

    # Perform 10-fold cross-validation for RandomForest
    rf_classifier = RandomForestClassifier(**best_params_rf, random_state=42)  # Using best parameters
    scores = cross_val_score(rf_classifier, X, y, cv=10)
    print("Accuracy scores for each fold:", scores)
    mean_score = scores.mean()
    print(f"\nMean accuracy across 10 folds: {mean_score:.4f}")

if __name__ == "__main__":
    train_and_evaluate_models()

