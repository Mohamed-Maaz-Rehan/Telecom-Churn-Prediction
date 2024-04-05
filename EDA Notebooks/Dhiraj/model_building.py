import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score



# Training the model
lr_pipeline.fit(X_train, y_train)

# Training the model
rf_pipeline.fit(X_train, y_train)

def evaluate_model(y_true, y_pred, y_pred_proba):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label='Yes')  # Adjust if necessary
    recall = recall_score(y_true, y_pred, pos_label='Yes')
    f1 = f1_score(y_true, y_pred, pos_label='Yes')
    roc_auc = roc_auc_score(y_true, y_pred_proba)  # Use probabilities here
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")

# Evaluate Logistic Regression Model
print("Logistic Regression Performance:")
evaluate_model(y_test, y_pred_lr, y_pred_proba_lr)

# Evaluate Random Forest Model
print("\nRandom Forest Performance:")
evaluate_model(y_test, y_pred_rf, y_pred_proba_rf)

# Training the model
lr_pipeline_best.fit(X_train, y_train)

# Evaluating the model
print("Logistic Regression Performance with Best Parameters:")
evaluate_model(y_test, y_pred_lr_best, y_pred_proba_lr_best)

# Training the model
rf_pipeline_best.fit(X_train, y_train)

# Evaluating the model
print("\nRandom Forest Performance with Best Parameters:")
evaluate_model(y_test, y_pred_rf_best, y_pred_proba_rf_best)

# Training the Gradient Boosting model
gb_pipeline.fit(X_train, y_train)

# Evaluating the Gradient Boosting model
print("Gradient Boosting Performance:")
evaluate_model(y_test, y_pred_gb, y_pred_proba_gb)