import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Predicting on the test set
y_pred_lr = lr_pipeline.predict(X_test)

# predicting For Logistic Regression
y_pred_proba_lr = lr_pipeline.predict_proba(X_test)[:, 1]  # Getting probabilities for the positive class

# Predicting on the test set
y_pred_rf = rf_pipeline.predict(X_test)

# For Random Forest
y_pred_proba_rf = rf_pipeline.predict_proba(X_test)[:, 1]

# Predicting on the test set
y_pred_lr_best = lr_pipeline_best.predict(X_test)
y_pred_proba_lr_best = lr_pipeline_best.predict_proba(X_test)[:, 1]

# Predicting on the test set
y_pred_rf_best = rf_pipeline_best.predict(X_test)
y_pred_proba_rf_best = rf_pipeline_best.predict_proba(X_test)[:, 1]

# Predicting on the test set
y_pred_gb = gb_pipeline.predict(X_test)
y_pred_proba_gb = gb_pipeline.predict_proba(X_test)[:, 1]