import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import Output
import Datasplit
import Model
from Model import models

# Load dataset
df = pd.read_csv('/Users/HP/ML_Projects/telecom.csv')
dff = Datasplit.dtreformat(df)
# Preprocess data and split into training and testing sets
X = dff.drop(columns=['Churn'])
y = dff['Churn']
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Oversample using SMOTE
smote = SMOTE()
X_oversampled, Y_oversampled = smote.fit_resample(X_train_scaled, Y_train)

# Perform hyperparameter tuning and evaluate models
outputs = [
    ("Logistic Regression", Output.hyperparameter_tuning_lr(X_oversampled, Y_oversampled)),
    ("Random Forest", Output.hyperparameter_tuning_rf(X_oversampled, Y_oversampled)),
    ("Decision Tree", Output.hyperparameter_tuning_dt(X_oversampled, Y_oversampled)),
    ("SVM", Output.hyperparameter_tuning_svm(X_oversampled, Y_oversampled))
]


print(Output.hyperparameter_tuning_lr())
print(Output.hyperparameter_tuning_dt())
print(Output.hyperparameter_tuning_svm())
print(Output.hyperparameter_tuning_rf())
# Call the models function with the specified classifiers
results_df = models(X_oversampled, Y_oversampled, X_test_scaled, Y_test)

# Print the results
print(results_df)
