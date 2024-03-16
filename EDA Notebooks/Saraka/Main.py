import pandas as pd
import DataSplit
import Outputs

df = pd.read_csv('Telecom Churn Prediction.csv')

ftData = DataSplit.dtreformat(df)
X_train, X_test, Y_train, Y_test = DataSplit.datasplit(ftData)
X_train, X_test = DataSplit.dtscale(X_train, X_test)
X_oversampled, y_oversampled = DataSplit.smote(X_train, Y_train)
X_undersampled, y_undersampled = DataSplit.RUS(X_train, Y_train)
X_combined, y_combined = DataSplit.combine(X_train, Y_train)

outputs = [
    Outputs.oversampled(X_oversampled, y_oversampled, X_test, Y_test),
    Outputs.undersampled(X_undersampled, y_undersampled, X_test, Y_test),
    Outputs.unbalanced(X_train, Y_train, X_test, Y_test),
    Outputs.combined(X_combined, y_combined, X_test, Y_test)
]

for i in outputs:
    i
