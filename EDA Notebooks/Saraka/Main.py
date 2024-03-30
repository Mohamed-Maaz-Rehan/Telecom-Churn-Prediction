import pandas as pd
import DataSplit
import Outputs

df = pd.read_csv('Telecom Churn Prediction.csv')

ftData = DataSplit.dtreformat(df)
X_train, X_test, Y_train, Y_test = DataSplit.datasplit(ftData)
X_train, X_test = DataSplit.dtscale(X_train, X_test)
X_oversampled, y_oversampled = DataSplit.smote(X_train, Y_train)
X_undersampled, y_undersampled = DataSplit.RUS(X_train, Y_train)
#X_combined, y_combined = DataSplit.combine(X_train, Y_train)

outputs = [
    #(Outputs.metrics(X_oversampled, y_oversampled, X_test, Y_test, "Oversampled")),
    #(Outputs.metrics(X_undersampled, y_undersampled, X_test, Y_test, "Undersampled")),
    (Outputs.metrics(X_train, Y_train, X_test, Y_test, "Unbalanced")),
    #(Outputs.metrics(X_combined, y_combined, X_test, Y_test, "Combined"))
]

for i in outputs:
    results_df = i
    #Outputs.visualize_metrics(results_df)
