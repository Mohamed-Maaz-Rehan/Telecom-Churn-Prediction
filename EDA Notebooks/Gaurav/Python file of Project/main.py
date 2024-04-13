import pandas as pd
import datasplit
import outputs

df = pd.read_csv('Telecom Churn Prediction.csv')
# Data loading and preprocessing
from datasplit import load_and_preprocess_data
X_train, X_test, y_train, y_test = load_and_preprocess_data()

# Model training and evaluation
from model import train_models
train_models(X_train, X_test, y_train, y_test)

# Output generation
from outputs import generate_outputs
generate_outputs()

# Accuracy calculation
from accuracies import calculate_accuracies
calculate_accuracies(X_test, y_test)

# Comparison of model accuracies
from comparison import compare_accuracies
compare_accuracies()