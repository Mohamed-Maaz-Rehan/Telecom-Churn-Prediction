import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score

df = pd.read_csv('Telecom Churn Prediction.csv')

with open('encoder.pickle', 'rb') as f:
    enc = pickle.load(f)

with open('LogisticRegression(L1).pickle', 'rb') as f:
    model = pickle.load(f)

X = df.drop(columns=['Churn'])
Y = df['Churn']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=10)

# Re-fit the encoder on the current dataset

X_test_encoded = enc.transform(X_test)  # Transform X_test using the encoder

prediction = model.predict(X_test_encoded)

test_accuracy = accuracy_score(Y_test, prediction)  # Compare predictions with actual Y_test
test_precision = precision_score(Y_test, prediction)  # Compare predictions with actual Y_test

print("Test Precision:", test_precision)
print("Test Accuracy:", test_accuracy)
