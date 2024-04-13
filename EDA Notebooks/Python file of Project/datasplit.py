import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, recall_score, confusion_matrix, precision_score, f1_score, \
    classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE


def load_data(filepath):
    return pd.read_csv(filepath)


def preprocess_data(df, target_column):
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_encoded = pd.get_dummies(X)
    return X_encoded, y


def split_data(X, y, test_size=0.2, random_state=42, stratify=None):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)


def standardize_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def train_knn_classifier(X_train, y_train):
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    return knn_model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    return y_pred


def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    return X_train_resampled, y_train_resampled


def logistic_regression(X_train, y_train):
    lr_model = LogisticRegression(max_iter=20000)
    lr_model.fit(X_train, y_train)
    return lr_model


def main():
    df = load_data('D:\\Telecom-Churn-Prediction\\Telecom Churn Prediction.csv')
    X, y = preprocess_data(df, 'Churn')
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.3, random_state=42)

    # Apply SMOTE
    X_train_resampled, y_train_resampled = apply_smote(X_train, y_train)

    # Standardize Data
    X_train_scaled, X_test_scaled = standardize_data(X_train_resampled, X_test)

    # Train logistic regression model
    lr_model = logistic_regression(X_train_scaled, y_train_resampled)

    # Evaluate logistic regression model
    print("Evaluation for Logistic Regression:")
    evaluate_model(lr_model, X_test_scaled, y_test)

    # Split for validation
    X_train_val, X_val, y_train_val, y_val = split_data(X_train_resampled, y_train_resampled, test_size=0.2)

    # Train and evaluate on validation set
    logistic_val_model = LogisticRegression()
    logistic_val_model.fit(X_train_val, y_train_val)
    y_val_pred = logistic_val_model.predict(X_val)

    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_classification_report = classification_report(y_val, y_val_pred)

    print(f"Validation Accuracy: {val_accuracy:.2f}")
    print("Classification Report:\n", val_classification_report)


if __name__ == "__main__":
    main()