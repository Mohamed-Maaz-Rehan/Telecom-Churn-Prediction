import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import make_pipeline as make_pipeline_imb


def load_data(filepath):
    return pd.read_csv(filepath)

def clean_data(df):
        # Drop duplicates
        df.drop_duplicates(inplace=True)

        # or impute missing values depending on your dataset and the importance of the column
        for col in df.columns:
            if df[col].dtype == "object":
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)

        return df


def preprocess_features(df, target_column):
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Identify categorical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns

    # Create a transformer for categorical features
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine transformer into a preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='passthrough'  # keep other columns untouched
    )

    return preprocessor, X, y


def train_and_evaluate_models(filepath, target_column):
    df = load_data(filepath)
    preprocessor, X, y = preprocess_features(df, target_column)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=10000),
        "SVM": SVC(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }
    # Iterate over classifiers and train each
    for name, classifier in classifiers.items():
        pipeline = make_pipeline_imb(
            preprocessor,
            SMOTE(random_state=42),
            classifier
        )

        # Train the classifier
        pipeline.fit(X_train, y_train)

        # Make predictions
        y_pred = pipeline.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {accuracy:.4f}")
        print(f"{name} Classification Report:")
        print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    train_and_evaluate_models('D:\\Telecom-Churn-Prediction\\Telecom Churn Prediction.csv', 'Churn')