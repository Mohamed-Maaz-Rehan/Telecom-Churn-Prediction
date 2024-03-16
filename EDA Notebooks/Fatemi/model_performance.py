import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

class ModelTraining:
    def __init__(self, data_path):
        self.data_path = data_path
        self.models = {
            "Logistic Regression": LogisticRegression(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
            "SVM": SVC(),
            "KNN": KNeighborsClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42),
            "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42)
        }
        self.X_train, self.X_test, self.y_train, self.y_test = self.prepare_data()
    
    def read_and_preprocess_data(self):
        df = pd.read_csv(self.data_path)
        df.drop("customerID", axis=1, inplace=True)
        # Define a mapping dictionary
        mappings = [
            ('gender', {'Male': 0, 'Female': 1}),
            ('MultipleLines', {'No': 0, 'Yes': 1, 'No phone service': 2}),
            ('Contract', {'Month-to-month': 0, 'One year': 1, 'Two year': 2}),
            ('InternetService', {'No': 0, 'DSL': 1, 'Fiber optic': 2}),
            ('PaymentMethod', {'Electronic check': 0, 'Mailed check': 1, 'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3})
        ]
        for col, mapping in mappings:
            df[col] = df[col].map(mapping)

        df['TotalCharges'] = df['TotalCharges'].replace({' ': 0})
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])

        binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
        multi_state_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

        for col in binary_cols + multi_state_cols:
            df[col] = df[col].map({'No': 0, 'Yes': 1, 'No internet service': 2})

        # Using min max scaler technique on tenure, monthlycharges and total charges column
        scaler = MinMaxScaler()
        df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(df[['tenure', 'MonthlyCharges', 'TotalCharges']])

        return df
    
    def prepare_data(self):
        df = self.read_and_preprocess_data()
        X = df.drop("Churn", axis=1)
        y = df['Churn']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=100)
        
        # Performing oversampling
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        
        return X_train, X_test, y_train, y_test
    
    def train_and_evaluate(self):
        for name, model in self.models.items():
            model.fit(self.X_train, self.y_train)
            train_predictions = model.predict(self.X_train)
            test_predictions = model.predict(self.X_test)

            print(f"Results for {name}:")
            print("Train Metrics:")
            self.print_metrics(self.y_train, train_predictions)
            print("Test Metrics:")
            self.print_metrics(self.y_test, test_predictions)
            print("-" * 50)
    
    def print_metrics(self, true_values, predictions):
        accuracy = accuracy_score(true_values, predictions)
        precision = precision_score(true_values, predictions)
        recall = recall_score(true_values, predictions)
        f1 = f1_score(true_values, predictions)

        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Precision: {precision * 100:.2f}%")
        print(f"Recall: {recall * 100:.2f}%")
        print(f"F1 Score: {f1 * 100:.2f}%")
        print(f"Confusion Matrix: \n{confusion_matrix(true_values, predictions)}")

if __name__ == "__main__":
    data_path = "Telecom Churn Prediction.csv”
    model_trainer = ModelTraining(data_path)
    model_trainer.train_and_evaluate()
