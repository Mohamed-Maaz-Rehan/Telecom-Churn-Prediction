import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

class TelecomChurnPredictor:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
    
    def preprocess_data(self):
        self.df.drop("customerID", axis=1, inplace=True)
        self.df['gender'] = self.df['gender'].map({'Male': 0, 'Female': 1})
        self.df['MultipleLines'] = self.df['MultipleLines'].map({'No': 0, 'Yes': 1, 'No phone service': 2})
        self.df['Contract'] = self.df['Contract'].map({'Month-to-month': 0, 'One year': 1, 'Two year': 2})
        self.df['InternetService'] = self.df['InternetService'].map({'No': 0, 'DSL': 1, 'Fiber optic': 2})
        self.df['PaymentMethod'] = self.df['PaymentMethod'].map({'Electronic check': 0, 'Mailed check': 1, 'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3})
        self.df['TotalCharges'] = self.df['TotalCharges'].replace({' ': 0})
        self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'])
        for col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']:
            self.df[col] = self.df[col].map({'No': 0, 'Yes': 1})
        for col in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']:
            self.df[col] = self.df[col].map({'No': 0, 'Yes': 1, 'No internet service': 2})
    
    def scale_data(self):
        scaler = MinMaxScaler()
        self.df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(self.df[['tenure', 'MonthlyCharges', 'TotalCharges']])
    
    def split_data(self):
        X = self.df.drop("Churn", axis=1)
        y = self.df['Churn']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=100)
        return X_train, X_test, y_train, y_test
    
    def oversample_data(self, X_train, y_train):
        smote = SMOTE(random_state=42)
        X_oversampled, y_oversampled = smote.fit_resample(X_train, y_train)
        return X_oversampled, y_oversampled
    
    def train_logistic_regression(self, X_train, y_train, X_test, y_test):
        logistic = LogisticRegression(max_iter=1000)
        logistic.fit(X_train, y_train)
        predict = logistic.predict(X_test)
        accuracy = accuracy_score(y_test, predict)
        print(f"Testing Accuracy (Logistic Regression): {accuracy * 100:.2f}%")
    
    def train_decision_tree(self, X_train, y_train, X_test, y_test):
        decision_tree = DecisionTreeClassifier()
        decision_tree.fit(X_train, y_train)
        predict = decision_tree.predict(X_test)
        accuracy = accuracy_score(y_test, predict)
        print(f"Testing Accuracy (Decision Tree): {accuracy * 100:.2f}%")
    
    def train_random_forest(self, X_train, y_train, X_test, y_test):
        random_forest = RandomForestClassifier(n_estimators=100)
        random_forest.fit(X_train, y_train)
        predict = random_forest.predict(X_test)
        accuracy = accuracy_score(y_test, predict)
        print(f"Testing Accuracy (Random Forest): {accuracy * 100:.2f}%")
    
    def train_svc(self, X_train, y_train, X_test, y_test):
        svc = SVC()
        svc.fit(X_train, y_train)
        predict = svc.predict(X_test)
        accuracy = accuracy_score(y_test, predict)
        print(f"Testing Accuracy (SVC): {accuracy * 100:.2f}%")
    
    def train_knn(self, X_train, y_train, X_test, y_test):
        knn = KNeighborsClassifier()
        knn.fit(X_train, y_train)
        predict = knn.predict(X_test)
        accuracy = accuracy_score(y_test, predict)
        print(f"Testing Accuracy (KNN): {accuracy * 100:.2f}%")


telecom_predictor = TelecomChurnPredictor("Telecom Churn Prediction.csv")
telecom_predictor.preprocess_data()
telecom_predictor.scale_data()
X_train, X_test, y_train, y_test = telecom_predictor.split_data()
X_oversampled, y_oversampled = telecom_predictor.oversample_data(X_train, y_train)
telecom_predictor.train_logistic_regression(X_oversampled, y_oversampled, X_test, y_test)
telecom_predictor.train_decision_tree(X_oversampled, y_oversampled, X_test, y_test)
telecom_predictor.train_random_forest(X_oversampled, y_oversampled, X_test, y_test)
telecom_predictor.train_svc(X_oversampled, y_oversampled, X_test, y_test)
telecom_predictor.train_knn(X_oversampled, y_oversampled, X_test, y_test)
