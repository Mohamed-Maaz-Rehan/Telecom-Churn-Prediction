import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from lime.lime_tabular import LimeTabularExplainer
import shap
import matplotlib.pyplot as plt

class TelecomChurnPredictor:
    def __init__(self, file_path):
        self.df = pd.read_csv(file_path)
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = MinMaxScaler()
        self.preprocess_data()
        
    def preprocess_data(self):
        self.df.drop("customerID", axis=1, inplace=True)
        mapping_dicts = {
            'gender': {'Male': 0, 'Female': 1},
            'MultipleLines': {'No': 0, 'Yes': 1, 'No phone service': 2},
            'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2},
            'InternetService': {'No': 0, 'DSL': 1, 'Fiber optic': 2},
            'PaymentMethod': {'Electronic check': 0, 'Mailed check': 1, 'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3},
            'TotalCharges': {' ': 0}
        }
        
        for col, mapping in mapping_dicts.items():
            self.df[col] = self.df[col].map(mapping)
        
        self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'])
        
        binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
        for col in binary_cols + ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']:
            self.df[col] = self.df[col].map({'No': 0, 'Yes': 1, 'No internet service': 2})
        
        self.df[['tenure', 'MonthlyCharges', 'TotalCharges']] = self.scaler.fit_transform(self.df[['tenure', 'MonthlyCharges', 'TotalCharges']])
        
    def split_data(self):
        X = self.df.drop("Churn", axis=1)
        y = self.df['Churn']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.7, random_state=100)
        
    def balance_data(self):
        smote = SMOTE(random_state=42)
        self.X_train, self.y_train = smote.fit_resample(self.X_train, self.y_train)
        
    def train_model(self):
        self.model = LogisticRegression()
        self.model.fit(self.X_train, self.y_train)
        
    def predict_and_evaluate(self):
        predictions = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)
        print(f"Testing Accuracy: {accuracy * 100:.2f}%")
        
    def explain_prediction(self, idx):
        feature_names = self.X_train.columns.tolist()
        class_names = [str(cn) for cn in self.y_train.unique().tolist()]
        
        explainer = LimeTabularExplainer(self.X_train.values, feature_names=feature_names, class_names=class_names, discretize_continuous=True)
        exp = explainer.explain_instance(self.X_test.iloc[idx].values, self.model.predict_proba)
        exp.show_in_notebook(show_all=False)
        
        shap_explainer = shap.Explainer(self.model, self.X_train)
        shap_values = shap_explainer.shap_values(self.X_test.iloc[[idx]])
        shap.initjs()
        shap.force_plot(shap_explainer.expected_value, shap_values, self.X_test.iloc[[idx]])
        
    def global_interpretation(self):
        explainer = shap.Explainer(self.model, self.X_train)
        shap_values = explainer.shap_values(self.X_test)
        plt.figure()
        shap.summary_plot(shap_values, self.X_test)
        shap.dependence_plot("tenure", shap_values, self.X_test)
        

predictor = TelecomChurnPredictor("Telecom Churn Prediction.csv")
predictor.split_data()
predictor.balance_data()
predictor.train_model()
predictor.predict_and_evaluate()
# Use a valid index in place of idx for explanation
# predictor.explain_prediction(idx)
predictor.global_interpretation()
