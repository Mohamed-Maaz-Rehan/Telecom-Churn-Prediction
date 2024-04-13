from catboost import metrics
from sklearn.metrics import classification_report, precision_score, confusion_matrix, accuracy_score, recall_score, \
    f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import datasplit
import model

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name='Model'):
    # Train the model
    model.fit(X_train, y_train)
    # Make predictions
    y_pred = model.predict(X_test)
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Print the evaluation results
    print(
        f"{model_name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

def evaluate_models():
    # Evaluate initial models
    print(classification_report(datasplit.y_test, datasplit.predicted_y))
    print(classification_report(datasplit.y_test, model.predictdt_y))

    model_rf = RandomForestClassifier(n_estimators=500, oob_score=True, n_jobs=-1,
                                  random_state=50, max_features='sqrt',
                                  max_leaf_nodes=30)
    train_and_evaluate_model(model_rf, datasplit.X_train, datasplit.y_train, datasplit.X_test, datasplit.y_test,
                             'Random Forest')


# Define and evaluate Logistic Regression
    lr_model = LogisticRegression()
    train_and_evaluate_model(lr_model, datasplit.X_train, datasplit.y_train, datasplit.X_test, datasplit.y_test, 'Logistic Regression')

    # SVM Classifier
    svm_model = SVC(kernel='linear')
    train_and_evaluate_model(svm_model, datasplit.X_train, datasplit.y_train, datasplit.X_test, datasplit.y_test, 'SVM')

    # KNN Classifier
    knn_model = KNeighborsClassifier(n_neighbors=5)
    train_and_evaluate_model(knn_model, datasplit.X_train, datasplit.y_train, datasplit.X_test, datasplit.y_test, 'KNN')

    # Decision Tree Classifier
    dt_model = DecisionTreeClassifier(random_state=42)
    train_and_evaluate_model(dt_model, datasplit.X_train, datasplit.y_train, datasplit.X_test, datasplit.y_test, 'Decision Tree')


if __name__ == "__main__":
    evaluate_models()
