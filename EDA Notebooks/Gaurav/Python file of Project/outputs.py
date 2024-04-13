from sklearn.metrics import confusion_matrix, roc_curve
import model
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

plt.figure(figsize=(14, 7))
model.df.corr()['Churn'].sort_values(ascending=False)


def distplot(feature, frame, color='r'):
    plt.figure(figsize=(8, 3))
    plt.title("Distribution for {}".format(feature))
    ax = sns.distplot(frame[feature])


plt.figure(figsize=(4, 3))
sns.heatmap(confusion_matrix(model.y_test, prediction_test),
            annot=True, fmt="d", linecolor="k", linewidths=3)

plt.title(" RANDOM FOREST CONFUSION MATRIX", fontsize=14)
plt.show()
y_rfpred_prob = model_rf.predict_proba(model.X_test)[:, 1]
fpr_rf, tpr_rf, thresholds = roc_curve(model.y_test, y_rfpred_prob)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_rf, tpr_rf, label='Random Forest', color="r")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest ROC Curve', fontsize=16)
plt.show();
plt.figure(figsize=(4, 3))
sns.heatmap(confusion_matrix(model.y_test, lr_pred),
            annot=True, fmt="d", linecolor="k", linewidths=3)

plt.title("LOGISTIC REGRESSION CONFUSION MATRIX", fontsize=14)
plt.show()
y_pred_prob = lr_model.predict_proba(model.X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(model.y_test, y_pred_prob)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Logistic Regression', color="r")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve', fontsize=16)
plt.show();
import matplotlib.pyplot as plt

# Model names
models = ['KNN', 'SVC', 'Random Forest', 'Logistic Regression', 'Decision Tree']

# Corresponding accuracies
accuracies = [0.704, 0.800, 0.793, 0.730, 0.704]

# Plotting the bar plot
plt.figure(figsize=(10, 6))
plt.bar(models, accuracies, color=['blue', 'orange', 'green', 'red', 'purple'])

# Adding labels and title
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Accuracy of Different Models')
plt.ylim(0, 1)  # Set the y-axis limit from 0 to 1
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

# Displaying the plot
plt.show()


# Generate synthetic data (you can replace this with your own dataset)
X, y = make_blobs(n_samples=100, centers=2, random_state=42)

# Create an SVM classifier
svm_classifier = SVC(kernel='linear', C=1.0)

# Fit the model to the data
svm_classifier.fit(X, y)

# Plot the decision boundary
xfit = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
yfit = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)
Xgrid, Ygrid = np.meshgrid(xfit, yfit)
Z = svm_classifier.decision_function(np.c_[Xgrid.ravel(), Ygrid.ravel()])
Z = Z.reshape(Xgrid.shape)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='autumn')
plt.contour(Xgrid, Ygrid, Z, colors='k', levels=[-1, 0, 1], alpha=0.5)
plt.xlabel("Featues")
plt.ylabel("Churn")
plt.title("SVM Classifier Decision Boundary")
plt.show()

# Plotting the feature importance of each feature
plt.figure(figsize=(12, 7))
plt.bar(model.X_train.columns, model_rf.feature_importances_ * 100, color='orange')
plt.xlabel('Features', fontsize=14)
plt.ylabel('Importance', fontsize=14)
plt.xticks(rotation=90)
plt.title('Feature Importance of each feature', fontsize=16)
# Using the provided feature importance data to create a comparison graph

# Features and their importances
features = ["TotalCharges", "MonthlyCharges", "tenure", "Contract", "PaymentMethod",
            "OnlineSecurity", "TechSupport", "InternetService", "gender", "OnlineBackup",
            "PaperlessBilling", "Partner", "MultipleLines", "SeniorCitizen", "DeviceProtection",
            "Dependents", "StreamingMovies", "StreamingTV", "PhoneService"]
importances = [0.18590472492273977, 0.1781053180579041, 0.1562856904959392, 0.08169260100242129,
               0.05042443343117293, 0.049066527428709136, 0.041871754510007964, 0.02926031462316956,
               0.0277477659533689, 0.026384681046870726, 0.02527880374588388, 0.023509568192995014,
               0.02242241857672751, 0.02115112171461009, 0.020881870265634737, 0.019636890571001483,
               0.01770687443240459, 0.017432630019875193, 0.0052360110085640284]

# Sort the features by importance
sorted_indices = np.argsort(importances)[::-1]
sorted_features = np.array(features)[sorted_indices]
sorted_importances = np.array(importances)[sorted_indices]

# Creating the plot
plt.figure(figsize=(12, 8))
plt.barh(sorted_features, sorted_importances, color='lightblue')
plt.xlabel('Importance')
plt.title('Feature Importances in Optimized Random Forest Model')
plt.gca().invert_yaxis()  # Invert y-axis to have the most important feature on top
plt.show()
