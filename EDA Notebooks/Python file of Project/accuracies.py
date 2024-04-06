from catboost import metrics
from sklearn.metrics import classification_report, precision_score, confusion_matrix, accuracy_score, recall_score, \
    f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import datasplit
import model

print(classification_report(datasplit.y_test, datasplit.predicted_y))
print(classification_report(datasplit.y_test, model.predictdt_y))

model_rf = RandomForestClassifier(n_estimators=500, oob_score=True, n_jobs=-1,
                                  random_state=50, max_features='sqrt',
                                  max_leaf_nodes=30)

model_rf.fit(datasplit.X_train, datasplit.y_train)

# Make predictions
prediction_test = model_rf.predict(datasplit.X_test)
print (metrics.accuracy_score(datasplit.y_test, prediction_test))
print(classification_report(datasplit.y_test, prediction_test))
lr_pred= lr_model.predict(datasplit.X_test)
report = classification_report(datasplit.y_test, lr_pred)
print(report)
print(classification_report(datasplit.y_test, model.predictdt_y))
# Define logistic regression model with adjusted class weights
logistic_regression_weighted = LogisticRegression(class_weight={0: 0.6809629219701162, 1: 1.8814984709480123}, max_iter=1000)

# Train the logistic regression model with adjusted class weights
logistic_regression_weighted.fit(datasplit.X_train_scaled, datasplit.y_train_resampled)

# Make predictions on the test set
y_pred_weighted = logistic_regression_weighted.predict(datasplit.X_test_scaled)

# Calculate precision
precision_weighted = precision_score(datasplit.y_test, y_pred_weighted)

# Print precision score
print("Precision Score (with adjusted class weights):", precision_weighted)

# Print confusion matrix
print("Confusion Matrix (with adjusted class weights):")
print(confusion_matrix(datasplit.y_test, y_pred_weighted))

# Print classification report
print("Classification Report (with adjusted class weights):")
print(classification_report(datasplit.y_test, y_pred_weighted))

from sklearn.tree import DecisionTreeClassifier

# Define decision tree classifier
decision_tree = DecisionTreeClassifier(random_state=42)

# Train the decision tree classifier with resampled data
decision_tree.fit(datasplit.X_train_resampled, datasplit.y_train_resampled)

# Make predictions on the test set
y_pred = decision_tree.predict(datasplit.X_test_scaled)

# Calculate precision
precision = precision_score(datasplit.y_test, y_pred)

# Print precision score
print("Precision Score:", precision)

# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(datasplit.y_test, y_pred))

# Print classification report
print("Classification Report:")
print(classification_report(datasplit.y_test, y_pred))


# Define Random Forest classifier
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the Random Forest classifier with resampled data
random_forest.fit(datasplit.X_train_resampled, datasplit.y_train_resampled)

# Make predictions on the test set
y_pred = random_forest.predict(datasplit.X_test_scaled)  # Assuming X_test_scaled is already scaled

# Calculate precision
precision = precision_score(datasplit.y_test, y_pred)

# Print precision score
print("Precision Score:", precision)

# Print confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(datasplit.y_test, y_pred))

# Print classification report
print("Classification Report:")
print(classification_report(datasplit.y_test, y_pred))


# Initialize the model with the best parameters
optimized_rf = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=2,
                                      min_samples_leaf=1, max_features='sqrt', random_state=42)

# Train the model
optimized_rf.fit(datasplit.X_train, datasplit.y_train)

# Make predictions
y_pred_optimized = optimized_rf.predict(datasplit.X_test)

# Evaluate the model
accuracy = accuracy_score(datasplit.y_test, y_pred_optimized)
precision = precision_score(datasplit.y_test, y_pred_optimized)
recall = recall_score(datasplit.y_test, y_pred_optimized)
f1 = f1_score(datasplit.y_test, y_pred_optimized)

print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
