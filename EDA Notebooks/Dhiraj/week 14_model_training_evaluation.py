
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
for i, predictor in enumerate(df.drop(columns=['churn', 'totalcharges', 'monthlycharges'])):
    plt.figure(i)
    sns.countplot(data=df, x=predictor, hue='churn')

df_updated['churn'] = np.where(df_updated.churn == 'Yes', 1, 0)

df_updated.head()

# Convert all the categorical variables into dummy variables
df_updated_dummies = pd.get_dummies(df_updated)
df_updated_dummies.head()

# Churn by Monthly charges
Mth = sns.kdeplot(df_updated_dummies.monthlycharges[(df_updated_dummies["churn"] == 0)],
                  color="Red", shade=True)
Mth = sns.kdeplot(df_updated_dummies.monthlycharges[(df_updated_dummies["churn"] == 1)],
                  ax=Mth, color="Blue", shade=True)
Mth.legend(["No Churn", "Churn"], loc='upper right')
Mth.set_ylabel('Density')
Mth.set_xlabel('Monthly Charges')
Mth.set_title('Monthly charges by churn')

# Churn by Total charges
Mth = sns.kdeplot(df_updated_dummies.totalcharges[(df_updated_dummies["churn"] == 0)],
                  color="Red", shade=True)
Mth = sns.kdeplot(df_updated_dummies.totalcharges[(df_updated_dummies["churn"] == 1)],
                  ax=Mth, color="Blue", shade=True)
Mth.legend(["No Churn", "Churn"], loc='upper right')
Mth.set_ylabel('Density')
Mth.set_xlabel('Total Charges')
Mth.set_title('Total charges by churn')

plt.figure(figsize=(19, 7))
df_updated_dummies.corr()['churn'].sort_values(ascending=False).plot(kind='bar')

plt.figure(figsize=(10, 10))
sns.heatmap(df_updated_dummies.corr(), cmap="Paired")

new_df1_target0 = df_updated.loc[df_updated["churn"] == 0]
new_df1_target1 = df_updated.loc[df_updated["churn"] == 1]


def uniplot(df, col, title, hue=None):
    sns.set_style('whitegrid')
    sns.set_context('talk')
    plt.rcParams["axes.labelsize"] = 20
    plt.rcParams['axes.titlesize'] = 22
    plt.rcParams['axes.titlepad'] = 30

    temp = pd.Series(data=hue)
    fig, ax = plt.subplots()
    width = len(df[col].unique()) + 7 + 4 * len(temp.unique())
    fig.set_size_inches(width, 8)
    plt.xticks(rotation=45)
    plt.yscale('log')
    plt.title(title)
    ax = sns.countplot(data=df, x=col, order=df[col].value_counts().index, hue=hue, palette='bright')

    plt.show()


uniplot(new_df1_target1, col='partner', title='Distribution of Gender for Churned Customers', hue='gender')

uniplot(new_df1_target0, col='partner', title='Distribution of Gender for Non Churned Customers', hue='gender')

uniplot(new_df1_target1, col='paymentmethod', title='Distribution of PaymentMethod for Churned Customers', hue='gender')

uniplot(new_df1_target1, col='contract', title='Distribution of Contract for Churned Customers', hue='gender')

uniplot(new_df1_target1, col='techsupport', title='Distribution of TechSupport for Churned Customers', hue='gender')

uniplot(new_df1_target1, col='seniorcitizen', title='Distribution of SeniorCitizen for Churned Customers', hue='gender')

sns.scatterplot(x='monthlycharges', y='totalcharges', data=df)

# selecting only categorical column
cat_cols = df.select_dtypes(include=['object']).columns
cat_cols

# dropping out target column from categorical
cat_cols = cat_cols.drop('churn')

cat_cols

sns.histplot(df['monthlycharges'], kde=True)

sns.histplot(df['tenure'], kde=True)

# Plot count plots for categorical columns
plt.figure(figsize=(25, 12))
sns.countplot(x='churn', data=df)
plt.title('Distribution of Categorical columns')
plt.show()

# Assuming 'Yes' is encoded as 1 and 'No' as 0 (instead of Hot-coding)
lr_pipeline.fit(X_train, y_train)

# Predicting on the test set
y_pred_lr = lr_pipeline.predict(X_test)

# predicting For Logistic Regression
y_pred_proba_lr = lr_pipeline.predict_proba(X_test)[:, 1]  # Getting probabilities for the positive class

# Creating a pipeline with preprocessing and the random forest model
rf_pipeline = make_pipeline(preprocessor, RandomForestClassifier(n_estimators=100))

# Training the model
rf_pipeline.fit(X_train, y_train)

# Predicting on the test set
y_pred_rf = rf_pipeline.predict(X_test)

# For Random Forest
y_pred_proba_rf = rf_pipeline.predict_proba(X_test)[:, 1]


def evaluate_model(y_true, y_pred, y_pred_proba):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label='Yes')  # Adjust if necessary
    recall = recall_score(y_true, y_pred, pos_label='Yes')
    f1 = f1_score(y_true, y_pred, pos_label='Yes')
    roc_auc = roc_auc_score(y_true, y_pred_proba)  # Use probabilities here

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")


# Evaluate Logistic Regression Model
print("Logistic Regression Performance:")
evaluate_model(y_test, y_pred_lr, y_pred_proba_lr)

# Evaluate Random Forest Model
print("\nRandom Forest Performance:")
evaluate_model(y_test, y_pred_rf, y_pred_proba_rf)

grid_search_lr.fit(X_train, y_train)

# Best parameters and best score
print("Best parameters for Logistic Regression:", grid_search_lr.best_params_)
print("Best ROC-AUC score from Grid Search for Logistic Regression:", grid_search_lr.best_score_)

# Defining the parameter grid for Random Forest
param_grid_rf = {
    'randomforestclassifier__n_estimators': [10, 50, 100, 200],
    'randomforestclassifier__max_depth': [None, 10, 20, 30]
}

# Create a GridSearchCV object
grid_search_rf = GridSearchCV(rf_pipeline, param_grid=param_grid_rf, cv=5, scoring='roc_auc', verbose=1)

# Fit it to the training data
grid_search_rf.fit(X_train, y_train)

# Best parameters and best score
print("Best parameters for Random Forest:", grid_search_rf.best_params_)
print("Best ROC-AUC score from Grid Search for Random Forest:", grid_search_rf.best_score_)

# Creating and training the Logistic Regression model with the best parameters
lr_pipeline_best = make_pipeline(preprocessor, LogisticRegression(C=10, solver='liblinear'))

# Training the model
lr_pipeline_best.fit(X_train, y_train)

# Predicting on the test set
y_pred_lr_best = lr_pipeline_best.predict(X_test)
y_pred_proba_lr_best = lr_pipeline_best.predict_proba(X_test)[:, 1]

# Evaluating the model
print("Logistic Regression Performance with Best Parameters:")
evaluate_model(y_test, y_pred_lr_best, y_pred_proba_lr_best)

# Creating and training the Random Forest model with the best parameters
rf_pipeline_best = make_pipeline(preprocessor, RandomForestClassifier(max_depth=10, n_estimators=100))

# Training the model
rf_pipeline_best.fit(X_train, y_train)

# Predicting on the test set
y_pred_rf_best = rf_pipeline_best.predict(X_test)
y_pred_proba_rf_best = rf_pipeline_best.predict_proba(X_test)[:, 1]

# Evaluating the model
print("\nRandom Forest Performance with Best Parameters:")
evaluate_model(y_test, y_pred_rf_best, y_pred_proba_rf_best)

# Creating a pipeline with preprocessing and the Gradient Boosting classifier
gb_pipeline = make_pipeline(preprocessor, GradientBoostingClassifier(random_state=42))

# Training the Gradient Boosting model
gb_pipeline.fit(X_train, y_train)

# Predicting on the test set
y_pred_gb = gb_pipeline.predict(X_test)
y_pred_proba_gb = gb_pipeline.predict_proba(X_test)[:, 1]

# Evaluating the Gradient Boosting model
print("Gradient Boosting Performance:")
evaluate_model(y_test, y_pred_gb, y_pred_proba_gb)

# Hyperparameter Tuning of Gradient Boosting Model
# Defining the parameter grid for Gradient Boosting
param_grid_gb = {
    'gradientboostingclassifier__n_estimators': [100, 200],
    'gradientboostingclassifier__learning_rate': [0.01, 0.1],
    'gradientboostingclassifier__max_depth': [3, 5]
}

# Create a GridSearchCV object for the Gradient Boosting pipeline
grid_search_gb = GridSearchCV(gb_pipeline, param_grid=param_grid_gb, cv=5, scoring='roc_auc', verbose=1)

# Fit it to the training data
grid_search_gb.fit(X_train, y_train)

# Best parameters and best score
print("Best parameters for Gradient Boosting:", grid_search_gb.best_params_)
print("Best ROC-AUC score from Grid Search for Gradient Boosting:", grid_search_gb.best_score_)

#
# #10-Fold Cross Validation

# 1. Logistic Regression with 10-Fold Cross-Validation
#Defining number of folds

cv_folds = 10

# Perform 10-fold cross-validation for Logistic Regression
scores_lr = cross_val_score(lr_pipeline, X, y, cv=cv_folds, scoring='roc_auc')

# Calculate average score
mean_score_lr = scores_lr.mean()
print("Average ROC-AUC for Logistic Regression:", mean_score_lr)

# 2. Random Forest with 10-Fold Cross-Validation

#Perform 10-fold cross-validation for Random Forest
scores_rf = cross_val_score(rf_pipeline, X, y, cv=cv_folds, scoring='roc_auc')

# Calculate average score
mean_score_rf = scores_rf.mean()
print("Average ROC-AUC for Random Forest:", mean_score_rf)

# 3. Gradient Boosting with 10-Fold Cross-Validation

# Perform 10-fold cross-validation for Gradient Boosting
scores_gb = cross_val_score(gb_pipeline, X, y, cv=cv_folds, scoring='roc_auc')

# Calculate average score
mean_score_gb = scores_gb.mean()
print("Average ROC-AUC for Gradient Boosting:", mean_score_gb)

