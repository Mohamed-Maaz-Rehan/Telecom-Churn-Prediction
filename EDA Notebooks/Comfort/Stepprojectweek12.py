
df['PhoneService'] = df['PhoneService'].astype(int)
df['Partner'] = df['Partner'].astype(int)
df['gender'] = df['gender'].astype(int)
df['Dependents'] = df['Dependents'].astype(int)
df['MultipleLines'] = df['MultipleLines'].astype(int)
df['OnlineSecurity'] = df['OnlineSecurity'].astype(int)
df['OnlineBackup'] = df['OnlineBackup'].astype(int)
df['DeviceProtection'] = df['DeviceProtection'].astype(int)
df['TechSupport'] = df['TechSupport'].astype(int)
df['StreamingTV'] = df['StreamingTV'].astype(int)
df['StreamingMovies'] = df['StreamingMovies'].astype(int)
df['Contract'] = df['Contract'].astype(int)
df['PaperlessBilling'] = df['PaperlessBilling'].astype(int)
df['PaymentMethod'] = df['PaymentMethod'].astype(int)
df['Churn'] = df['Churn'].astype(int)

df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
df['TotalCharges'] = df['TotalCharges'].astype(float)
df.fillna(df["TotalCharges"].mean(),inplace=True)

X_col = df.columns[df.columns != 'Churn'].tolist()
y_col = 'Churn'
X = df[X_col]
y = df[y_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
knn = KNeighborsClassifier(n_neighbors=5)
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Support Vector Machine': SVM(),
    'K-Nearest Neighbors': KNeighborsClassifier()
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy

ada = AdaBoostClassifier(n_estimators=100, random_state=42)
ada.fit(X_train, y_train)


gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

bagging = BaggingClassifier(n_estimators=100, random_state=42)
bagging.fit(X_train, y_train)

adaboost_pred = ada.predict(X_test)


gb_pred = gb.predict(X_test)
rf_pred = rf.predict(X_test)

bagging_pred = bagging.predict(X_test)
# Accuracy scores
ada_accuracy = accuracy_score(y_test, adaboost_pred)
gb_accuracy = accuracy_score(y_test, gb_pred)
rf_accuracy = accuracy_score(y_test, rf_pred)
bagging_accuracy = accuracy_score(y_test, bagging_pred)

param_dist = {
    'n_estimators': randint(100, 500),  
    'max_depth': [None, 10, 20],         
    'min_samples_split': randint(2, 20)  
}

model = RandomForestClassifier()

# Perform randomized search
randomized_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=10, cv=5, scoring='accuracy')
randomized_search.fit(X_train, y_train)

# Best hyperparameters
best_params_random = randomized_search.best_params_

best_max_depth = 10
best_min_samples_split = 13
best_n_estimators = 259

Last_rf_model = RandomForestClassifier(max_depth=best_max_depth, 
                                        min_samples_split=best_min_samples_split,
                                        n_estimators=best_n_estimators)

# Training the last Random Forest model on the entire training dataset
Last_rf_model.fit(X_train, y_train)

y_pred = Last_rf_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
classifiers = [
        {
        'name': 'Support Vector Machine',
        'classifier': SVM(),
        'param_distributions': {
            'C': uniform(loc=0, scale=10),
            'gamma': ['scale', 'auto'],
            'kernel': ['linear', 'rbf', 'poly']
        }
        }
] 

# Perform hyperparameter tuning for each classifier
best_models = {}
for classifier in classifiers:
    print(f"Hyperparameter tuning for {classifier['name']}...")
    random_search = RandomizedSearchCV(estimator=classifier['classifier'], 
                                       param_distributions=classifier['param_distributions'], 
                                       n_iter=100, 
                                       cv=5, 
                                       scoring='accuracy', 
                                       random_state=42)
    random_search.fit(X_train, y_train)
    best_models[classifier['name']] = {
        'best_estimator': random_search.best_estimator_,
        'best_params': random_search.best_params_,
        'best_score': random_search.best_score_
    }
for name, model_info in best_models.items():
    best_model = model_info['best_estimator']
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

def perform_cross_validation(model, X, y, n_splits=10):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
    
    print("Cross-Validation Scores:")
    for fold, score in enumerate(cv_scores, start=1):
        print(f"Fold {fold}: {score}")
    
    avg_cv_score = cv_scores.mean()

models = [
    LogisticRegression(),
    SVM(),

]


for model in models:
    perform_cross_validation(model, X, y)


explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values,
                                                   feature_names=X_train.columns.tolist(),
                                                   class_names=['Not Churn', 'Churn'],
                                                   discretize_continuous=True)


instance = X_test.iloc[6]

explanation = explainer.explain_instance(instance.values, model.predict_proba)

# Visualize the explanation
explanation.show_in_notebook()

baseline_accuracy = accuracy_score(y_test, model.predict(X_test))

def sensitivity_analysis(model, X_test, feature_name, values_to_test):
    sensitivities = []
    for value in values_to_test:
        X_test_copy = X_test.copy()
        X_test_copy[feature_name] = value
        predictions = model.predict(X_test_copy)
        sensitivity = accuracy_score(y_test, predictions) - baseline_accuracy
        sensitivities.append(sensitivity)
    return sensitivities

values_to_test = np.linspace(0, 1, 11)  

sensitivity_results = {}
for feature in X.columns:
    sensitivities = sensitivity_analysis(model, X_test, feature, values_to_test)
    sensitivity_results[feature] = sensitivities

for feature, sensitivities in sensitivity_results.items():
    print(f"Sensitivity analysis for {feature}:")
    for value, sensitivity in zip(values_to_test, sensitivities):
        print(f"   {feature}={value}: Change in accuracy = {sensitivity:.4f}")


