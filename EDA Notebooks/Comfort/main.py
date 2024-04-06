import pandas as pd
from datapreprocessing import preprocess_data, preprocess_and_split_data
from model_training import train_models, evaluate_models

import os

os.chdir(r'C:\Users\COMFORT\Documents\LOYALIST COLLEGE')

def main():
        
    df = pd.read_csv('telecom1.csv')
    
    processed_df = preprocess_data(df)
    
    X_train, X_test, y_train, y_test = preprocess_and_split_data(processed_df) 
    
    trained_models = train_models(X_train, y_train)
    
    results = evaluate_models(trained_models, X_test, y_test)
    
    for name, metrics in results.items():
        print(f'Model: {name}')
        print(f'Accuracy: {metrics["Accuracy"]:.2f}')
        print(f'Precision: {metrics["Precision"]:.2f}')
        print(f'Recall: {metrics["Recall"]:.2f}')
        print(f'F1 Score: {metrics["F1 Score"]:.2f}')
        if metrics["ROC AUC"] is not None:
            print(f'ROC AUC: {metrics["ROC AUC"]:.2f}')
        print('Classification Report:')
        print(metrics['Classification Report'])
        print('\n')

if __name__ == "__main__":
    main()
