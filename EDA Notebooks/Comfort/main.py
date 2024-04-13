import pandas as pd
from datapreprocessing import preprocess_data, preprocess_and_split_data
from model_training import train_models, evaluate_models
import os

def main():
    


    # Read the data from CSV
    df = pd.read_csv('telecom1.csv')
    
    # Preprocess the data
    processed_df = preprocess_data(df)
    
    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = preprocess_and_split_data(processed_df) 
    
    # Train models
    trained_models = train_models(X_train, y_train)
    
    # Evaluate models on both train and test sets
    results_train, results_test = evaluate_models(trained_models, X_train, y_train, X_test, y_test)
    
    # Store results in DataFrames
    results_train_df = pd.DataFrame(results_train).transpose()
    results_test_df = pd.DataFrame(results_test).transpose()
    
    # Print the results
    print("Evaluation Results on Training Data:")
    print(results_train_df)
    print("\nEvaluation Results on Test Data:")
    print(results_test_df)

    with pd.ExcelWriter('evaluation_results.xlsx') as writer:
        results_train_df.to_excel(writer, sheet_name='Training Data')
        results_test_df.to_excel(writer, sheet_name='Test Data')

        os.getcwd()

if __name__ == "__main__":
    main()
