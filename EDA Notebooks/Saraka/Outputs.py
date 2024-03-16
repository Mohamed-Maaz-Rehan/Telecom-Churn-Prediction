import Models


def unbalanced(X_train, Y_train, X_test, Y_test):
    print("\n-------------Unbalanced------------------\n")
    results_df = Models.models(X_train, Y_train, X_test, Y_test)
    print(results_df.to_string())
    return results_df


def oversampled(X_oversampled, y_oversampled, X_test, Y_test):
    print("\n-------------Oversampled------------------\n")
    results_df = Models.models(X_oversampled, y_oversampled, X_test, Y_test)
    print(results_df.to_string())
    return results_df


def undersampled(X_undersampled, y_undersampled, X_test, Y_test):
    print("\n-------------Undersampled-----------------\n")
    results_df = Models.models(X_undersampled, y_undersampled, X_test, Y_test)
    print(results_df.to_string())
    return results_df


def combined(X_combined, y_combined, X_test, Y_test):
    print("\n-------------Combined------------------\n")
    results_df = Models.models(X_combined, y_combined, X_test, Y_test)
    print(results_df.to_string())
    return results_df
