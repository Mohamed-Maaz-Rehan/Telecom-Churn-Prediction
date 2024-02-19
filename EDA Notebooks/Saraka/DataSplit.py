import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def dtreformat(df):
    df = df.drop((['gender','PhoneService','MultipleLines','InternetService','StreamingTV','StreamingMovies']), axis=1)
    df['TotalCharges']=df['TotalCharges'].replace({' ': 0})
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
    numeric_cols = df._get_numeric_data().columns
    categ_cols = list(set(df.columns) - set(numeric_cols))
    lb = LabelEncoder() 
    for i in categ_cols:
        df[i] = lb.fit_transform(df[i])
        
    return df


def datasplit(df,target):
    X = df.drop(columns=[target])
    Y = df[target]
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size =0.2, random_state=10) 
    print(X_train.shape)
    print(X_test.shape)
    print(Y_train.shape)
    print(Y_train.shape)
    
    return X_train, X_test, Y_train, Y_test


