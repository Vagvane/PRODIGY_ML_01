import pandas as pd

def preprocess_train_data(df):
    df = df.fillna(0)
    features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
    X = df[features]
    y = df['SalePrice']
    return X, y

def preprocess_test_data(df):
    df = df.fillna(0)
    features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
    return df[features]
