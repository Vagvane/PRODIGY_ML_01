import pandas as pd

def load_train_data():
    return pd.read_csv("train.csv")

def load_test_data():
    return pd.read_csv("test.csv")