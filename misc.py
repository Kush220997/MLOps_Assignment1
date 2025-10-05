# src/misc.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error

def load_data():
    """
    Loads the Boston Housing dataset from the CMU repository
    as instructed in the assignment.
    Returns: DataFrame with feature columns + target (MEDV)
    """
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    feature_names = [
        "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
        "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"
    ]
    df = pd.DataFrame(data, columns=feature_names)
    df["MEDV"] = target
    return df

def preprocess(df):
    """
    Splits features and target from dataframe.
    Returns: X (DataFrame), y (Series)
    """
    X = df.drop(columns=["MEDV"])
    y = df["MEDV"]
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Performs train-test split.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def train_model(model, X_train, y_train):
    """
    Fits the provided model.
    """
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Returns Mean Squared Error on test data.
    """
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    return mse

